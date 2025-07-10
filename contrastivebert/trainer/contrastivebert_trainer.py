import json
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from contrastivebert.classifier.contrastivebert_classifier import ContrastiveSBERT
from contrastivebert.classifier.contrastivebert_configuration import ModelConf, OptimizerConf, LrComputerConf, TrainerConf
from contrastivebert.trainer.contrastivebert_dataset import ContrastiveBertDataset
from utils.pt_utils import run_gc
from utils.lm_core import instantiate_ml_components
from utils.lr_computer_with_decay import LRComputerWithDecay

OOM_LIMIT: int = 8


def initialize_modules() -> tuple[ContrastiveSBERT, optim.AdamW, _Loss, LRComputerWithDecay]:
    model_conf = ModelConf()
    optimizer_conf = OptimizerConf()
    lr_conf = LrComputerConf()

    model = ContrastiveSBERT(model_conf)
    optimizer = optim.AdamW(model.parameters(), betas=optimizer_conf.betas, weight_decay=optimizer_conf.weight_decay)
    criterion: _Loss = nn.CosineEmbeddingLoss(margin=0.5)
    lr_wd = LRComputerWithDecay(lr_conf)

    return model, optimizer, criterion, lr_wd


def identity_collate(batch: list[dict[str, Any]]):
    # batch: list[dict[str, Any]]
    # simply return the List without collating it, so that `batch` inside the loop is a list of examples
    return batch


class Trainer:
    def __init__(self, df: pd.DataFrame, trainer_conf: TrainerConf):
        self.df = df
        self.trainer_conf = trainer_conf

        self.ml_components = instantiate_ml_components()

        self.train_dataset, self.test_dataset = self.prepare_dataset()
        self.model, self.optimizer, self.criterion, self.lr_wd = initialize_modules()

    def prepare_dataset(self) -> tuple[Dataset, Dataset]:
        # Splitting the data
        train_df, test_df = train_test_split(
            self.df,
            test_size=self.trainer_conf.dataset_split_ratio,
            random_state=self.trainer_conf.dataset_random_state
        )

        train_dataset = ContrastiveBertDataset(train_df)
        test_dataset = ContrastiveBertDataset(test_df)
        return train_dataset, test_dataset

    def train(self) -> None:

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.trainer_conf.batch_size,
            shuffle=True,
            collate_fn=identity_collate,
        )

        self.model.train()
        for epoch in tqdm(range(self.trainer_conf.epochs), desc='Training Epochs'):
            total_loss: float = 0.0

            # Compute Learning Rate per epoch and set it in the AdamW optimizer
            lr = self.lr_wd.compute(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1} Batches', leave=False):
                # Process all Flower Records in one pass
                all_hfd_ids = torch.stack([ex['input_ids_hfd'] for ex in batch], dim=0)
                all_hfd_mask = torch.stack([ex['attention_mask_hfd'] for ex in batch], dim=0)
                all_hfd_ids = all_hfd_ids.to(self.ml_components.tensor_device)
                all_hfd_mask = all_hfd_mask.to(self.ml_components.tensor_device)
                emb_hfd = self.model(all_hfd_ids, all_hfd_mask)  # (N, H)

                # flower_name&flower_structure
                # Flatten list[list[Tensor]] into list[Tensor] for all flowers across the batch
                all_nas_ids: list[torch.Tensor] = []
                all_nas_masks: list[torch.Tensor] = []
                counts: list[int] = []

                for ex in batch:
                    nas_ids_list: list[torch.Tensor] = ex['input_ids_nas_list'][:OOM_LIMIT]
                    nas_masks_list: list[torch.Tensor] = ex['attention_mask_nas_list'][:OOM_LIMIT]
                    counts.append(len(nas_ids_list))
                    all_nas_ids.extend(nas_ids_list)
                    all_nas_masks.extend(nas_masks_list)

                # Stack list[Tensor] vertically into two tensors of shape (total_flowers, seq_len)
                stacked_nas_ids: torch.Tensor = torch.stack(all_nas_ids, dim=0).to(self.ml_components.tensor_device)
                stacked_nas_masks: torch.Tensor = torch.stack(all_nas_masks, dim=0).to(self.ml_components.tensor_device)

                # One forward pass for all flowers → (total_flowers, hidden_dim)
                stacked_nas_embs: torch.Tensor = self.model(stacked_nas_ids, stacked_nas_masks)

                # Split back by example and mean‐pool each chunk → List of (hidden_dim,)
                agg_embs: list[torch.Tensor] = []
                offset: int = 0
                for c in counts:
                    slice_emb: torch.Tensor = stacked_nas_embs[offset: offset + c]  # (c, hidden_dim)
                    pooled_emb: torch.Tensor = slice_emb.mean(dim=0)  # (hidden_dim,)
                    agg_embs.append(pooled_emb)
                    offset += c

                # Stack into final (batch_size, hidden_dim) tensor
                emb_nas: torch.Tensor = torch.stack(agg_embs, dim=0)  # (batch, hidden_dim)

                # contrastive loss
                # Labels: +1 for positive pairs (since every row is matched)
                target = torch.ones(emb_nas.shape[0]).to(device=self.ml_components.tensor_device)

                loss = self.criterion(emb_nas, emb_hfd, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                del all_hfd_ids, all_hfd_mask
                del agg_embs, stacked_nas_ids, stacked_nas_masks, counts
                run_gc()


            print(f'Epoch: {epoch + 1:>3}/{self.trainer_conf.epochs:>3}, '
                  f'LR:{lr:.5f}, Loss:{total_loss / len(train_loader):>4.5f}')

    def evaluate(self, fqfn_metrics: Optional[str] = None) -> dict[str, float]:
        """
        Simplified evaluation for a Siamese/contrastive SBERT:

        1. Always compute the average CosineEmbeddingLoss over held‐out positive pairs.
        2. If with_negatives=True, also sample an equal number (or negative_ratio × #positives) of negative pairs,
           compute cosine similarities on both sets, then report ROC-AUC. 

        Returns a dict with keys:
          - 'avg_pos_loss'       : float
          - 'avg_pos_cosine'     : float   (mean cosine similarity on positives)
          - 'avg_neg_cosine'     : float   (if with_negatives=True)
          - 'roc_auc'            : float   (if with_negatives=True)
        """
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.trainer_conf.batch_size,
            shuffle=False,
            collate_fn=identity_collate,
        )

        self.model.to(self.ml_components.compute_device)
        self.model.eval()

        total_pos_loss = 0.0
        n_pos_examples = 0
        pos_cosines = []
        neg_cosines = []

        with torch.no_grad():
            for batch in test_loader:
                # Extract exactly as in train():
                # Process all Flower Records in one pass
                all_hfd_ids = torch.stack([ex['input_ids_hfd'] for ex in batch], dim=0)
                all_hfd_mask = torch.stack([ex['attention_mask_hfd'] for ex in batch], dim=0)
                all_hfd_ids = all_hfd_ids.to(self.ml_components.tensor_device)
                all_hfd_mask = all_hfd_mask.to(self.ml_components.tensor_device)
                emb_hfd = self.model(all_hfd_ids, all_hfd_mask)  # (N, H)

                # flower_name&flower_structure
                # Flatten list[list[Tensor]] into list[Tensor] for all flowers across the batch
                all_nas_ids: list[torch.Tensor] = []
                all_nas_masks: list[torch.Tensor] = []
                counts: list[int] = []

                for ex in batch:
                    nas_ids_list: list[torch.Tensor] = ex['input_ids_nas_list'][:OOM_LIMIT]
                    nas_masks_list: list[torch.Tensor] = ex['attention_mask_nas_list'][:OOM_LIMIT]
                    counts.append(len(nas_ids_list))
                    all_nas_ids.extend(nas_ids_list)
                    all_nas_masks.extend(nas_masks_list)

                # Stack list[Tensor] vertically into two tensors of shape (total_flowers, seq_len)
                stacked_nas_ids: torch.Tensor = torch.stack(all_nas_ids, dim=0).to(self.ml_components.tensor_device)
                stacked_nas_masks: torch.Tensor = torch.stack(all_nas_masks, dim=0).to(self.ml_components.tensor_device)

                # One forward pass for all flowers → (total_flowers, hidden_dim)
                stacked_nas_embs: torch.Tensor = self.model(stacked_nas_ids, stacked_nas_masks)

                # Split back by example and mean‐pool each chunk → List of (hidden_dim,)
                agg_embs: list[torch.Tensor] = []
                offset: int = 0
                for c in counts:
                    slice_emb: torch.Tensor = stacked_nas_embs[offset: offset + c]  # (c, hidden_dim)
                    pooled_emb: torch.Tensor = slice_emb.mean(dim=0)  # (hidden_dim,)
                    agg_embs.append(pooled_emb)
                    offset += c

                # Stack into final (batch_size, hidden_dim) tensor
                emb_nas: torch.Tensor = torch.stack(agg_embs, dim=0)  # (batch, hidden_dim)

                # Evaluations
                # 1) Compute average contrastive loss on positives
                target_pos = torch.ones(emb_nas.shape[0], device=self.ml_components.tensor_device)
                pos_loss = self.criterion(emb_nas, emb_hfd, target_pos)
                total_pos_loss += pos_loss.item() * emb_nas.shape[0]
                n_pos_examples += emb_nas.shape[0]

                # 2) Record cosine similarities on positives
                cos_sim_pos = nn.functional.cosine_similarity(emb_nas, emb_hfd, dim=-1)
                pos_cosines.extend(cos_sim_pos.cpu().tolist())

        # End no_grad()

        # Compute average positive loss & cosine
        avg_pos_loss = total_pos_loss / max(n_pos_examples, 1)
        avg_pos_cosine = float(np.mean(pos_cosines)) if pos_cosines else float('nan')

        metrics: dict[str, float] = {
            'avg_pos_loss': round(avg_pos_loss, 6),
            'avg_pos_cosine': round(avg_pos_cosine, 6),
        }

        if fqfn_metrics:
            with open(fqfn_metrics, 'w+') as fp:
                json.dump(metrics, fp, indent=2)

        return metrics
