import json
from typing import Optional

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
from utils.lm_core import instantiate_ml_components
from utils.lr_computer_with_decay import LRComputerWithDecay


def initialize_modules() -> tuple[ContrastiveSBERT, optim.AdamW, _Loss, LRComputerWithDecay]:
    model_conf = ModelConf()
    optimizer_conf = OptimizerConf()
    lr_conf = LrComputerConf()

    model = ContrastiveSBERT(model_conf)
    optimizer = optim.AdamW(model.parameters(), betas=optimizer_conf.betas, weight_decay=optimizer_conf.weight_decay)
    criterion: _Loss = nn.CosineEmbeddingLoss(margin=0.5)
    lr_wd = LRComputerWithDecay(lr_conf)

    return model, optimizer, criterion, lr_wd


class Trainer:
    def __init__(self, df: pd.DataFrame, trainer_conf: TrainerConf):
        self.df = df
        self.trainer_conf = trainer_conf

        self.train_dataset, self.test_dataset = self.prepare_dataset()
        self.model, self.optimizer, self.criterion, self.lr_wd = initialize_modules()
        self.ml_components = instantiate_ml_components()

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
        train_loader = DataLoader(self.train_dataset, batch_size=self.trainer_conf.batch_size, shuffle=True)

        self.model.train()

        for epoch in tqdm(range(self.trainer_conf.epochs), desc='Training Epochs'):
            total_loss: float = 0.0

            # Compute Learning Rate per epoch and set it in the AdamW optimizer
            lr = self.lr_wd.compute(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1} Batches', leave=False):
                # Move inputs to device
                input_ids_left = batch['input_ids_left'].to(self.ml_components.tensor_device)
                mask_left = batch['attention_mask_left'].to(self.ml_components.tensor_device)
                input_ids_right = batch['input_ids_right'].to(self.ml_components.tensor_device)
                mask_right = batch['attention_mask_right'].to(self.ml_components.tensor_device)

                # Compute embeddings
                emb_left = self.model(input_ids_left, mask_left)
                emb_right = self.model(input_ids_right, mask_right)

                # Labels: +1 for positive pairs (since every row is matched)
                target = torch.ones(emb_left.shape[0]).to(self.ml_components.tensor_device)

                loss = self.criterion(emb_left, emb_right, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f'Epoch: {epoch + 1:>3}/{self.trainer_conf.epochs:>3}, '
                  f'LR:{lr:.5f}, Loss:{total_loss / len(train_loader):>4.5f}')

    def evaluate(
        self,
        fqfn_metrics: Optional[str] = None,
        with_negatives: bool = False,
        negative_ratio: float = 1.0
    ) -> dict[str, float]:
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
        test_loader = DataLoader(self.test_dataset, batch_size=self.trainer_conf.batch_size, shuffle=False)

        self.model.to(self.ml_components.compute_device)
        self.model.eval()

        total_pos_loss = 0.0
        n_pos_examples = 0
        pos_cosines = []
        neg_cosines = []

        with torch.no_grad():
            for batch in test_loader:
                # Extract exactly as in train():
                input_ids_left = batch['input_ids_left'].to(self.ml_components.tensor_device)
                mask_left = batch['attention_mask_left'].to(self.ml_components.tensor_device)
                input_ids_right = batch['input_ids_right'].to(self.ml_components.tensor_device)
                mask_right = batch['attention_mask_right'].to(self.ml_components.tensor_device)

                # Compute embeddings
                emb_left = self.model(input_ids_left, mask_left)  # shape (B, H)
                emb_right = self.model(input_ids_right, mask_right)  # shape (B, H)

                # 1) Compute average contrastive loss on positives
                target_pos = torch.ones(emb_left.shape[0], device=self.ml_components.tensor_device)
                pos_loss = self.criterion(emb_left, emb_right, target_pos)
                total_pos_loss += pos_loss.item() * emb_left.shape[0]
                n_pos_examples += emb_left.shape[0]

                # 2) Record cosine similarities on positives
                cos_sim_pos = nn.functional.cosine_similarity(emb_left, emb_right, dim=-1)
                pos_cosines.extend(cos_sim_pos.cpu().tolist())

                # 3) (Optional) Create negative pairs on the fly
                if with_negatives:
                    batch_size = emb_left.shape[0]

                    # For each sample in this batch, randomly pick a different example from test_dataset
                    neg_input_ids_left = []
                    neg_mask_left = []
                    neg_input_ids_right = []
                    neg_mask_right = []

                    # We need an “index_list” to know the global index of each example:
                    #   make sure ContrastiveBertDataset returns {'index_list': idx_tensor} in __getitem__.
                    # If you haven’t added that, negative sampling is trickier—
                    # you’d need a separate “all_right_embeddings” cache, etc.

                    for i in range(batch_size):
                        pos_idx = batch['index_list'][i].item()
                        neg_idx = pos_idx
                        while neg_idx == pos_idx:
                            neg_idx = torch.randint(0, len(self.test_dataset), (1,)).item()

                        neg_example = self.test_dataset[neg_idx]
                        neg_input_ids_left.append(neg_example['input_ids_left'])
                        neg_mask_left.append(neg_example['attention_mask_left'])
                        neg_input_ids_right.append(neg_example['input_ids_right'])
                        neg_mask_right.append(neg_example['attention_mask_right'])

                    neg_input_ids_left = torch.stack(neg_input_ids_left).to(self.ml_components.tensor_device)
                    neg_mask_left = torch.stack(neg_mask_left).to(self.ml_components.tensor_device)
                    neg_input_ids_right = torch.stack(neg_input_ids_right).to(self.ml_components.tensor_device)
                    neg_mask_right = torch.stack(neg_mask_right).to(self.ml_components.tensor_device)

                    neg_emb_left = self.model(neg_input_ids_left, neg_mask_left)
                    neg_emb_right = self.model(neg_input_ids_right, neg_mask_right)

                    cos_sim_neg = nn.functional.cosine_similarity(neg_emb_left, neg_emb_right, dim=-1)
                    neg_cosines.extend(cos_sim_neg.cpu().tolist())

        # End no_grad()

        # Compute average positive loss & cosine
        avg_pos_loss = total_pos_loss / max(n_pos_examples, 1)
        avg_pos_cosine = float(np.mean(pos_cosines)) if pos_cosines else float('nan')

        metrics: dict[str, float] = {
            'avg_pos_loss': round(avg_pos_loss, 6),
            'avg_pos_cosine': round(avg_pos_cosine, 6),
        }

        if with_negatives:
            avg_neg_cosine = float(np.mean(neg_cosines)) if neg_cosines else float('nan')
            labels = [1] * len(pos_cosines) + [0] * len(neg_cosines)
            scores = pos_cosines + neg_cosines
            try:
                roc_auc = roc_auc_score(labels, scores)
            except ValueError:
                roc_auc = float('nan')

            metrics.update({
                'avg_neg_cosine': round(avg_neg_cosine, 6),
                'roc_auc': round(roc_auc, 6),
            })

        if fqfn_metrics:
            with open(fqfn_metrics, 'w+') as fp:
                json.dump(metrics, fp, indent=2)

        return metrics
