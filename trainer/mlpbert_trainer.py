import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Dataset

from mlpbertproj.classifier.mlpbert_classifier import MlpBertModel
from mlpbertproj.classifier.mlpbert_configuration import ModelConf, OptimizerConf, LrComputerConf, TrainerConf
from trainer.mlpbert_dataset import MlpBertDataset
from trainer.lr_computer_with_decay import LRComputerWithDecay


def initialize_modules() -> tuple[MlpBertModel, optim.AdamW, _Loss, LRComputerWithDecay]:
    model_conf = ModelConf()
    optimizer_conf = OptimizerConf()
    lr_conf = LrComputerConf()

    model = MlpBertModel(model_conf)
    optimizer = optim.AdamW(model.parameters(), betas=optimizer_conf.betas, weight_decay=optimizer_conf.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    lr_wd = LRComputerWithDecay(lr_conf)

    return model, optimizer, criterion, lr_wd


class Trainer:
    def __init__(self, df: pd.DataFrame, trainer_conf: TrainerConf):
        self.df = df
        self.trainer_conf = trainer_conf

        self.train_dataset, self.test_dataset = self.prepare_dataset()
        self.model, self.optimizer, self.criterion, self.lr_wd = initialize_modules()

    def prepare_dataset(self) -> tuple[Dataset, Dataset]:
        # Splitting the data
        train_df, test_df = train_test_split(
            self.df,
            test_size=self.trainer_conf.dataset_split_ratio,
            random_state=self.trainer_conf.dataset_random_state
        )

        # Creating datasets
        train_dataset = MlpBertDataset(train_df)
        test_dataset = MlpBertDataset(test_df)
        return train_dataset, test_dataset

    def train(self) -> None:
        train_loader = DataLoader(self.train_dataset, batch_size=self.trainer_conf.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.trainer_conf.epochs):
            total_loss = 0

            # Compute Learning Rate per epoch and set it in the AdamW optimizer
            lr = self.lr_wd.compute(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            for embeddings, labels in train_loader:
                outputs = self.model(embeddings)  # Logits (batch_size, 1)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f'Epoch: {epoch + 1:>3}/{self.trainer_conf.epochs:>3}, '
                  f'LR:{lr:.5f}, Loss:{total_loss / len(train_loader):>4.5f}')

    def evaluate(self, fqfn_metrics: str = None) -> dict[str, float]:
        test_loader = DataLoader(self.test_dataset, batch_size=self.trainer_conf.batch_size, shuffle=False)

        self.model.eval()
        all_labels: list[np.ndarray] = []
        all_predictions: list[np.ndarray] = []
        all_probabilities: list[np.ndarray] = []  # List to store probabilities

        with torch.no_grad():
            for embeddings, labels in test_loader:
                outputs = self.model(embeddings)  # Logits
                probabilities = torch.sigmoid(outputs).squeeze(dim=1)  # Convert logits to probabilities

                predicted = (probabilities >= 0.5).long()  # Apply threshold for binary classification
                labels = labels.squeeze(dim=1)  # extract actual class

                # Store labels, predictions, and probabilities
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        roc_auc = roc_auc_score(all_labels, all_probabilities)

        metrics = {
            'accuracy': round(accuracy, 4),
            'f1': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'roc-auc': round(roc_auc, 4),
        }
        if fqfn_metrics:
            with open(fqfn_metrics, 'w+') as metric_file:
                json.dump(metrics, metric_file, indent=2)
        return metrics
