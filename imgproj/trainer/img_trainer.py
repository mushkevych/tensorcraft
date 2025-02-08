import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from imgproj.classifier.img_classifier import ImgClassifier
from imgproj.classifier.img_configuration import ModelConf, OptimizerConf, LrComputerConf, TrainerConf
from imgproj.trainer.img_dataset import ImgDataset
from utils.lr_computer_with_decay import LRComputerWithDecay
from utils.compute_device import resolve_device_mapping
from utils.system_logger import logger


def initialize_modules(device: torch.device) -> tuple[ImgClassifier, optim.AdamW, _Loss, LRComputerWithDecay]:
    model_conf = ModelConf()
    optimizer_conf = OptimizerConf()
    lr_conf = LrComputerConf()

    model = ImgClassifier(model_conf)
    model = model.to(device)
    if TrainerConf.compile_model:
        model.compile()

    optimizer = optim.AdamW(model.parameters(), betas=optimizer_conf.betas, weight_decay=optimizer_conf.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    lr_wd = LRComputerWithDecay(lr_conf)

    return model, optimizer, criterion, lr_wd


class Trainer:
    def __init__(self, df: pd.DataFrame, trainer_conf: TrainerConf, device_name: str = None):
        self.df = df
        self.trainer_conf = trainer_conf
        self.device_name, self.compute_device, self.tensor_device = resolve_device_mapping(device_name)

        self.train_loader, self.test_loader = self.prepare_dataset()
        self.model, self.optimizer, self.criterion, self.lr_wd = initialize_modules(self.compute_device)


    def to(self, device_name: str) -> None:
        self.device_name, self.compute_device, self.tensor_device = resolve_device_mapping(device_name)
        self.model = self.model.to(self.compute_device)

    def prepare_dataset(self):
        # Splitting the data
        train_df, test_df = train_test_split(
            self.df,
            test_size=self.trainer_conf.dataset_split_ratio,
            random_state=self.trainer_conf.dataset_random_state
        )

        # Creating datasets
        train_dataset = ImgDataset(train_df)
        test_dataset = ImgDataset(test_df)

        # Creating data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.trainer_conf.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.trainer_conf.batch_size, shuffle=False)

        return train_loader, test_loader

    def train(self, writer: 'SummaryWriter' = None) -> None:
        self.model.train()

        # tracking best weights and early stopping
        best_model_weights = None
        best_loss = float('inf')
        patience_counter: int = 0

        for epoch in tqdm(range(self.trainer_conf.epochs), total=self.trainer_conf.epochs, desc='Epochs'):
            total_loss = 0

            # compute Learning Rate per epoch and set it at AdamW optimizer
            lr = self.lr_wd.compute(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            for inputs, labels in tqdm(self.train_loader, total=len(self.train_loader), desc='Batches of the epoch'):
                inputs, labels = inputs.to(self.tensor_device), labels.to(self.tensor_device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # early stopping logic to track best_loss
            if total_loss < best_loss:
                best_loss = total_loss
                best_model_weights = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if writer:
                writer.add_scalar('Training loss', total_loss, global_step=epoch)
                writer.add_scalar('Learning rate', lr, global_step=epoch)

                for name, weight in self.model.named_parameters():
                    writer.add_histogram(name, weight, global_step=epoch)
                    writer.add_histogram(f'{name}.grad', weight.grad, global_step=epoch)
            else:
                logger.info(f'Epoch: {epoch + 1:>3}/{self.trainer_conf.epochs:>3}, LR:{lr:>5.5f}, Loss:{total_loss:>.5f}')

            # Early stopping check
            if patience_counter >= self.trainer_conf.patience:
                logger.info(f'Early stopping at epoch {epoch + 1}. Best loss: {best_loss:.5f}')
                break

        # Load the best model weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)

    def evaluate(self, fqfn_metrics: str = None) -> dict[str, float]:
        self.model.eval()
        all_labels: list[np.ndarray] = []
        all_predictions: list[np.ndarray] = []
        all_probabilities: list[np.ndarray] = []  # List to store probabilities

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.tensor_device), labels.to(self.tensor_device)

                # Forward pass
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs)  # Convert logits to probabilities

                prediction = (probs > 0.5).float()  # Binary thresholding at 0.5
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(prediction.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())

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
