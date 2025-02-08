import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from xplainproj.classifier.xplain_configuration import ModelConf, TrainerConf
from xplainproj.classifier.xplain_classifier import XplainClassifier
from trainer.xplain_dataset_np import XplainNpDataset


class Trainer:
    def __init__(self, df: pd.DataFrame, trainer_conf: TrainerConf):
        self.df = df
        self.trainer_conf = trainer_conf

        self.train_dataset, self.test_dataset = self.prepare_dataset()

        self.model_conf = ModelConf()
        self.model = XplainClassifier(self.model_conf)

    def prepare_dataset(self) -> Tuple[XplainNpDataset, XplainNpDataset]:
        # Splitting the data
        train_df, test_df = train_test_split(
            self.df,
            test_size=self.trainer_conf.dataset_split_ratio,
            random_state=self.trainer_conf.dataset_random_state
        )

        # Creating datasets
        train_dataset = XplainNpDataset(train_df)
        test_dataset = XplainNpDataset(test_df)

        return train_dataset, test_dataset

    def train(self) -> None:
        # Assuming FastFilterDataset can return the features (X) and labels (y) in the expected format
        X_train, y_train = self.train_dataset.get_data()
        X_test, y_test = self.test_dataset.get_data()

        # Train the model
        self.model._model.fit(X_train, y_train)

        # Optionally, evaluate the model on the test set here and print out metrics
        test_accuracy = self.model._model.score(X_test, y_test)
        print(f'Test Accuracy: {test_accuracy}')

    def evaluate(self, fqfn_metrics: str = None) -> dict[str, float]:
        """
        Evaluate the model on the test set and return a dictionary of performance metrics.

        :param fqfn_metrics: Fully qualified function name for additional metrics (optional).
        :return: A dictionary of evaluation metrics.
        """
        # Assuming FastFilterDataset can return the features (X) and labels (y) in the expected format
        X_test, y_test = self.test_dataset.get_data()

        # Making predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model._model.predict_proba(X_test)[:, 1]  # Assuming binary classification

        # Calculating metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_proba)

        # Compiling metrics into a dictionary
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
