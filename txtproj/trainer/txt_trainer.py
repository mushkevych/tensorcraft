import json
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, \
    precision_recall_curve, auc
from sklearn.model_selection import train_test_split

from txtproj.trainer.txt_dataset import TxtDataset
from txtproj.classifier.txt_classifier import LrTxtClassifier, SvmTxtClassifier, LgbmTxtClassifier
from txtproj.classifier.txt_configuration import TrainerConf, ModelConf


class Trainer:
    def __init__(
        self,
        model_class: type[LrTxtClassifier | SvmTxtClassifier | LgbmTxtClassifier],
        df: pd.DataFrame,
        trainer_conf: TrainerConf
    ):
        self.df = df
        self.trainer_conf = trainer_conf
        self.train_dataset, self.test_dataset = self.prepare_dataset()

        self.model_conf = ModelConf()
        self.model = model_class(self.model_conf)

    def prepare_dataset(self) -> Tuple[TxtDataset, TxtDataset]:
        train_df, test_df = train_test_split(
            self.df,
            test_size=self.trainer_conf.dataset_split_ratio,
            random_state=self.trainer_conf.dataset_random_state
        )

        train_dataset = TxtDataset(df=train_df)
        test_dataset = TxtDataset(df=test_df)
        return train_dataset, test_dataset

    def train(self) -> None:
        X_train, y_train = self.train_dataset.get_data()
        self.model.train(X_train, y_train)

    def evaluate(self, fqfn_metrics: Optional[str] = None, average: str = 'binary') -> dict[str, float]:
        """
        Evaluate the model on the test set and return a dictionary of performance metrics.

        :param fqfn_metrics: Fully qualified function name for additional metrics (optional).
        :param average: {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='binary'
                        This parameter is required for multiclass/multilabel targets.
        :return: A dictionary of evaluation metrics.
        """
        dataset = TxtDataset(df=self.df)
        X, y = dataset.get_data()

        # 0 indicates "negative" case, while "1" - positive
        predicted_labels = self.model.predict(X)
        true_labels = (y == 1).astype(int)

        print(f'predicted_labels.values={np.unique(predicted_labels, return_counts=True)}')
        print(f'true_labels.values={np.unique(true_labels, return_counts=True)}')

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average=average)
        precision = precision_score(true_labels, predicted_labels, average=average)
        recall = recall_score(true_labels, predicted_labels, average=average)

        # Compiling metrics into a dictionary
        metrics = {
            'accuracy': round(accuracy, 4),
            'f1': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
        }

        # Compute AUC
        scores = self.model.evaluate(X)
        if len(np.unique(true_labels)) > 1:
            roc_auc = roc_auc_score(true_labels, scores)
            metrics['roc-auc'] = round(roc_auc, 4)
        else:
            precision, recall, _ = precision_recall_curve(true_labels, scores)
            pr_auc = auc(recall, precision)
            metrics['precision-recall-auc'] = round(pr_auc, 4)

        if fqfn_metrics:
            with open(fqfn_metrics, 'w+') as metric_file:
                json.dump(metrics, metric_file, indent=2)
        return metrics
