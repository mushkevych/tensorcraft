from os import path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorWithPadding, EvalPrediction, TrainingArguments

from llmadapterproj.classifier.llmadapter_classifier import (
    instantiate_llmadapter_model,
    instantiate_tokenizer,
    ADAPTER_NAME,
    ADAPTER_PATH
)
from llmadapterproj.classifier.llmadapter_configuration import TrainerConf
from llmadapterproj.trainer.llmadapter_dataset import LlmAdapterDataset
from utils.compute_device import PREFERRED_DEVICE


def compute_metrics(eval_pred: EvalPrediction, *args, **kwargs) -> dict[str, float]:
    """ Compute evaluation metrics using Scikit-Learn """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)  # Convert logits to class predictions

    try:
        roc_auc = roc_auc_score(labels, predictions)
    except ValueError:  # Raised if there's only one class in `labels`
        roc_auc = float('nan')

    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted'),
        'roc_auc': roc_auc,
    }
    return metrics


class LlmAdapterTrainer:
    def __init__(self, df: pd.DataFrame, trainer_conf: TrainerConf):
        self.df = df
        self.trainer_conf = trainer_conf
        self.model = instantiate_llmadapter_model(device_name=PREFERRED_DEVICE)
        self.tokenizer = instantiate_tokenizer()

        self.train_dataset, self.test_dataset = self.prepare_dataset()

        training_args = TrainingArguments(
            output_dir=ADAPTER_PATH,
            per_device_train_batch_size=TrainerConf.batch_size,
            per_device_eval_batch_size=TrainerConf.batch_size,
            learning_rate=2e-4,
            num_train_epochs=TrainerConf.epochs,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            label_names=['labels'],  # must match the LoraDataset.label column name
            push_to_hub=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )

    def prepare_dataset(self) -> tuple[Dataset, Dataset]:
        # Split the data
        train_df, test_df = train_test_split(
            self.df,
            test_size=self.trainer_conf.dataset_split_ratio,
            random_state=self.trainer_conf.dataset_random_state,
        )

        train_dataset: Dataset = LlmAdapterDataset(train_df)
        test_dataset: Dataset = LlmAdapterDataset(test_df)
        return train_dataset, test_dataset

    def train(self) -> None:
        self.trainer.train()

    def save(self) -> None:
        """
        Save the trained model (including its adapter weights) and the tokenizer to ADAPTER_PATH.
        """
        # Option 1: Save the *entire* model (base + adapter) with Hugging Face's standard API:
        # self.trainer.save_model(output_dir=ADAPTER_PATH)
        # This calls self.model.save_pretrained(ADAPTER_PATH) under the hood.
        # The adapter weights and classification head are included.

        # Option 2: Save only the trained adapter checkpoint
        adapter_subfolder = path.join(ADAPTER_PATH, ADAPTER_NAME)
        self.model.save_adapter(adapter_subfolder, adapter_name=ADAPTER_NAME, with_head=True)

        # Save tokenizer
        self.tokenizer.save_pretrained(ADAPTER_PATH)
