import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorWithPadding, EvalPrediction
from transformers import TrainingArguments

from loraproj.classifier.lora_classifier import instantiate_lora_model, instantiate_tokenizer
from loraproj.classifier.lora_configuration import TrainerConf
from loraproj.trainer.lora_dataset import LoraDataset
from utils.compute_device import PREFERRED_DEVICE


def compute_metrics(eval_pred: EvalPrediction, *args, **kwargs) -> dict[str, float]:
    """
    Compute evaluation metrics using Scikit-Learn.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)  # Convert logits to class predictions

    try:
        roc_auc = roc_auc_score(labels, predictions)
    except ValueError:  # Raised if there's only one class in `labels`
        roc_auc = float('nan')

    metrics = {
        # 'eval_accuracy': accuracy_score(labels, predictions),
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted'),
        'roc_auc': roc_auc,
    }
    return metrics


class LoraTrainer:
    def __init__(self, df: pd.DataFrame, trainer_conf: TrainerConf):
        self.df = df
        self.trainer_conf = trainer_conf
        self.model = instantiate_lora_model(device_name=PREFERRED_DEVICE)
        self.tokenizer = instantiate_tokenizer()

        self.train_dataset, self.test_dataset = self.prepare_dataset()

        training_args = TrainingArguments(
            output_dir='./lora_graphcodebert_powershell',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-4,
            num_train_epochs=TrainerConf.epochs,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_accuracy',
            label_names=['labels'],  # must match the name of the Dataset property for Label
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
        # Splitting the data
        train_df, test_df = train_test_split(
            self.df,
            test_size=self.trainer_conf.dataset_split_ratio,
            random_state=self.trainer_conf.dataset_random_state
        )

        train_dataset: Dataset = LoraDataset(train_df)
        test_dataset: Dataset = LoraDataset(test_df)
        return train_dataset, test_dataset

    def train(self) -> None:
        self.trainer.train()

    def save_and_merge(self):
        # Merge LoRA with base model for inference
        merged_model = self.model.merge_and_unload()

        # Save model
        merged_model.save_pretrained('./lora_graphcodebert_powershell')
        self.tokenizer.save_pretrained('./lora_graphcodebert_powershell')
