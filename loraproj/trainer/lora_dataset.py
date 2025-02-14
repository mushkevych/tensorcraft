import numpy
import pandas as pd
import torch
from torch.utils.data import Dataset


class LoraDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df.to_dict('records')  # Convert DataFrame to list of dictionaries

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary formatted for Hugging Face Trainer.
        Labels (0 or 1) are used for classification and included in the dictionary.
        """
        row = self.data[idx]

        # Convert inputs to PyTorch tensors
        pt_input_ids = torch.tensor(row['input_ids'], dtype=torch.long)
        pt_attention_mask = torch.tensor(row['attention_mask'], dtype=torch.long)
        pt_label = torch.tensor(row['label'], dtype=torch.long)  # Classification label (0 or 1)

        # Prepare dictionary for GraphCodeBERT
        inputs = {
            'input_ids': pt_input_ids,
            'attention_mask': pt_attention_mask,
            'labels': pt_label,  # Hugging Face Trainer expects 'labels' for classification
        }

        # Only include token_type_ids if available (some models don’t use it)
        if 'token_type_ids' in row and row['token_type_ids'] is not None and not numpy.isnan(row['token_type_ids']):
            inputs['token_type_ids'] = torch.tensor(row['token_type_ids'], dtype=torch.long)

        return inputs
