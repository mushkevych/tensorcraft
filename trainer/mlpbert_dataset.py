import pandas as pd
import torch
from torch.utils.data import Dataset

from mlpbertproj.classifier.mlpbert_configuration import ModelConf


class MlpBertDataset(Dataset):
    def __init__(self, df: pd.DataFrame, *_):
        self.df = df

        # Define feature names for SHAP interpretation
        self.feature_names = [name for name, _, _ in ModelConf.input_features]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the text embeddings(stored as ndarray)
        np_text_embeddings = self.df.iloc[idx]['text_embeddings']
        pt_text_embeddings = torch.tensor(np_text_embeddings, dtype=torch.float32)

        # Get the label (0 or 1)
        # Add extra dimension for Criterion conformity
        pt_label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float32).unsqueeze(0)

        return pt_text_embeddings, pt_label
