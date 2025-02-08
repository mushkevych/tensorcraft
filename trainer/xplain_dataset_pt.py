import torch
import pandas as pd

from xplainproj.classifier.xplain_configuration import ModelConf


class XplainPtDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Define feature names for SHAP interpretation
        self.feature_names = [name for name, _, _ in ModelConf.input_features]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        row = self.df.iloc[idx]
        features: list[torch.Tensor] = list()
        for name, _, shape in ModelConf.input_features:
            if len(shape) > 1 or max(shape) > 1:
                features.append(torch.tensor(row[name], dtype=torch.float32))
            else:
                features.append(torch.tensor(row[name], dtype=torch.float32).unsqueeze(0))

        # Get the label as a float32 scalar
        label = torch.tensor(row['label'], dtype=torch.float32).unsqueeze(0)
        return *features, label

