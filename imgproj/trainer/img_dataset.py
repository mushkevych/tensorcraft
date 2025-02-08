import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    def __init__(self, df: pd.DataFrame, value_column_name: str = 'img_grey', label_column_name: str = 'label'):
        for column_name in [value_column_name, label_column_name]:
            assert column_name in df.columns, f'Column {column_name} not found in dataframe'
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # Get the grayscale image (stored as ndarray)
        np_img_grey = self.df.iloc[idx]['img_grey']
        pt_img_grey = torch.tensor(np_img_grey, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (1)

        # Get the label (0 or 1)
        pt_label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float32).unsqueeze(0)  # Add extra dimension for Criterion conformity

        return pt_img_grey, pt_label
