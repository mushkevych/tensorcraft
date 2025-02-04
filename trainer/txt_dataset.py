import numpy as np
import pandas as pd

from txtproj.classifier.txt_configuration import ModelConf


def concat_row_elements(row: pd.Series) -> np.ndarray:
    concatenated_data: list[float] = list()

    for column_name, value in row.to_dict().items():
        if isinstance(value, np.ndarray | list):
            concatenated_data.extend(value)
        else:
            concatenated_data.append(value)

    # Return a single ndarray
    return np.array(concatenated_data, dtype=np.float32)


class TxtDataset:
    def __init__(self, df: pd.DataFrame = None):
        # 0 indicates a negative case, while +1 indicates a positive case.
        self.df = df

        # Define feature names for SHAP interpretation
        self.feature_names = list(name for name, _, _ in ModelConf.input_features)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        # Ensure that features are in the correct order and dtype is np.float32
        features = self.df.iloc[idx][self.feature_names].apply(concat_row_elements, axis=1).astype(dtype=np.float32)

        # Get the label as a scalar value, assuming binary classification
        label = self.df.iloc[idx]['label'].astype(dtype=np.float32)

        return features, label

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get aggregated features and labels for the entire dataset.

        :return: A tuple of (features, labels), where each is a numpy array.
        """
        features = self.df[self.feature_names].apply(concat_row_elements, axis=1).to_list()
        labels = self.df['label'].values.astype(dtype=np.float32)
        return features, labels
