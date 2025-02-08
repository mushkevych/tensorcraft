import numpy as np
import pandas as pd

from xplainproj.classifier.xplain_configuration import ModelConf


class XplainNpDataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Define feature names for SHAP interpretation
        self.feature_names = [name for name, _, _ in ModelConf.input_features]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        row = self.df.iloc[idx]

        # Efficiently concatenate all features using list comprehension
        features = np.concatenate([
            np.array(
                row[name], dtype=np.float32
            ).flatten() if len(shape) > 1 else np.array([row[name]], dtype=np.float32)
            for name, _, shape in ModelConf.input_features
        ], axis=0)

        # Get the label as a float32 scalar
        label = np.float32(row['label'])

        return features, label

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get aggregated features and labels for the entire dataset.

        :return: A tuple of (features, labels), where each is a numpy array.
        """
        def row_features(row: pd.Series) -> np.ndarray:
            features: list[np.ndarray] = list()
            for name, _, shape in ModelConf.input_features:
                if len(shape) > 1 or max(shape) > 1:
                    features.append(np.array(row[name], dtype=np.float32).flatten())
                else:
                    features.append(np.array([row[name]], dtype=np.float32))
            return np.concatenate(features, axis=0)

        np_features = np.stack([row_features(row) for _, row in self.df.iterrows()])
        np_labels = self.df['label'].values.astype(np.float32)

        return np_features, np_labels
