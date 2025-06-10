import pandas as pd
from torch.utils.data import Dataset

from contrastivebert.classifier.contrastivebert_configuration import ModelConf
from contrastivebert.index.hnsw_index import build_text_body
from utils.bert_embeddings import encode_text_to_inputs
from utils.lm_core import instantiate_ml_components


class ContrastiveBertDataset(Dataset):
    """
    Expects a DataFrame with columns: [ 'ObjectID', 'LEFT_TEXT', 'RIGHT_TEXT']
    """

    def __init__(self, df: pd.DataFrame, *_):
        self.df = df
        self.ml_components = instantiate_ml_components()

        # Define feature names for SHAP interpretation
        self.feature_names = [name for name, _, _ in ModelConf.input_features]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        left_text = row['LEFT_TEXT']
        right_text = row['RIGHT_TEXT']

        # Tokenize both sequences for Siamese/SBERT-style encoding
        input_ids_left, attention_mask_left = encode_text_to_inputs(self.ml_components, left_text)
        input_ids_right, attention_mask_right = encode_text_to_inputs(self.ml_components, right_text)

        # Each item: two sets of input_ids, attention_mask for contrastive learning
        return {
            'input_ids_left': input_ids_left.squeeze(0),
            'attention_mask_left': attention_mask_left.squeeze(0),
            'input_ids_right': input_ids_right.squeeze(0),
            'attention_mask_right': attention_mask_right.squeeze(0)
        }
