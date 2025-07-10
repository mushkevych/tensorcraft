import pandas as pd
import torch
from pandas.core.groupby import DataFrameGroupBy
from torch.utils.data import Dataset

from contrastivebert.classifier.contrastivebert_configuration import ModelConf
from contrastivebert.index.hnsw_index import build_text_body
from utils.bert_embeddings import encode_text_to_inputs
from utils.lm_core import instantiate_ml_components


class ContrastiveBertDataset(Dataset):
    """
    Dataset grouping by HUMAN_DESCRIPTION and FLOWER_NAME.
    Expects a DataFrame with the columns:
      ['HUMAN_DESCRIPTION', 'FLOWER_NAME', 'PETAL_COLOR', 'PETAL_NUMBER', 'STEM_LENGTH', ...]

    Groups rows by (HUMAN_DESCRIPTION, FLOWER_NAME), so that each item
    returns all flower sequences for a given human_description-flower_name pair.
    """

    def __init__(self, df: pd.DataFrame, *args):
        self.ml_components = instantiate_ml_components()

        # Group DataFrame by both human_description and flower_name 
        grouped: DataFrameGroupBy = df.groupby(by=['HUMAN_DESCRIPTION', 'FLOWER_NAME'])

        # Store mapping from (human_description, flower_name) -> sub-DataFrame
        self.by_group: dict[tuple[str, str], pd.DataFrame] = {key: group for key, group in grouped}

        # List of keys for indexing
        self.group_keys: list[tuple[str, str]] = list(self.by_group.keys())

        # Feature names (if needed for interpretability)
        self.feature_names = [name for name, _, _ in ModelConf.input_features]

    def __len__(self) -> int:
        return len(self.group_keys)

    def __getitem__(self, idx: int) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        # *nas* stands for Name and Structure
        # *hfd* stands for Human Flower Description

        # Retrieve grouped rows for the given index
        human_description, flower_name = self.group_keys[idx]
        group_df: pd.DataFrame = self.by_group[(human_description, flower_name)]

        # Build text bodies for each description of the flower
        text_bodies = [build_text_body(row) for _, row in group_df.iterrows()]
        # Encode each flower's text into (input_ids, attention_mask)
        nas_inputs: list[tuple[torch.Tensor, torch.Tensor]] = [encode_text_to_inputs(self.ml_components, t) for t in text_bodies]
        input_ids_nas: list[torch.Tensor] = [ids.squeeze(0) for ids, _ in nas_inputs]
        attention_mask_nas: list[torch.Tensor] = [mask.squeeze(0) for _, mask in nas_inputs]

        # Encode the human flower description text
        input_ids_hfd, attention_mask_hfd = encode_text_to_inputs(self.ml_components, human_description)
        input_ids_hfd = input_ids_hfd.squeeze(0)
        attention_mask_hfd = attention_mask_hfd.squeeze(0)

        # Return all flower-structure-description inputs along with the human description inputs
        return {
            'input_ids_nas_list': input_ids_nas,
            'attention_mask_nas_list': attention_mask_nas,
            'input_ids_hfd': input_ids_hfd,
            'attention_mask_hfd': attention_mask_hfd,
        }
