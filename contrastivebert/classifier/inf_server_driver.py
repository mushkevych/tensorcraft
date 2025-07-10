import base64
from os import path
from typing import Any

import numpy as np
import pandas as pd

from contrastivebert.classifier.contrastivebert_configuration import ModelConf
from contrastivebert.classifier.contrastivebert_session_onnx import OnnxSession
from contrastivebert.index.hnsw_index import build_text_body, HNSWIndexWrapper
from utils.lm_components import FQFP_MODEL_BERT_BASE


class InferenceServerDriver:
    def __init__(self, fqfn_index: str, batch_size: int = 32) -> None:
        self.session = OnnxSession(model_path=FQFP_MODEL_BERT_BASE)
        self.batch_size = batch_size
        self.fqfn_index = fqfn_index
        self._index: HNSWIndexWrapper = None
        self._index_metadata: dict[str, Any] = None

    @property
    def index(self):
        if self._index is None:
            self._index = HNSWIndexWrapper(ModelConf.output_size, batch_size=self.batch_size)

            self._index.load_index(
                path.join(self.fqfn_index, 'index.hnsw'),
                path.join(self.fqfn_index, 'metadata.json')
            )
        return self._index

    def build_embeddings(self, inputs: pd.DataFrame) -> pd.DataFrame:
        inputs['text_body'] = inputs.apply(build_text_body, axis=1)
        texts: list[str] = inputs['text_body'].tolist()

        # run in batches
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            end = start + self.batch_size
            batch_texts = texts[start:end]

            # session.run accepts a list or array of strings and returns (BATCH, 768) float32 ndarray
            batch_emb: np.ndarray = self.session.run(batch_texts)
            all_embeddings.append(batch_emb)

        # concatenate back to an (N, 768) array
        pooler_output = np.vstack(all_embeddings)  # shape: (N, 768)

        wrapped = list(pooler_output)
        return pd.DataFrame({'pooler_output': wrapped})

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        df = self.build_embeddings(inputs)
        df_flowers = self.index.search(df)
        return df_flowers
