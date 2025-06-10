import base64
from os import path

import numpy as np
import pandas as pd

from contrastivebert.classifier.contrastivebert_session_onnx import OnnxSession
from contrastivebert.index.hnsw_index import build_text_body
from utils.lm_components import FQFP_MODEL_BERT_BASE


class InferenceServerDriver:
    def __init__(self, batch_size: int = 32):
        self.session = OnnxSession(model_path=FQFP_MODEL_BERT_BASE)
        self.batch_size = batch_size

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
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
