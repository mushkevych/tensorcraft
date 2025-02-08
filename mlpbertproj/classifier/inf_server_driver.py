import base64
from os import path

import numpy as np
import pandas as pd

from mlpbertproj.classifier.mlpbert_session_onnx import OnnxSession
from utils.lm_components import FQFP_MODEL_BERT_BASE, FQFP_MODEL_BERT_BASE_XLA
from utils.system_logger import logger

try:
    # Attempt to import torch_xla for XLA (Accelerated Linear Algebra) device architecture
    import torch_xla.core.xla_model as xm
    from mlpbertproj.classifier.mlpbert_session_xla import XlaSession
    from mlpbertproj.classifier.neuronx_pt_exporter import FQFP_MODEL_SEC_XLA, export_bert_to_xla, export_sec_to_xla

    xla_available = True
except Exception as e:
    logger.info(f'XLA Device Not Supported: {e}')
    xla_available = False


def ensure_xla_pt():
    if not path.exists(FQFP_MODEL_SEC_XLA) or not path.exists(FQFP_MODEL_BERT_BASE_XLA):
        export_bert_to_xla()
        export_sec_to_xla()


class InferenceServerDriver:
    def __init__(self):
        if xla_available:
            ensure_xla_pt()
            self.session = XlaSession(model_path=FQFP_MODEL_BERT_BASE)
        else:
            self.session = OnnxSession(model_path=FQFP_MODEL_BERT_BASE, reuse_gpu_tensor=False)

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        inputs['text_body'] = inputs['text_body_b64'].apply(
            lambda x: base64.b64decode(x).decode('utf-8')
        )

        labels: np.ndarray = self.session.run(inputs['text_body'].values)
        for label in labels:
            print(f'Label={label}')
        return pd.DataFrame(labels, columns=('labels',))
