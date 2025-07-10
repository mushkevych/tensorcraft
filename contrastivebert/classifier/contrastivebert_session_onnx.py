from os import path

import numpy as np
import onnxruntime as ort
import torch

from contrastivebert.classifier.contrastivebert_configuration import ModelConf
from utils.bert_embeddings import encode_text_to_inputs
from utils.lm_core import instantiate_ml_components
from utils.system_logger import logger


class OnnxSession:
    def __init__(self, model_path: str):
        self.ml_components = instantiate_ml_components(model_path)
        model_file = path.join(path.dirname(__file__), 'ContrastiveSBERT.onnx')
        self.session = ort.InferenceSession(
            model_file,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        logger.info(
            f'ONNX Session {self.session.__class__.__name__} initialized with providers {self.session.get_providers()}.'
        )
        self.model_config = ModelConf()

    def run(self, text_bodies: list[str]) -> np.ndarray:
        """
        Tokenize each string to a 2D PyTorch tensor of shape (1, seq_len),
        then concatenate them into a single (batch_size, seq_len).
        Convert to NumPy and pass into ONNX. Finally, return the batch of pooled outputs.
        """
        # Encode text body -> two tensors of shape (1, seq_len)
        input_tuples = [encode_text_to_inputs(self.ml_components, text) for text in text_bodies]

        # Concatenate along dim=0 to produce (batch_size, seq_len)
        batch_input_ids = torch.cat([it[0] for it in input_tuples], dim=0)        # shape: (N, seq_len)
        batch_attention_mask = torch.cat([it[1] for it in input_tuples], dim=0)   # shape: (N, seq_len)

        # Convert torch.Tensors to NumPy int64 arrays
        np_input_ids = batch_input_ids.cpu().numpy().astype(np.int64)
        np_attention_mask = batch_attention_mask.cpu().numpy().astype(np.int64)

        outputs = self.session.run(
            output_names=['pooler_output'],
            input_feed={
                'input_ids': np_input_ids,
                'attention_mask': np_attention_mask,
            },
        )
        # outputs[0].shape == (batch_size, output_size)
        return outputs[0]
