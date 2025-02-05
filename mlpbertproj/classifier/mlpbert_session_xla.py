import torch
import numpy as np

from mlpbertproj.classifier.mlpbert_configuration import ModelConf
from mlpbertproj.classifier.neuronx_pt_exporter import FQFP_MODEL_SEC_XLA
from utils.bert_embeddings import compute_bert_embeddings
from lm_components import LmComponents
from utils.system_logger import logger


class XlaSession:
    def __init__(self, model_path):
        logger.info(f'Initializing XLA Session instance...')
        self.model_config = ModelConf()

        # LmComponents.load is XLA-aware
        self.ml_components = LmComponents(model_path)
        self.ml_components.load(output_hidden_states=True)
        logger.info(f'Loaded XLA LmComponents for {self.ml_components.compute_device} device.')

        # Loads an XLA PyTorch model into CPU memory.
        # DO NOT allocate model .to(device=xla)
        self.xla_model = torch.jit.load(FQFP_MODEL_SEC_XLA)
        logger.info(f'Loaded XLA lm-based-model for {self.ml_components.compute_device} device.')

    def run(self, text_bodies: list[str]) -> np.ndarray:
        with torch.no_grad():
            # Compute text embeddings for the entire batch
            list_pt_text_embeddings = [
                compute_bert_embeddings(
                    self.ml_components,
                    text_body,
                    remove_batch_dim=True,
                )
                for text_body in text_bodies
            ]
            batch_text_embeddings = torch.stack(
                list_pt_text_embeddings,
                dim=0,
            )

            pt_logits: torch.Tensor = self.xla_model(batch_text_embeddings)

        # Convert logits to probabilities
        pt_probabilities = torch.sigmoid(pt_logits)  # Apply sigmoid
        np_labels = (pt_probabilities >= 0.5).numpy().astype(np.int32)
        return np_labels
