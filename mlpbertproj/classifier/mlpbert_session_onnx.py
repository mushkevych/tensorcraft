from os import path

import numpy as np
import onnxruntime as ort
import torch

from mlpbertproj.classifier.mlpbert_configuration import ModelConf
from utils.bert_embeddings import compute_bert_embeddings
from utils.system_logger import logger
from np_utils import sigmoid
from trainer.lm_core import instantiate_ml_components, MODEL_BERT_BASE


class OnnxSession:
    def __init__(self, model_path, reuse_gpu_tensor=False):
        self.reuse_gpu_tensor = reuse_gpu_tensor
        logger.info(f'Reuse GPU Tensor: {self.reuse_gpu_tensor}')
        self.ml_components = instantiate_ml_components(MODEL_BERT_BASE)
        self.ml_components.load(output_hidden_states=True)
        model_file = path.join(path.dirname(__file__), 'MlpBertModel.onnx')
        self.session = ort.InferenceSession(
            model_file,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        logger.info(
            f'ONNX Session {self.session.__class__.__name__} initialized with providers {self.session.get_providers()}.'
        )
        self.model_config = ModelConf()

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

            if batch_text_embeddings.is_cuda and self.reuse_gpu_tensor:
                io_binding = self.session.io_binding()
                pt_text_embeddings = batch_text_embeddings.contiguous()

                io_binding.bind_input(
                    name='np_text_embeddings',
                    device_type='cuda',
                    device_id=0,
                    element_type=np.float32,
                    shape=tuple(pt_text_embeddings.shape),
                    buffer_ptr=pt_text_embeddings.data_ptr(),
                )

                logit_output_shape = (batch_text_embeddings.shape[0], 2)
                logit_output = torch.empty(
                    logit_output_shape,
                    dtype=torch.float32,
                    device='cuda',
                ).contiguous()

                io_binding.bind_output(
                    name='np_logits',
                    device_type='cuda',
                    device_id=0,
                    element_type=np.float32,
                    shape=tuple(logit_output.shape),
                    buffer_ptr=logit_output.data_ptr(),
                )

                self.session.run_with_iobinding(io_binding)
                np_logits: np.ndarray = logit_output.cpu().numpy()
            else:
                np_text_embeddings = batch_text_embeddings.cpu().numpy()
                outputs: list[np.ndarray] = self.session.run(
                    output_names=['np_logits'],
                    input_feed={'np_text_embeddings': np_text_embeddings},
                )

                # outputs[0] refers to the 'np_logits' tensor from output_names parameter
                np_logits: np.ndarray = outputs[0]

        # Convert logits to probabilities
        np_probabilities: np.ndarray = sigmoid(np_logits)
        np_labels = (np_probabilities > 0.5).astype(np.int32)
        return np_labels

    def io_binding(self):
        return self.session.io_binding()

    def run_with_iobinding(self, io_binding):
        self.session.run_with_iobinding(io_binding)
