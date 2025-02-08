import os

import numpy as np
import onnxruntime as ort

from utils.lm_core import instantiate_ft
from xplainproj.classifier.textcode_features import compute_ft_embeddings, script_attributes_log_scale, \
    ratio_of_comments_to_code, is_64base_content_present

ft_model = instantiate_ft()


class XplainSession:
    def __init__(self):
        model_file = os.path.join(
            os.path.dirname(__file__), 'XplainClassifier.onnx'
        )
        self.session = ort.InferenceSession(
            model_file, providers=['CPUExecutionProvider']
        )

    def run(
        self,
        text_bodies: list[str],
        text_file_names: list[str],
    ) -> list[np.ndarray]:
        outputs = self.session.run(
            output_names=['label'],
            input_feed={
                'np_features': self.compute_ndarray(
                    text_bodies,
                    text_file_names,
                )
            },
        )
        return outputs

    def compute_ndarray(self, text_bodies: list[str], text_file_names: list[str]) -> np.ndarray:
        assert len(text_file_names) == len(text_bodies), \
            f'Lengths of text_file_names and text_bodies are not equal: {len(text_file_names)} vs. {len(text_bodies)}'

        features: list[np.ndarray] = list()
        for text_body, file_name in zip(text_bodies, text_file_names):
            longest_code_line_length, median_code_line_length, lines_of_code, code_size_in_bytes = \
                script_attributes_log_scale(text_body)
            ratio_of_comments_to_code_value = ratio_of_comments_to_code(text_body)
            is_64base_content_present_value = is_64base_content_present(text_body)

            features.append(np.concatenate(
                [
                    np.array([
                        float(longest_code_line_length),
                        float(median_code_line_length),
                        float(lines_of_code),
                        float(code_size_in_bytes),
                        ratio_of_comments_to_code_value,
                        float(is_64base_content_present_value),
                    ], dtype=np.float32),
                    compute_ft_embeddings(file_name, ft_model)
                ]
            ))
        array = np.stack(features, axis=0)
        return array
