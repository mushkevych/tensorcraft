from os import path

import numpy as np
import onnxruntime as ort
import pandas as pd

from imgproj.classifier.img_configuration import ModelConf
from utils.image_toolbox import resize_crop_and_pad, resize_and_pad


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the Sigmoid function for the input array."""
    return 1 / (1 + np.exp(-x))


class ImgProjSession:
    def __init__(self):
        model_file = path.join(
            path.dirname(__file__),
            f'ImgClassifier.{ModelConf.image_size[0]}px.onnx',
        )
        self.session = ort.InferenceSession(
            model_file, providers=['CPUExecutionProvider']
        )


    def run(self, inputs: pd.DataFrame) -> np.ndarray:
        def process_image(img_grey: np.ndarray) -> np.ndarray:
            resized_img = resize_crop_and_pad(image=img_grey, target_size=ModelConf.intermediary_image_size)
            img_grey_scaled = resize_and_pad(image=resized_img, target_size=ModelConf.image_size)

            # Expand dimensions to add color_dimension (1) before the height and width
            # shape becomes (1, image_height, image_width)
            image_expanded = np.expand_dims(img_grey_scaled, axis=0)
            return image_expanded

        # Convert a color image in shape [Height x Width x Channels] into grayscale [1 x Height x Width] using OpenCV
        inputs['img_grey_scaled'] = inputs['img_grey'].apply(process_image)
        imgs_grey_scaled_fp = np.stack(inputs['img_grey_scaled'].values, dtype=np.float32)

        logits: np.ndarray = self.session.run(
            input_feed={'img_grey': imgs_grey_scaled_fp},
            output_names=['logits'],
        )

        # Convert logits to probabilities
        probs: np.ndarray = sigmoid(logits[0])

        # Binary thresholding at 0.5
        labels = (probs > 0.5).astype(np.int32)
        return labels
