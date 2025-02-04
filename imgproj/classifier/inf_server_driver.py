import base64

import cv2
import numpy as np
import pandas as pd

from imgproj.classifier.img_session import ImgProjSession
from utils.system_logger import logger


def decode_jpeg(b64_str: str) -> np.ndarray:
    """
    Decode a single base64-encoded JPEG string into an OpenCV grayscale image.
    :param b64_str: A base64-encoded JPEG string.
    :return: Decoded grayscale image (NumPy array).
    """
    return cv2.imdecode(
        np.frombuffer(base64.b64decode(b64_str), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )


class InferenceServerDriver:
    def __init__(self):
        self.session = ImgProjSession()

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        # logger.info(f'Converting from: {inputs["jpeg_file_b64"].dtype}')
        # Decode JPEG images from base64 strings
        inputs['img_grey'] = inputs['jpeg_file_b64'].apply(decode_jpeg)

        # Log the shape of the first element
        if not inputs['img_grey'].empty:
            logger.info(f'img_grey[0].shape={inputs["img_grey"].iloc[0].shape}')
        else:
            logger.info('img_grey is empty.')

        labels = self.session.run(inputs)
        for label in labels:
            logger.info(f'Label={label.item()}')
        return pd.DataFrame(labels, columns=('labels',))
