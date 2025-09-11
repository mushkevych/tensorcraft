import json
from os import path

import numpy as np
import onnxruntime as ort

from utils.text_nlp_helper import process_text, vectorize

TEXT_SIZE_THRESHOLD = 256000

class TxtSession:
    def __init__(self):
        model_file = path.join(
            path.dirname(__file__),
            'SvmTxtClassifier.onnx',
        )
        self.session = ort.InferenceSession(
            model_file, providers=['CPUExecutionProvider']
        )
        vocabulary_file = path.join(
            path.dirname(__file__),
            'vocabulary.json',
        )
        with open(vocabulary_file, 'r') as vf:
            self.vocabulary = json.load(vf)

    def run(self, text_body: list[str]) -> np.ndarray:
        X: list[str] = list()
        for text_line in text_body:
            if len(text_line) > TEXT_SIZE_THRESHOLD:
                text_line = 'EMPTY TEXT BODY'
            processed_text = process_text(text_line)
            X.append(processed_text)

        tfidf = vectorize(self.vocabulary, X)

        outputs = self.session.run(
            input_feed={'text_tfidf': tfidf},
            output_names=['label'],
        )

        # outputs[0] refers to the "label" tensor from output_names parameter
        # "label" of 0 indicated a negative classification, while 1 indicates a positive classification
        labels = outputs[0]
        return np.array(labels, dtype=np.int32, copy=None)
