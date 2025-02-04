import base64

import pandas as pd

from txtproj.classifier.txt_session import TxtSession


class InferenceServerDriver:
    def __init__(self):
        self.session = TxtSession()

    def predict(self, inputs: pd.DataFrame, params=None) -> pd.DataFrame:
        inputs['text_body'] = inputs['text_body_b64'].apply(
            lambda x: base64.b64decode(x).decode('utf-8')
        )

        labels = self.session.run(inputs['text_body'].values)
        for label in labels:
            print(f'Label={label}')
        return pd.DataFrame(labels, columns=('labels',))
