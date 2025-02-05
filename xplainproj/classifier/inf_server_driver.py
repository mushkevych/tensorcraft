import base64

import numpy as np
import pandas as pd

from xplainproj.classifier import xplain_session


class InferenceServerDriver:
    def __init__(self):
        self.session = xplain_session.XplainSession()

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        inputs['text_body'] = inputs['text_body_b64'].apply(
            lambda x: base64.b64decode(x).decode('utf-8')
        )
        inputs['file_name'] = inputs['file_name_b64'].apply(
            lambda x: base64.b64decode(x).decode('utf-8')
        )

        results: np.ndarray = self.session.run(
            inputs['text_body'].values,
            inputs['file_name'].values,
        )[0]  # [0] stands for a batch dimension

        for result in results:
            print(result.item())
        return pd.DataFrame(results, columns=('labels',))
