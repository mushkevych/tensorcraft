import numpy as np
import torch
from joblib import dump, load
from lightgbm import LGBMClassifier

from xplainproj.classifier.xplain_configuration import ModelConf


class XplainClassifier:
    def __init__(self, model_conf: ModelConf = None) -> None:
        self._model = LGBMClassifier(**model_conf.lgbm_classifier)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Intercept function calls, convert torch.Tensor to numpy.ndarray if needed,
        and dispatch to the predict method.
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return torch.from_numpy(self.predict(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform prediction on the provided input.

        :param X: The input features for prediction, as a numpy array.
        :return: The predicted labels, as a numpy array.
        """
        return self._model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw decision scores, useful for Kernel Shapley and other interpretability methods.

        :param X: A numpy array of shape (n_samples, n_features) to evaluate the model.
        :return: Raw decision scores for each sample.
        """
        if hasattr(self._model, 'predict_proba'):
            # Probability of the positive class
            return self._model.predict_proba(X)[:, 1]
        elif hasattr(self._model, 'decision_function'):
            return self._model.decision_function(X)
        else:
            raise AttributeError('The model does not support decision_function or probability prediction.')

    def load_model_weights(self, file_path: str) -> None:
        self._model = load(filename=file_path)

    def save_model_weights(self, file_path: str) -> None:
        dump(self._model, filename=file_path)
