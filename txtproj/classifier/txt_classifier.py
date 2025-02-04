from joblib import dump, load
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

from txtproj.classifier.txt_configuration import ModelConf


class LrTxtClassifier:
    def __init__(self, model_conf: ModelConf) -> None:
        self._model = LogisticRegression(**model_conf.lr_classifier)

    def train(self, X: np.ndarray, y: np.ndarray) -> IsolationForest:
        outlier_detector = self._model.fit(X, y)
        return outlier_detector

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the label using the Logistic Regression algorithm.

        :param X: A numpy array of shape (n_samples, n_vocabulary) to make predictions on.
        :return: Array of shape (n_samples,) of predicted labels.
        """
        return self._model.predict(X)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return self._model.decision_function(X)

    def load_model_weights(self, file_path: str) -> None:
        self._model = load(filename=file_path)

    def save_model_weights(self, file_path: str) -> None:
        dump(self._model, filename=file_path)


class SvmTxtClassifier:
    def __init__(self, model_conf: ModelConf) -> None:
        self._model = SVC(**model_conf.svm_classifier)

    def train(self, X: np.ndarray, y: np.ndarray) -> SVC:
        classifier = self._model.fit(X, y)
        return classifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        # Returns the distance to the decision boundary
        return self._model.decision_function(X)

    def load_model_weights(self, file_path: str) -> None:
        self._model = load(filename=file_path)

    def save_model_weights(self, file_path: str) -> None:
        dump(self._model, filename=file_path)


class LgbmTxtClassifier:
    def __init__(self, model_conf: ModelConf) -> None:
        self._model = LGBMClassifier(**model_conf.lgbm_classifier)

    def train(self, X: np.ndarray, y: np.ndarray) -> LGBMClassifier:
        classifier = self._model.fit(X, y)
        return classifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        # Returns the probability of the positive class
        return self._model.predict_proba(X)[:, 1]

    def load_model_weights(self, file_path: str) -> None:
        self._model = load(filename=file_path)

    def save_model_weights(self, file_path: str) -> None:
        dump(self._model, filename=file_path)