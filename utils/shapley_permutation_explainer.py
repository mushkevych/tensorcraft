import numpy as np
import shap
import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from typing import Any

"""
Function in this module employs the PermutationExplainer, which:

  *  Treats the model as a black box.
  *  Perturbs the input features (e.g., sets them to baseline values like mean or marginal samples).
  *  Evaluates the model's predictions for different subsets of features.
  *  Estimates the Shapley values based on these perturbations.
"""

def compute_shap_values_generic(
    model: Any, features: np.ndarray, labels: np.ndarray, num_samples: int = 100
) -> np.ndarray:
    """
    Compute SHAP values for a given model using the Permutation Explainer. The model is expected
    to follow the scikit-learn API, particularly having a `predict_proba` method for classification tasks.

    :param model: The trained model, following the scikit-learn model API.
    :param features: The features of the dataset, as a numpy array.
    :param labels: The labels of the dataset, as a numpy array. This parameter is currently not used in the function,
                   but included for API consistency or future use.
    :param num_samples: Number of samples to use for SHAP approximation.
    :return: SHAP values.
    """

    # If the dataset is large, sample a subset
    if len(features) > num_samples:
        indices = np.random.choice(len(features), num_samples, replace=False)
        sampled_features = features[indices]
    else:
        sampled_features = features

    # Define a function that takes a batch of data and outputs the model predictions
    def model_predict(data):
        # Check if the model has a `predict_proba` method, use it; otherwise, fall back to `predict`
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(data)
        else:
            return model.predict(data)

    # Using PermutationExplainer
    explainer = shap.explainers.Permutation(model_predict, sampled_features)
    shap_values = explainer(sampled_features)

    return shap_values


def compute_shap_values(model: nn.Module, dataset: Dataset, num_samples: int = 100) -> np.ndarray:
    """
    Compute SHAP values for a given PyTorch model and dataset using the Permutation Explainer.

    :param model: The trained PyTorch model.
    :param dataset: The dataset to use for computing SHAP values.
                    Assumption is that this dataset contains `labels` as the last dimension.
    :param num_samples: Number of samples to use for SHAP approximation.
    :return: SHAP values.
    """
    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    # Select a batch of data
    for batch in dataloader:
        features = batch[:-1]
        label = batch[-1]
        break

    # Define a function that takes a batch of data and outputs the model predictions
    def model_predict(*args):
        with torch.no_grad():
            predictions = model(*args).cpu().numpy()
        return predictions

    # Prepare the data for SHAP
    # Convert each tensor to a numpy array and stack them along a new axis
    data_for_shap = np.stack([f.cpu().numpy() for f in features], axis=1)

    # Using PermutationExplainer
    explainer = shap.explainers.Permutation(model_predict, data_for_shap)
    shap_values = explainer(data_for_shap)

    return shap_values


def compute_shap_values_for_high_dim(
    model: nn.Module,
    dataset: Dataset,
    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...],
    num_samples: int = 100
) -> np.ndarray:
    """
    Compute SHAP values for a given PyTorch model and dataset using the Permutation Explainer.

    :param model: The trained PyTorch model. Assumption is that it overloads:
        `__call__(X: torch.Tensor) -> torch.Tensor`
    :param dataset: The dataset to use for computing SHAP values.
                    Assumption is that this dataset provides list of torch.Tensors,
                    where features are presented by individual tensors,
                    and `labels` are the last element of the list.
    :param input_features: A tuple of feature specifications, where each entry is a tuple
           (feature_name, dtype, shape) representing the feature name, data type, and its shape.
    :param num_samples: Number of samples to use for SHAP approximation.
    :return: SHAP values.
    """
    # Implementation notes:
    # first, we reduce the input data by computing *mean* for the high-dimensional features
    # next, in the custom *model_predict* we expand previously computed *means* to a full-width of *hidden_shape[0]*
    # WARNING: consider that *shap.explainers.Permutation* and its *explainer* method must be given
    #          ndarray of the same shape, hence the expansion of the reduced features in the *model_predict*

    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    # Select a batch of data
    for batch in dataloader:
        features = batch[:-1]
        label = batch[-1]
        break

    reduced_dimensionality_features: list[torch.Tensor] = list()
    for i, (feature_name, dtype, shape) in enumerate(input_features):
        size: int = shape[0]
        if size == 1:
            # Scalar feature
            reshaped_feature = features[i]
            reduced_dimensionality_features.append(reshaped_feature)
        else:
            # High-dimensional features: aggregate (mean) the values
            mean_vals: torch.Tensor = torch.mean(features[i], dim=1)
            reduced_dimensionality_features.append(mean_vals.unsqueeze(dim=1))

    # Concatenate features
    pt_reduced_dimensionality_features = torch.cat(reduced_dimensionality_features, dim=1)
    np_reduced_dimensionality_features = pt_reduced_dimensionality_features.cpu().numpy().astype(dtype=np.float32)

    # Define a function that takes a batch of data and outputs the model predictions
    def model_predict(data: np.ndarray) -> np.ndarray:
        expanded_features: list[torch.Tensor] = list()

        with torch.no_grad():
            for i, (feature_name, dtype, shape) in enumerate(input_features):
                size: int = shape[0]
                if size == 1:
                    # Scalar feature: simply unsqueeze
                    np_features = data[:, i]
                    pt_features = torch.tensor(np_features, dtype=torch.float32)
                    pt_reshaped_feature = pt_features.unsqueeze(dim=1)
                    expanded_features.append(pt_reshaped_feature)
                else:
                    # High-dimensional feature: replicate *mean_value* from shape (1,) to (1, size)
                    np_mean_values = data[:, i]
                    pt_mean_values = torch.tensor(np_mean_values, dtype=torch.float32)
                    pt_expanded_feature = pt_mean_values.unsqueeze(dim=1).repeat(1, size)
                    expanded_features.append(pt_expanded_feature)

            pt_expanded_features = torch.cat(expanded_features, dim=1)
            np_predictions = model(pt_expanded_features).cpu().numpy().astype(dtype=np.float32)

        return np_predictions

    # Using PermutationExplainer
    explainer = shap.explainers.Permutation(model_predict, np_reduced_dimensionality_features)
    shap_values = explainer(np_reduced_dimensionality_features)

    return shap_values
