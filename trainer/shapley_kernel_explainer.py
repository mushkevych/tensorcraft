import logging
from typing import Literal

import numpy as np
import shap
from joblib import Parallel, delayed
from sklearn.base import OutlierMixin

logger = logging.getLogger('kernel_shapley')

"""
Kernel SHAP is a model-agnostic approximation method for computing Shapley Values using a linear surrogate model to 
estimate contributions of features. This module exposes function `compute_kernel_shapley_values_for_high_dim` that 
utilizes Kernel Shapley approximation to compute feature importance.
It also provides plot functions.

Details:
Kernel SHAP is a specific algorithm in the shap library that uses a weighted linear regression model 
to approximate Shapley values by sampling subsets of features and assigning weights to these subsets. 
Kernel SHAP is particularly well-suited for black-box models, and it explicitly builds 
on Shapley kernel functions for efficient computation.

In contrast: PermutationExplainer, which is implemented by `shapley_permutation_explainer.py` directly perturbs 
the input features to observe changes in model output, without employing the weighted regression approach 
used in Kernel SHAP.
"""

def aggregate_input_values(
    values: np.ndarray,
    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...]
) -> np.ndarray:
    """
    Aggregate values for high-dimensional input features based on the input feature structure.

    :param values: A 2D array of shape (n_samples, n_features) to be aggregated.
           Typically, SHAP values or input feature data.
    :param input_features: A tuple of feature specifications, where each entry is a tuple
           (feature_name, dtype, shape) representing the feature name, data type, and its shape.
    :return: A 2D array of aggregated values of shape (n_samples, n_features_aggregated).
    """
    aggregated_values = []
    current_index = 0

    for feature_name, dtype, shape in input_features:
        size: int = shape[0]
        if size == 1:
            # Single-dimensional features: retain values as is
            aggregated_values.append(values[:, current_index])
            current_index += size
        else:
            # High-dimensional features: aggregate (mean) the values
            agg_value = np.mean(values[:, current_index:current_index + size], axis=1)
            aggregated_values.append(agg_value)
            current_index += size

    # Combine aggregated values into a single array
    return np.column_stack(aggregated_values)


def compute_kernel_shapley_values_for_high_dim(
    model: OutlierMixin,
    X: np.ndarray,
    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...],
    background_size: int = 100,
    sampling_strategy: Literal['random', 'kmeans'] = 'kmeans',
    num_parallel_calls: int = 1,
) -> np.ndarray:
    """
    Compute SHAP values for high-dimensional features and aggregate them for interpretability.

    :param model: Trained outlier model to explain (such as One-Class SVM)
    :param X: Input data of shape (n_samples, n_features).
    :param input_features: A tuple of feature specifications, where each entry is a tuple
           (feature_name, dtype, shape) representing the feature name, data type, and its shape.
    :param background_size: Number of samples to use for the SHAP background dataset.
    :param sampling_strategy: Sampling strategy for summarizing the background dataset ("random" or "kmeans").
    :return: Aggregated SHAP values of shape (n_samples, n_features_aggregated).
    """
    assert hasattr(model, 'decision_function'), f'decision_function must be implemented by {model.__class__.__name__}.'

    if len(X) > background_size:
        # Summarize the background data using random sampling or k-means
        if sampling_strategy == 'random':
            # Random sampling
            background = shap.sample(X, background_size)
        elif sampling_strategy == 'kmeans':
            # K-means clustering
            background = shap.kmeans(X, background_size)
        else:
            raise ValueError(f'Unknown sampling strategy "{sampling_strategy}".')
    else:
        # If X is already small, use it as-is
        background = X

    # Initialize Kernel SHAP explainer using the model's decision function with summarized background
    explainer = shap.KernelExplainer(model.decision_function, data=background)

    # Parallelize SHAP computations
    # Full SHAP values, where X shape is (n_samples, n_features)
    if num_parallel_calls == 1:
        logger.info(f'Computing SHAP values for the input data {X.shape}')
        shap_values = explainer.shap_values(X)
    else:
        logger.info(f'Computing SHAP values in {num_parallel_calls} processes for the input data {X.shape}')
        batch_size = X.shape[0] // num_parallel_calls
        batches = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]

        # Parallelize computation over batches
        shap_values_batches = Parallel(n_jobs=num_parallel_calls)(
            delayed(explainer.shap_values)(batch) for batch in batches
        )

        # Concatenate the results
        shap_values = np.concatenate(shap_values_batches, axis=0)

    # Aggregate SHAP values for high-dimensional features
    aggregated_shap_values = aggregate_input_values(shap_values, input_features)
    return aggregated_shap_values


def shap_beeswarm_plot(
    shap_values: np.ndarray,
    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...],
    X: np.ndarray
) -> None:
    """
    Generate a SHAP beeswarm plot using the modern SHAP API.

    :param shap_values: SHAP values array of shape (n_samples, n_features).
    :param input_features: A tuple of feature specifications, where each entry is a tuple
                   (feature_name, dtype, shape) representing the feature name, data type, and its shape.
    :param X: Dataset array of shape (n_samples, n_features).
    """
    feature_names = [name for name, _, _ in input_features]
    reduced_dim_X = aggregate_input_values(X, input_features)

    # Create a SHAP Explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.zeros(len(shap_values)),  # Base values are optional for beeswarm
        data=reduced_dim_X,
        feature_names=feature_names
    )

    # Generate the beeswarm plot
    shap.plots.beeswarm(explanation)


def shap_bar_plot(
    shap_values: np.ndarray,
    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...],
    X: np.ndarray
) -> None:
    """
    Generate a SHAP beeswarm plot using the modern SHAP API.
    
    :param shap_values: SHAP values array of shape (n_samples, n_features).
    :param input_features: A tuple of feature specifications, where each entry is a tuple
           (feature_name, dtype, shape) representing the feature name, data type, and its shape.
    :param X: Dataset array of shape (n_samples, n_features).
    """
    feature_names = [name for name, _, _ in input_features]
    reduced_dim_X = aggregate_input_values(X, input_features)

    # Create a SHAP Explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.zeros(len(shap_values)),  # Base values are optional for beeswarm
        data=reduced_dim_X,
        feature_names=feature_names
    )

    # Generate the bar plot
    shap.plots.bar(explanation)


def shap_force_plot(
    shap_values: np.ndarray,
    base_values: np.ndarray,
    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...],
    X: np.ndarray,
    index: int
) -> None:
    """
    Generate a SHAP force plot using the modern SHAP API.

    :param shap_values: SHAP values array of shape (n_samples, n_features).
    :param base_values: Base values array of shape (n_samples,) or a scalar base value.
                        Represents the expected value of the model output.
    :param input_features: A tuple of feature specifications, where each entry is a tuple
           (feature_name, dtype, shape) representing the feature name, data type, and its shape.
    :param X: Dataset array of shape (n_samples, n_features).
    :param index: Index of the data point for which to generate the force plot.
    """
    feature_names = [name for name, _, _ in input_features]

    if X.shape[1] > len(feature_names):
        reduced_dim_X = aggregate_input_values(X, input_features)
    else:
        reduced_dim_X = X


    if shap_values.shape[1] > len(feature_names):
        aggregated_shap_values = aggregate_input_values(shap_values, input_features)
    else:
        aggregated_shap_values = shap_values

    # Create a SHAP Explanation object
    explanation = shap.Explanation(
        values=aggregated_shap_values,
        base_values=base_values,  # Base values are used as is
        data=reduced_dim_X,
        feature_names=feature_names
    )

    # Generate the force plot for a specific data point
    shap.initjs()
    shap.plots.force(
        base_value=explanation[index].base_values,
        shap_values=explanation[index].values,
        features=explanation[index].data,
        feature_names=feature_names,
        matplotlib=True,
    )


def shap_heatmap_plot(
    shap_values: np.ndarray,
    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...],
    X: np.ndarray,
) -> None:
    """
    Generate a SHAP heatmap plot using the modern SHAP API.

    :param shap_values: SHAP values array of shape (n_samples, n_features).
    :param input_features: A tuple of feature specifications, where each entry is a tuple
           (feature_name, dtype, shape) representing the feature name, data type, and its shape.
    :param X: Dataset array of shape (n_samples, n_features).
    """
    # Extract feature names
    feature_names = [name for name, _, _ in input_features]

    # Aggregate input values to match the dimensionality of shap_values
    reduced_dim_X = aggregate_input_values(X, input_features)

    # Create a SHAP Explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.zeros(len(shap_values)),  # Base values are optional for heatmap
        data=reduced_dim_X,
        feature_names=feature_names
    )

    # Generate the heatmap plot
    shap.plots.heatmap(explanation)


def shap_decision_plot(
    shap_values: np.ndarray,
    base_value: float,
    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...],
    X: np.ndarray,
    index: int
) -> None:
    """
    Generate a SHAP decision plot for a specific data point.

    :param shap_values: SHAP values array of shape (n_samples, n_features).
    :param base_value: The base value (expected value) of the model's output.
    :param input_features: A tuple of feature specifications, where each entry is a tuple
           (feature_name, dtype, shape) representing the feature name, data type, and its shape.
    :param X: Dataset array of shape (n_samples, n_features).
    :param index: Index of the data point for which to generate the decision plot.
    """
    # Extract feature names
    feature_names = [name for name, _, _ in input_features]

    # Aggregate input values to match the dimensionality of shap_values
    reduced_dim_X = aggregate_input_values(X, input_features)

    # Extract the SHAP values and input data for the specific sample
    sample_shap_values = shap_values[index]
    sample_data = reduced_dim_X[index]

    # Generate the decision plot for the specified sample
    shap.plots.decision(
        base_value=base_value,
        shap_values=sample_shap_values,
        features=sample_data,
        feature_names=feature_names
    )


def shap_waterfall_plot(
    shap_values: np.ndarray,
    base_value: float,
    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...],
    X: np.ndarray,
    index: int
) -> None:
    """
    Generate a SHAP waterfall plot for a specific data point.

    :param shap_values: SHAP values array of shape (n_samples, n_features).
    :param base_value: The base value (expected value) of the model's output.
    :param input_features: A tuple of feature specifications, where each entry is a tuple
           (feature_name, dtype, shape) representing the feature name, data type, and its shape.
    :param X: Dataset array of shape (n_samples, n_features).
    :param index: Index of the data point for which to generate the waterfall plot.
    """
    # Extract feature names
    feature_names = [name for name, _, _ in input_features]

    # Aggregate input values to match the dimensionality of shap_values
    reduced_dim_X = aggregate_input_values(X, input_features)

    # Extract the SHAP values and input data for the specific sample
    sample_shap_values = shap_values[index]
    sample_data = reduced_dim_X[index]

    explanation = shap.Explanation(
        values=sample_shap_values,
        base_values=base_value,
        data=sample_data,
        feature_names=feature_names
    )

    # Generate the waterfall plot for the specified sample
    shap.plots.waterfall(explanation)
