import textwrap
from dataclasses import dataclass, asdict, astuple
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# pip install umap-learn
import umap
from hdbscan import HDBSCAN


@dataclass(kw_only=True)
class UmapParams:
    umap_n_neighbors: int = 25
    umap_min_dist: float = 0.15
    cluster_min_size: int = 20
    cluster_min_samples: int = 35
    cluster_selection_epsilon: float = 0.2

    def asdict(self):
        return asdict(self)

    def astuple(self):
        return astuple(self)


def title_for_figure(text: str, figure: plt.Figure) -> str:
    # Get width of figure in pixels
    fig_width_pixels = figure.get_figwidth() * figure.dpi

    char_width_pixels = 30  # Empirical guess

    # Estimate number of characters that fit the width
    num_chars = fig_width_pixels // char_width_pixels

    title = textwrap.fill(text, width=max(num_chars, 48))
    # print(f'{num_chars} {title}')

    return title


def plot_2d(
    X: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray = None,
    parameters: dict[str, Any] = None,
    ax: plt.Axes = None,
    projection='rectilinear',
):
    if ax is None:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(nrows=1, ncols=1, index=1, projection=projection)
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],

                marker='x' if k == -1 else 'o',
                markerfacecolor=tuple(col),
                markeredgecolor='k',
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    title = f'clusters={n_clusters_}'
    if parameters is not None:
        parameters_str = ', '.join(f'{k}={v}' for k, v in parameters.items())
        title += f' | {parameters_str}'
    title = title_for_figure(title, ax.figure)
    ax.set_title(title)
    plt.tight_layout()


def plot_3d(
    X: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray = None,
    parameters: dict = None,
    ax: plt.Axes = None
):
    if ax is None:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(nrows=1, ncols=1, index=1, projection='3d')
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            marker_style = 'x' if k == -1 else 'o'

            ax.scatter(
                X[ci, 0],
                X[ci, 1],
                X[ci, 2],
                marker=marker_style,
                facecolors=col if marker_style != 'x' else 'none',
                # edgecolor='k' if marker_style != 'x' else col,
                s=4 if k == -1 else 1 + 5 * proba_map[ci],
            )

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    title = f'clusters={n_clusters_}'
    if parameters is not None:
        parameters_str = ', '.join(f'{k}={v}' for k, v in parameters.items())
        title += f' | {parameters_str}'
    title = title_for_figure(title, ax.figure)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()


def plot_bucketed_average_distribution(data: np.ndarray, num_buckets: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot a bar graph showing the distribution of data across buckets, with the x-axis
    displaying equally spaced average values of each bucket and the y-axis showing the count of records per bucket.

    Parameters:
    - data: NumPy array of data points to be bucketed and averaged.
    - num_buckets: Number of buckets to divide the data into.
    """
    # Determine range of data
    min_value = data.min()
    max_value = data.max()

    # Calculate the bucket size and assign records to buckets
    bucket_range = (max_value - min_value) / num_buckets
    buckets = np.floor((data - min_value) / bucket_range).astype(int)
    buckets = np.clip(buckets, 0, num_buckets-1)  # Ensure buckets are within range

    # Calculate the average value and count per bucket
    bucket_means = np.array([data[buckets == i].mean() if len(data[buckets == i]) > 0 else 0 for i in range(num_buckets)])
    bucket_counts = np.array([len(data[buckets == i]) for i in range(num_buckets)])

    # Plotting
    plt.figure(figsize=(10, 6))
    bar_positions = range(num_buckets)
    plt.bar(bar_positions, bucket_counts, align='center')

    # Set x-axis labels to string representations of averages, ensuring equal distances
    plt.xticks(bar_positions, labels=[f"{avg:.0f}" for avg in bucket_means], rotation=45)
    plt.xlabel('Average Value per Bucket')
    plt.ylabel('Number of Records')

    plt.title('Distribution of Values Across Buckets')
    plt.tight_layout()
    plt.show()

    return bucket_means, bucket_counts


def draw_cat_histogram(
    column: pd.Series, title:str, limit: int | None = None, figsize:tuple[int, int]=(8, 5), sort_values: bool = True
) -> pd.Series:
    # Count the number of records for each cluster_id
    cluster_counts = column.value_counts(sort=sort_values).sort_index()

    # Plotting the histogram of the top-50 clusters
    plt.figure(figsize=figsize)
    if sort_values:
        cluster_counts.sort_values(ascending=False, inplace=False)[:limit].plot(
            kind='bar', color='skyblue', edgecolor='black'
        )
    else:
        cluster_counts.plot(
            kind='bar', color='skyblue', edgecolor='black'
        )

    plt.title(title)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of records')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    return cluster_counts


def plot_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    aggfunc: str = 'size',
    colorscale: str = 'Viridis',
    title: str = 'Heatmap',
    xaxis_title: str = None,
    yaxis_title: str = None
) -> None:
    """
    Plots a heatmap showing the count or aggregation of intersecting records
    based on two specified columns of a DataFrame.

    :param df: The input DataFrame containing the data.
    :param x_col: The column name to use as the X-axis (categories).
    :param y_col: The column name to use as the Y-axis (levels or groups).
    :param aggfunc: The aggregation function to use in the pivot table
                    (default is 'size' to count records).
    :param colorscale: The colorscale for the heatmap (default is 'Viridis').
    :param title: The title of the heatmap (default is 'Heatmap').
    :param xaxis_title: Title of the X-axis. If not provided, defaults to the name of x_col.
    :param yaxis_title: Title of the Y-axis. If not provided, defaults to the name of y_col.
    """
    y_categories = sorted(df[y_col].unique())


    # Create a pivot table with specified categories
    heatmap_data = (
        df.pivot_table(
            index=y_col,
            columns=x_col,
            aggfunc=aggfunc,
            fill_value=0
        )
        .reindex(index=y_categories, fill_value=0)  # Ensure all Y categories are included
    )

    # Convert to matrix and extract labels
    z_values = heatmap_data.values
    x_labels = heatmap_data.columns.tolist()
    y_labels = heatmap_data.index.tolist()

    # Create the heatmap figure
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            text=z_values,
            texttemplate='%{text}',  # Show values in cells
            textfont={'size': 12}
        )
    )

    # Update layout for better readability and show all Y labels
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or x_col,
        yaxis_title=yaxis_title or y_col,
        xaxis=dict(tickangle=45),
        yaxis=dict(
            tickmode='array',  # Show all labels
            tickvals=y_labels,
            ticktext=[str(label) for label in y_labels]
        ),
        height=600,
        width=800
    )

    # Show the heatmap
    fig.show()


def clusterer(
    embeddings_array: np.ndarray,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = 'minkowski',
    umap_n_component: int = 5,
    umap_use_random_state: bool = True,
    cluster_selection_method='eom',
    cluster_min_size: int = 15,
    cluster_min_samples: int = 30,
    cluster_selection_epsilon: float = 0.2,
) -> tuple[HDBSCAN, float, float]:
    if umap_use_random_state:
        # Use no seed for parallelism.
        random_state = None
    else:
        random_state = np.random.RandomState(42)

    reducer = umap.UMAP(
        random_state=random_state,

        n_neighbors=umap_n_neighbors,   # The size of local neighborhood (in terms of number of neighboring sample points)
                                        # used for manifold approximation


        min_dist=umap_min_dist,     # The effective minimum distance between embedded points.
                                    # Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold
                                    # are drawn closer together, while larger values will result on a more even dispersal of points.

        metric=umap_metric,

        n_components=umap_n_component,  # The dimension of the space to embed into.
                                        # This defaults to 2 to provide easy visualization,
                                        # but can reasonably be set to any integer value in the range 2 to 100

    )
    reduced_embeddings = reducer.fit_transform(embeddings_array.tolist())

    hdb = HDBSCAN(
        min_cluster_size=cluster_min_size,
        min_samples=cluster_min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,  # 'eom' (Excess of Mass algorithm), 'leaf' (select the clusters at the leaves of the tree)
        gen_min_span_tree=True,
        # approx_min_span_tree=False
    )
    hdb.fit(reduced_embeddings)

    clustered = [x for x in hdb.labels_ if x > -1]
    coverage = len(clustered) / len(hdb.labels_)
    dbcv = hdb.relative_validity_

    return hdb, coverage, dbcv
