from dataclasses import dataclass

import numpy as np
from hdbscan import HDBSCAN

from utils.hdbscan_clustering import clusterer


@dataclass(frozen=True)
class ClustererRun:
    """Serializable worker input for one clusterer invocation."""

    embeddings_array: np.ndarray
    metric: str
    umap_n_neighbors: int
    umap_min_dist: float
    cluster_min_size: int
    cluster_min_samples: int
    cluster_selection_epsilon: float
    umap_n_component: int = 5
    umap_use_random_state: bool = True
    cluster_selection_method: str = "eom"
    n_jobs: int = 1
    return_hdb: bool = False

    def params_tuple(self) -> tuple[int, float, int, int, float]:
        return (
            int(self.umap_n_neighbors),
            float(self.umap_min_dist),
            int(self.cluster_min_size),
            int(self.cluster_min_samples),
            float(self.cluster_selection_epsilon),
        )


@dataclass(frozen=True)
class ClustererResult:
    """Serializable worker output for one clusterer invocation."""

    metric: str
    umap_n_neighbors: int
    umap_min_dist: float
    cluster_min_size: int
    cluster_min_samples: int
    cluster_selection_epsilon: float
    dbcv: float
    coverage: float
    unclustered: int
    n_clusters: int
    n_rows: int
    labels: np.ndarray | None
    hdb: HDBSCAN | None
    error: str | None = None

    def params_tuple(self) -> tuple[int, float, int, int, float]:
        return (
            int(self.umap_n_neighbors),
            float(self.umap_min_dist),
            int(self.cluster_min_size),
            int(self.cluster_min_samples),
            float(self.cluster_selection_epsilon),
        )


def execute_cluster_run(run: ClustererRun) -> ClustererResult:
    """Top-level worker for ProcessPoolExecutor; calls `clusterer` once."""
    n_rows = int(run.embeddings_array.shape[0])
    try:
        hdb, coverage, dbcv = clusterer(
            run.embeddings_array,
            umap_n_neighbors=run.umap_n_neighbors,
            umap_min_dist=run.umap_min_dist,
            umap_metric=run.metric,
            umap_metric_kwds={"p": 2} if run.metric == "minkowski" else None,
            umap_n_component=run.umap_n_component,
            umap_use_random_state=run.umap_use_random_state,
            cluster_selection_method=run.cluster_selection_method,
            cluster_min_size=run.cluster_min_size,
            cluster_min_samples=run.cluster_min_samples,
            cluster_selection_epsilon=run.cluster_selection_epsilon,
            n_jobs=run.n_jobs,
        )
        labels = np.asarray(hdb.labels_, dtype=np.int32)
        unclustered = int((labels == -1).sum())
        n_clusters = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
        return ClustererResult(
            metric=run.metric,
            umap_n_neighbors=run.umap_n_neighbors,
            umap_min_dist=run.umap_min_dist,
            cluster_min_size=run.cluster_min_size,
            cluster_min_samples=run.cluster_min_samples,
            cluster_selection_epsilon=run.cluster_selection_epsilon,
            dbcv=float(dbcv),
            coverage=float(coverage),
            unclustered=unclustered,
            n_clusters=n_clusters,
            n_rows=n_rows,
            labels=labels,
            hdb=hdb if run.return_hdb else None,
            error=None,
        )
    except Exception as exc:
        return ClustererResult(
            metric=run.metric,
            umap_n_neighbors=run.umap_n_neighbors,
            umap_min_dist=run.umap_min_dist,
            cluster_min_size=run.cluster_min_size,
            cluster_min_samples=run.cluster_min_samples,
            cluster_selection_epsilon=run.cluster_selection_epsilon,
            dbcv=float("-inf"),
            coverage=float("-inf"),
            unclustered=-1,
            n_clusters=-1,
            n_rows=n_rows,
            labels=np.empty((0,), dtype=np.int32),
            hdb=None,
            error=repr(exc),
        )
