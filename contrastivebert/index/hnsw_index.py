import json
from typing import Any

# pip install faiss-cpu
import faiss
import numpy as np
import pandas as pd


def build_text_body(row: pd.Series) -> str:
    struct_text = (
        f"[PETAL_COLOR: {row['PETAL_COLOR']}] "
        f"[PETAL_NUMBER: {row['PETAL_NUMBER']}] "
        f"[STEM_LENGTH: {row['STEM_LENGTH']}] "
        f"[LEAF_SHAPE: {row['LEAF_SHAPE']}]"
    )
    text_body = f"Flower: {row['FLOWER_NAME']} | Structure: {struct_text}"
    return text_body


class HNSWIndexWrapper:
    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200, metric=faiss.METRIC_L2, batch_size: int = 32):
        """
        dim             : dimensionality of your embeddings (e.g. 768)
        M               : number of bi‐directional links for HNSW (16, 32, ...)
        ef_construction : higher -> better recall but slower build
        metric          : faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
        """
        self.dim = dim
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.batch_size = batch_size
        self.index = None

        # Store a Python list that maps FAISS‐internal‐IDs -> metadata.
        # By default, FAISS will assign IDs 0..N-1 in the order of index.add() invocations
        self.id_to_metadata: list[dict] = []

    def fit_index(self, reference_df: pd.DataFrame):
        """
        Build a FAISS HNSW index over all rows in reference_df.
        reference_df is expected to contain both source columns, and the embeddings
            e.g. 'PETAL_COLOR', 'PETAL_NUMBER', 'STEM_LENGTH', 'LEAF_SHAPE', ..., 'pooler_output'.

        Steps:
        * Store metadata for each FAISS‐ID.
        * Stack the embeddings -> (N, 768) np.array.
        * Initialize FAISS HNSW and add embeddings.
        """
        # Build metadata list
        self.id_to_metadata = []
        for i, row in reference_df.iterrows():
            self.id_to_metadata.append({
                'row_index': i,
                'flower_name': row['FLOWER_NAME'],
                'petal_color': row['PETAL_COLOR'],
                'petal_number': row['PETAL_NUMBER'],
                'stem_length': row['STEM_LENGTH'],
                'leaf_shape': row['LEAF_SHAPE'],
                'human_description': row['HUMAN_DESCRIPTION'],
            })

        # Build HNSW Index:
        q_embeddings = np.vstack(reference_df['pooler_output'].values).astype(np.float32)  # (N, 768)
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, self.metric)
        self.index.hnsw.efConstruction = self.ef_construction

        N, D = q_embeddings.shape
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch_emb = q_embeddings[start:end]
            # ensure float32 & C-contiguous
            batch_emb = np.ascontiguousarray(batch_emb, dtype=np.float32)
            self.index.add(batch_emb)

    def search(self, query_df: pd.DataFrame, top_k: int = 5, ef_search: int = 50) -> pd.DataFrame:
        """
        Given a DataFrame `query_df` (with the columns: 'FLOWER_NAME', 'PETAL_COLOR', ..., 'pooler_output'),
        for each embedding perform a Top-K search against the previously built index.
        Returns a DataFrame where each row contains one hit per query row
        """
        if not self.index:
            raise RuntimeError('ANN Index not built: call fit_index(...)')

        # Retrieve embeddings
        q_embeddings = np.vstack(query_df['pooler_output'].values).astype(np.float32)  # (N, 768)
        Q, Dq = q_embeddings.shape

        # Configure search‐time parameter
        self.index.hnsw.efSearch = ef_search

        # Search
        distances, indices = self.index.search(q_embeddings, top_k)  # both are shape (Q, top_k)

        # Build an output DataFrame that "flattens" the top_k neighbors for each query.
        results: list[dict[str, Any]] = []
        for qi in range(Q):
            for rank in range(top_k):
                faiss_id = int(indices[qi, rank])
                distance = float(distances[qi, rank])
                meta = self.id_to_metadata[faiss_id]

                retrieved_document = {
                    'query_embeddings': q_embeddings[qi],
                    'rank': rank,
                    'distance': distance,
                    'FLOWER_NAME': meta['flower_name'],
                    'PETAL_COLOR': meta['petal_color'],
                    'PETAL_NUMBER': meta['petal_number'],
                    'STEM_LENGTH': meta['stem_length'],
                    'LEAF_SHAPE': meta['leaf_shape'],
                    'HUMAN_DESCRIPTION': meta['human_description'],
                }
                results.append(retrieved_document)

        return pd.DataFrame(results)

    def save_index(self, index_path: str, metadata_path: str):
        """
        Save the FAISS index to disk (binary) and pickle/JSON‐dump the metadata list.
        """
        if self.index is None:
            raise RuntimeError('ANN Index not built: call fit_index(...)')
        # Save FAISS‐index structure:
        faiss.write_index(self.index, index_path)

        # Serialize id_to_metadata to file
        with open(metadata_path, 'w+') as f:
            json.dump(self.id_to_metadata, f)

    def load_index(self, index_path: str, metadata_path: str):
        """
        Load a previously saved FAISS index and metadata.
        """
        # Load FAISS index:
        self.index = faiss.read_index(index_path)

        # Load metadata:
        with open(metadata_path, 'r') as f:
            self.id_to_metadata = json.load(f)
