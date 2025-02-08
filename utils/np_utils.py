import json

import numpy as np


class NpEncoder(json.JSONEncoder):
    """
        A custom JSON encoder that converts NumPy data types into native Python types
        to ensure compatibility with JSON serialization.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the Sigmoid function for the input array."""
    return 1 / (1 + np.exp(-x))
