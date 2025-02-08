from dataclasses import dataclass, field
from typing import Any

import numpy as np


FEATURE_LENGTH: int = 2 ** 5  # must reflect dimensionality of the `cc.en.XX.bin`


@dataclass(kw_only=True)
class ModelConf:
    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...] = (
        ('longest_code_line_length', np.float32, [1]),
        ('median_code_line_length', np.float32, [1]),
        ('lines_of_code', np.float32, [1]),
        ('code_size_in_bytes', np.float32, [1]),
        ('ratio_of_comments_to_code', np.float32, [1]),
        ('is_64base_content_present', np.float32, [1]),
        ('file_name_embedding', np.float32, [FEATURE_LENGTH]),
    )

    input_size = sum([feature[2][0] for feature in input_features])  # Total number of features

    output_size = 1  # Binary classification

    output_features: tuple[tuple[str, np.dtype, tuple[int, ...]], ...] = (
        ('logits', np.float32, [output_size]),
    )

    lgbm_classifier: dict[str, Any] = field(default_factory=lambda: {
        # Maximum tree leaves for base learners. (default=31)
        'num_leaves': 31,

        # Maximum tree depth for base learners, <=0 means no limit. (default=-1)
        'max_depth': -1,

        'learning_rate': 0.06,

        # Number of boosted trees to fit. (default=100)
        'n_estimators': 100,

        # objective : str, callable or None, optional (default=None)
        # Specify the learning task and the corresponding learning objective or
        # a custom objective function to be used (see note below).
        # Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        'objective': 'binary',

        # 'gbdt', traditional Gradient Boosting Decision Tree.
        # 'dart', Dropouts meet Multiple Additive Regression Trees.
        # 'rf', Random Forest
        'boosting_type': 'gbdt',

        'random_state': 42,

        # reg_alpha : float, optional (default=0.)
        # L1 regularization term on weights.
        'reg_alpha': 0.1,

        # reg_lambda : float, optional (default=0.)
        # L2 regularization term on weights.
        'reg_lambda': 0.85,

        # importance_type : str, optional (default='split')
        # The type of feature importance to be filled into ``feature_importances_``.
        # If 'split', result contains numbers of times the feature is used in a model.
        # If 'gain', result contains total gains of splits which use the feature.
        'importance_type': 'split'
    })


@dataclass(kw_only=True)
class TrainerConf:
    dataset_split_ratio: float = 0.2
    dataset_random_state: int | None = 42
