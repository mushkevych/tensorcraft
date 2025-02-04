from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(kw_only=True)
class ModelConf:
    vocabulary_size: int = 256

    input_features: tuple[tuple[str, np.dtype, tuple[int, ...]]] = (
        ('text_tfidf', np.float32, [vocabulary_size]),
    )

    input_size = sum([feature[2][0] for feature in input_features])  # Total number of features

    output_size = 1

    output_features: tuple[tuple[str, np.dtype, tuple[int, ...]]] = (
        ('label', np.float32, [output_size]),
    )

    lr_classifier: dict[str, Any] = field(default_factory=lambda: {
        'penalty': 'l2',           # L2 regularization (Ridge)
        'C': 1.0,                  # Inverse of regularization strength; smaller values specify stronger regularization.
        'solver': 'lbfgs',         # An efficient solver for large datasets with good convergence properties.
        'max_iter': 100,           # Maximum number of iterations for the solver to converge.
        'class_weight': None,      # Can be set to 'balanced' if classes are imbalanced.
        'random_state': 42,        # Ensures reproducibility of the results.
        'n_jobs': -1               # Uses all available processors to speed up training.
    })

    svm_classifier: dict[str, Any] = field(default_factory=lambda: {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    })

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
