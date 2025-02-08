from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class ModelConf:
    input_size = 768  # Total number of features

    hidden_sizes = [512, 128]  # Sizes of hidden layers

    output_size = 1  # Binary classification

    dropout: float = 0.25

    negative_slope: float = 0.01

    compile_model: bool = True

    input_features: tuple[tuple[str, torch.dtype, torch.Size]] = (
        ('text_embeddings', torch.float32, [768]),
        ('label', torch.float32, [1]),
    )

    output_features: tuple[tuple[str, torch.dtype, torch.Size]] = (
        ('logits', torch.float32, [output_size]),
    )


@dataclass(kw_only=True)
class OptimizerConf:
    # use AdamW with β1 = 0.9, β2 = 0.95 from https://arxiv.org/pdf/2302.13971.pdf
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-4


@dataclass(kw_only=True)
class TrainerConf:
    batch_size: int = 32
    epochs: int = 12
    patience: int = 3  # Number of epochs to wait for improvement
    dataset_split_ratio: float = 0.2
    dataset_random_state: int | None = 42


@dataclass(kw_only=True)
class LrComputerConf:
    use_fixed_lr: bool = False
    learning_rate: float = 6e-4
    warmup_epochs: int = 1
    lr_decay_epochs: int = TrainerConf.epochs
    min_lr: float = 6e-5
