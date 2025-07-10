from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class ModelConf:
    input_size = 512

    output_size = 768

    compile_model: bool = True

    input_features: tuple[tuple[str, torch.dtype, torch.Size]] = (
        ('input_ids', torch.int64, [input_size]),
        ('attention_mask', torch.int64, [input_size]),
    )

    output_features: tuple[tuple[str, torch.dtype, torch.Size]] = (
        ('pooler_output', torch.float32, [output_size]),
    )


@dataclass(kw_only=True)
class OptimizerConf:
    # use AdamW with β1 = 0.9, β2 = 0.95 from https://arxiv.org/pdf/2302.13971.pdf
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-4


@dataclass(kw_only=True)
class TrainerConf:
    batch_size: int = 16
    epochs: int = 4
    dataset_split_ratio: float = 0.2
    dataset_random_state: int | None = 42


@dataclass(kw_only=True)
class LrComputerConf:
    use_fixed_lr: bool = False
    learning_rate: float = 2e-5
    warmup_epochs: int = 1
    lr_decay_epochs: int = TrainerConf.epochs
    min_lr: float = 6e-5
