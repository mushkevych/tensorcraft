from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(kw_only=True)
class ModelConf:
    # ImgProj_scrape_image -> intermediary_image
    # image dimensions used by preprocessing to cast all incoming images to the same dimensions via resize and crop
    intermediary_image_size = (980, 980)  # (height, width)

    # intermediary_image -> input_image_for_model
    # target image size used as an input by the ML Model
    image_size: tuple[int, int] = (240, 240)  # (height, width)

    input_size: int = image_size[0] * image_size[1]  # Total number of features

    output_size: int = 1

    input_features: tuple[tuple[str, torch.dtype, torch.Size]] = (
        ('img_grey', torch.float32, [768]),
        ('label', torch.float32, [2]),
    )

    output_features: tuple[tuple[str, torch.dtype, torch.Size]] = (
        ('logits', torch.float32, [output_size]),
    )

    eff_net_overrides: dict[str, Any] = field(default_factory=lambda: {
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
        'image_size': ModelConf.image_size,
    })


@dataclass(kw_only=True)
class OptimizerConf:
    # use AdamW with β1 = 0.9, β2 = 0.95 from https://arxiv.org/pdf/2302.13971.pdf
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-4


@dataclass(kw_only=True)
class TrainerConf:
    batch_size: int = 32
    epochs: int = 20
    patience: int = 3  # Number of epochs to wait for improvement
    dataset_split_ratio: float = 0.2
    dataset_random_state: int | None = 42
    compile_model: bool = False


@dataclass(kw_only=True)
class LrComputerConf:
    use_fixed_lr: bool = False
    learning_rate: float = 6e-4
    warmup_epochs: int = 1
    lr_decay_epochs: int = TrainerConf.epochs
    min_lr: float = 6e-5
