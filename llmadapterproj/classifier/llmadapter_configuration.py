from dataclasses import dataclass, field
from typing import Any


@dataclass(kw_only=True)
class ModelConf:
    output_size = 1  # Binary classification

    # https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
    lora_config: dict[str, Any] = field(default_factory=lambda: {
        'r': 16,           # demonstration only: ranks bellow 32 are for code proofing. actual values [32..256]
        'lora_alpha': 8,   # Scale of the LoRA weight parameters merged back with the main weights of model at `x LoRA Alpha`. actual values: [1..2]
        'lora_dropout': 0.1,
        'target_modules': ['query', 'key', 'value'],  # Apply LoRA to self-attention layers
    })


@dataclass(kw_only=True)
class TrainerConf:
    batch_size: int = 8
    epochs: int = 7
    dataset_split_ratio: float = 0.2
    dataset_random_state: int | None = 42


@dataclass(kw_only=True)
class LrComputerConf:
    use_fixed_lr: bool = False
    learning_rate: float = 6e-4
    warmup_epochs: int = 1
    lr_decay_epochs: int = TrainerConf.epochs
    min_lr: float = 6e-5
