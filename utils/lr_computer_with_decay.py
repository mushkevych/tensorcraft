import math
from typing import Any


class LRComputerWithDecay:
    def __init__(self, config: 'LrComputerConf'):
        """
        :param config: has the following fields
        * use_fixed_lr: whether to decay the learning rate or use fixed `learning_rate`
        * learning_rate: fixed LR value if `use_fixed_lr is True` or max LR otherwise
        * warmup_epochs: how many steps to warm up for
        * lr_decay_epochs: should be ~= max_epochs per Chinchilla
        * min_lr: minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        """
        self.use_fixed_lr = config.use_fixed_lr
        self.learning_rate = config.learning_rate
        self.warmup_epochs = config.warmup_epochs
        self.lr_decay_epochs = config.lr_decay_epochs
        self.min_lr = config.min_lr

    def state_dict(self) -> dict[str, Any]:
        return {
            'use_fixed_lr': self.use_fixed_lr,
            'learning_rate': self.learning_rate,
            'warmup_epochs': self.warmup_epochs,
            'lr_decay_epochs': self.lr_decay_epochs,
            'min_lr': self.min_lr,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.use_fixed_lr = state_dict['use_fixed_lr']
        self.learning_rate = state_dict['learning_rate']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.lr_decay_epochs = state_dict['lr_decay_epochs']
        self.min_lr = state_dict['min_lr']

    def compute(self, epoch: int = 0) -> float:
        """ learning rate decay scheduler (cosine with warmup)
        :parameter epoch: iteration number
        :return: float representing learning rate for this iteration
        """
        if self.use_fixed_lr:
            return self.learning_rate

        # 1) linear warmup for warmup_iters steps
        if epoch < self.warmup_epochs:
            epoch = epoch if epoch > 0 else 1
            return self.learning_rate * epoch / self.warmup_epochs

        # 2) if it > lr_decay_iters, return min learning rate
        if epoch > self.lr_decay_epochs:
            return self.min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (epoch - self.warmup_epochs) / (self.lr_decay_epochs - self.warmup_epochs)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


if __name__ == '__main__':
    from dataclasses import dataclass

    @dataclass(kw_only=True)
    class LrComputerConf:
        use_fixed_lr: bool = False
        learning_rate: float = 6e-4
        warmup_epochs: int = 1
        lr_decay_epochs: int = 10
        min_lr: float = 6e-5


    config = LrComputerConf(warmup_epochs=2)
    lr_wd = LRComputerWithDecay(config)
    for i in range(LrComputerConf.lr_decay_epochs + 2):
        print(f'i={i:>3}\t lr={lr_wd.compute(i):.5f}')
