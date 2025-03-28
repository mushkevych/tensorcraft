import gc
from typing import Any

import torch
from torch import nn, device
# pip install tensorboard
from torch.utils.tensorboard import SummaryWriter


# pylint: disable = abstract-method
class ModelWrapper(nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple('ModelEndpoints', sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data


def capture_model_architecture(model: nn.Module, t: torch.Tensor, writer: SummaryWriter) -> None:
    if t.shape[0] > 1:
        sample = t.unsqueeze(dim=0)
    else:
        sample = t
    writer.add_graph(ModelWrapper(model), input_to_model=sample)


def clear_cuda_cache(compute_device: device) -> None:
    if compute_device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

        torch.cuda.empty_cache()
        gc.collect()
        print('Post GC Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Post GC Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


def save_model_weights(model: nn.Module, file_path: str) -> None:
    torch.save(model.state_dict(), file_path)


def load_model_weights(model: nn.Module, file_path: str) -> nn.Module:
    model.load_state_dict(torch.load(file_path, weights_only=True, map_location=device('cpu')))
    model.eval()
    return model
