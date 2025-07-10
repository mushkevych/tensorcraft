import gc
from typing import Any

import torch
from torch import nn, device
# pip install tensorboard
from torch.utils.tensorboard import SummaryWriter

from utils.system_logger import logger


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


def run_gc(supress_output: bool = True) -> None:
    """ Run Python's Garbage Collector and cleans up GPU cache (if GPU is available) """
    try:
        gc.collect()
    except Exception as e:
        logger.error(f'ERROR while gc.collect(): {e}')

    if torch.cuda.is_available():
        if not supress_output:
            logger.info(torch.cuda.memory_summary())

        torch.cuda.empty_cache()

        if not supress_output:
            logger.info('Cleaned CUDA cache')
            logger.info(torch.cuda.memory_summary())
    elif torch.mps.is_available():
        torch.mps.empty_cache()


def save_model_weights(model: nn.Module, file_path: str) -> None:
    torch.save(model.state_dict(), file_path)


def load_model_weights(model: nn.Module, file_path: str) -> nn.Module:
    model.load_state_dict(torch.load(file_path, weights_only=True, map_location=device('cpu')))
    model.eval()
    return model
