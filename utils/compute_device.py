import torch
from utils.system_logger import logger

try:
    # Attempt to import torch_xla for XLA (Accelerated Linear Algebra) device support
    import torch_xla.core.xla_model as xm
    xla_devices = xm.get_xla_supported_devices()
    num_xla_cores = len(xla_devices)
except Exception as e:
    logger.info(f'XLA Device Not Supported: {e}')
    xla_available = False
    num_xla_cores = 0


PREFERRED_DEVICE = 'cpu'
DEVICES: dict[str, torch.device] = {
    'cpu': torch.device('cpu')
}
if torch.backends.mps.is_available():
    DEVICES['mps'] = torch.device('mps')
    PREFERRED_DEVICE = 'mps'
if num_xla_cores:
    DEVICES['xla'] = xm.xla_device()
    PREFERRED_DEVICE = 'xla'
    logger.info(f'Num XLA Devices Available: {num_xla_cores}')
if torch.cuda.is_available():
    DEVICES['cuda'] = torch.device('cuda')
    PREFERRED_DEVICE = 'cuda'
    logger.info(f'Num GPUs Available: {torch.cuda.device_count()}')

logger.info(
    f'Pytorch version={torch.__version__} preferred device={PREFERRED_DEVICE} '
    f'build with MPS support={torch.backends.mps.is_built()}'
)


def resolve_device_mapping(device_name: str) -> tuple[str, torch.device, torch.device]:
    if not device_name or device_name not in DEVICES:
        device_name = PREFERRED_DEVICE

    compute_device = DEVICES[device_name]
    tensor_device = DEVICES['cpu'] if device_name == 'xla' else compute_device
    logger.info(f'resolved device_name: {device_name} compute_device: {compute_device} tensor_device: {tensor_device}')
    return device_name, compute_device, tensor_device
