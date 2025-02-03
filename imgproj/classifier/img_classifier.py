import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from imgproj.classifier.img_configuration import ModelConf


class ImgClassifier(nn.Module):
    def __init__(self, model_conf: ModelConf):
        super(ImgClassifier, self).__init__()

        # Load EfficientNet-B0 with pretrained weights, adapting to grayscale images
        self.model = EfficientNet.from_pretrained(
            model_name='efficientnet-b0', num_classes=1, **model_conf.eff_net_overrides
        )

        # Modify the first convolution layer to accept grayscale (1 channel instead of 3)
        self.model._conv_stem = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor):
        # EfficientNet models expect their inputs to be float tensors of pixels with values in the [0-255] range.
        return self.model(x)

    def load_model_weights(self, file_path: str) -> None:
        self.load_state_dict(torch.load(file_path, weights_only=True, map_location=torch.device('cpu')))
        self.eval()

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
