import torch
import torch.nn as nn

from mlpbertproj.classifier.mlpbert_configuration import ModelConf


class MlpBertModel(nn.Module):
    """ Multi-layer Perceptron classifier that takes BERT embeddings as input """
    def __init__(self, model_conf: ModelConf):
        super(MlpBertModel, self).__init__()

        # Adjusting the layer sizes for a more gradual reduction
        self.layer1 = nn.Linear(model_conf.input_size, model_conf.hidden_sizes[0])          # First reduction
        self.layer2 = nn.Linear(model_conf.hidden_sizes[0], model_conf.hidden_sizes[1])     # Second reduction
        self.output_layer = nn.Linear(model_conf.hidden_sizes[1], model_conf.output_size)   # Final layer for output

        self.leaky_relu = nn.LeakyReLU(negative_slope=model_conf.negative_slope)
        self.dropout = nn.Dropout(model_conf.dropout)

        # Initialize layers
        self._init_weights(self.layer1, model_conf.negative_slope)
        self._init_weights(self.layer2, model_conf.negative_slope)
        self._init_weights(self.output_layer, model_conf.negative_slope)

    def _init_weights(self, layer: nn.Linear, negative_slope: float):
        nn.init.kaiming_normal_(layer.weight, a=negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(layer.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Applying fully connected layers
        x = self.layer1(embeddings)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        logits = self.output_layer(x)
        return logits

    def load_model_weights(self, file_path: str) -> None:
        self.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
        self.eval()

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
