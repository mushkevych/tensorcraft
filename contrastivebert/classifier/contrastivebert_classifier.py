import torch
import torch.nn as nn

from contrastivebert.classifier.contrastivebert_configuration import ModelConf
from utils.lm_components import LmComponents
from utils.lm_core import instantiate_ml_components


class ContrastiveSBERT(nn.Module):
    def __init__(self, model_conf: ModelConf):
        super(ContrastiveSBERT, self).__init__()
        self.ml_components: LmComponents = instantiate_ml_components()
        self.llm: nn.Module = self.ml_components.llm

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (Tensor): token IDs of shape (batch_size, seq_len)
            attention_mask (Tensor): attention mask of shape (batch_size, seq_len)

        Returns:
            Tensor: pooled [CLS] embeddings of shape (batch_size, hidden_size)
        """
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use pooler_output for dense representation
        return outputs.pooler_output

    def load_model_weights(self, file_path: str) -> None:
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
        self.eval()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
