from dataclasses import dataclass
from os import path

import fasttext
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedTokenizer

from system_logger import logger
from utils.compute_device import resolve_device_mapping
from utils.fs_utils import get_module_location
from utils.system_logger import logger

DSMODELS_PREFIX = path.join(get_module_location(), '..', 'ext_models')
MODEL_BERT_BASE = 'microsoft/graphcodebert-base'
MODEL_BERT_BASE_XLA = f'{MODEL_BERT_BASE}/graphcodebert.xlapt'
FQFP_MODEL_BERT_BASE = path.join(DSMODELS_PREFIX, MODEL_BERT_BASE)
FQFP_MODEL_BERT_BASE_XLA = path.join(DSMODELS_PREFIX, MODEL_BERT_BASE_XLA)

FAST_TEXT_MODEL = 'facebookresearch/fastText'
MODEL_FASTTEXT_D32 = 'cc.en.32.bin'
MODEL_FASTTEXT_D300 = 'cc.en.300.bin'
FQFP_MODEL_FASTTEXT_D32 = path.join(DSMODELS_PREFIX, FAST_TEXT_MODEL, MODEL_FASTTEXT_D32)


@dataclass(kw_only=True)
class GraphCodeBertConf:
    # default value of 514 from https://huggingface.co/microsoft/graphcodebert-base/blob/main/config.json is incorrect
    # and causing "CUDA error: device-side assert triggered" and "Assertion `srcIndex < srcSelectDimSize` failed"
    max_position_embeddings = 512


class LmComponents:
    def __init__(self, model_name: str, device_name: str = None):
        self.device_name, self.compute_device, self.tensor_device = resolve_device_mapping(device_name)

        self.model_name = model_name
        self.config: Optional[PretrainedConfig] = None
        self.llm: Optional[nn.Module] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def load(self, hf_token: str = None, output_hidden_states: bool = True) -> None:
        """
        :param hf_token: Hugging Face token to retrieve models requiring authorization
        :param output_hidden_states: whether output hidden state of the last layer from the LLM
            NOTE: output_hidden_states is True by default, as it is required by `compute_code_embeddings` function
        """
        logger.info(f'LM Components are being loaded from {self.model_name}...')
        self.config = AutoConfig.from_pretrained(self.model_name, token=hf_token)
        if output_hidden_states:
            self.config.output_hidden_states = True

        if self.device_name == 'xla':
            # DO NOT allocate model .to(device=xla)
            self.llm = torch.jit.load(FQFP_MODEL_BERT_BASE_XLA)
        else:
            self.llm = AutoModel.from_pretrained(self.model_name, config=self.config, token=hf_token)
            self.llm = self.llm.to(device=self.compute_device)
            if torch.cuda.is_available():
                self.llm.compile(mode='reduce-overhead')

        self.llm.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, config=self.config, token=hf_token, clean_up_tokenization_spaces=False
        )
        logger.info(f'LM Components successfully loaded for {self.device_name} compute architecture')


def load_ft_model(model_path: str) -> fasttext.FastText:
    try:
        logger.debug(f'Attempting to load FastText model from {model_path}')
        model = fasttext.load_model(model_path)
        logger.debug('Successfully loaded FastText model')
        return model
    except Exception as e:
        logger.error(f'Failed to load FastText model: {str(e)}')
        raise e
