from os import path

from adapters import BnConfig, AutoAdapterModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from llmadapterproj.classifier.llmadapter_configuration import ModelConf
from utils.compute_device import resolve_device_mapping
from utils.lm_components import MODEL_BERT_BASE
from utils.system_logger import logger

ADAPTER_PATH = './llmadapter_graphcodebert_powershell'
ADAPTER_NAME = 'powershell_classification'

# Global variables for multi-processing
_llmadapter_tokenizer = None
_llmadapter_model = None


def instantiate_tokenizer(model_name: str = MODEL_BERT_BASE) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    global _llmadapter_tokenizer
    if _llmadapter_tokenizer is None:
        # Only instantiate if it hasn't been done in this process
        _llmadapter_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _llmadapter_tokenizer


def instantiate_llmadapter_model(model_name: str = MODEL_BERT_BASE, device_name: str = 'cpu') -> AutoAdapterModel:
    global _llmadapter_model
    if _llmadapter_model is None:
        # Only instantiate if it hasn't been done in this process
        device_name, compute_device, tensor_device = resolve_device_mapping(device_name)

        # Load GraphCodeBERT (or any model) as an adapter-supported model
        _llmadapter_model = AutoAdapterModel.from_pretrained(model_name)

        # Define Adapter configuration
        model_conf = ModelConf()
        adapter_config = BnConfig(
            mh_adapter=True,        # Apply adapters to multi-head attention layers
            output_adapter=True,    # Add adapter layers to the model output
            reduction_factor=16,    # Controls adapter size (higher = more compression)
            non_linearity='relu',   # Activation function inside adapters
        )

        # Add a new adapter named 'powershell_classification'
        _llmadapter_model.add_adapter(ADAPTER_NAME, config=adapter_config)

        # Add a classification head to use the adapter (2 labels: benign vs malicious)
        _llmadapter_model.add_classification_head(ADAPTER_NAME, num_labels=2)

        # Activate the adapter (only adapter parameters are trained)
        _llmadapter_model.set_active_adapters(ADAPTER_NAME)
        _llmadapter_model.to(device=compute_device)

        trainable_params = sum(p.numel() for p in _llmadapter_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in _llmadapter_model.parameters())
        logger.info(f'LLM Adapter model={model_name} on {device_name} with '
              f'(#trainable_parameters, #all_parameters)={trainable_params, total_params}')
    return _llmadapter_model



def load_model_with_adapter(
    model_name: str = MODEL_BERT_BASE, device_name: str = 'cpu'
) -> tuple[AutoAdapterModel, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    """
    Load the base model (AutoAdapterModel) and then load a trained adapter from disk,
    attaching it to the model and setting it active for inference.
    """
    # Load adapter-supported model
    model = AutoAdapterModel.from_pretrained(model_name)

    # Load the trained adapter by name
    model.load_adapter(path.join(ADAPTER_PATH, ADAPTER_NAME), load_as=ADAPTER_NAME)

    # Activate the adapter
    model.set_active_adapters(ADAPTER_NAME)
    model = model.eval()

    # Move to preferred device
    device_name, compute_device, tensor_device = resolve_device_mapping(device_name)
    model = model.to(compute_device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    return model, tokenizer
