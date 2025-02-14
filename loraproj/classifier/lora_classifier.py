import types

import torch
from peft import LoraConfig, get_peft_model, PeftModel, PeftMixedModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from utils.compute_device import resolve_device_mapping
from utils.lm_components import MODEL_BERT_BASE

# Global variables for multi-processing
_lora_tokenizer = None
_lora_model = None


def instantiate_tokenizer(model_name: str = MODEL_BERT_BASE) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    global _lora_tokenizer
    if _lora_tokenizer is None:
        # Only instantiate if it hasn't been done in this process
        _lora_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _lora_tokenizer


def instantiate_lora_model(model_name: str = MODEL_BERT_BASE, device_name: str = 'cpu') -> PeftModel | PeftMixedModel:
    global _lora_model
    if _lora_model is None:
        device_name, compute_device, tensor_device = resolve_device_mapping(device_name)

        # Only instantiate if it hasn't been done in this process
        # Load GraphCodeBERT as a sequence classification model (2 classes: benign vs malicious)
        classification_model = AutoModelForSequenceClassification.from_pretrained(MODEL_BERT_BASE, num_labels=2)

        # Apply LoRA to GraphCodeBERT
        lora_config = LoraConfig(
            r = 8,  # Low-rank matrices size
            lora_alpha = 32,
            lora_dropout = 0.1,
            target_modules = ['query', 'key', 'value'],  # Apply LoRA to self-attention layers
        )

        _lora_model = get_peft_model(classification_model, lora_config)
        _lora_model.to(device=compute_device)
        print(f'LoRA model={model_name} on {device_name} with #trainable_parameters={_lora_model.print_trainable_parameters()}')
    return _lora_model

