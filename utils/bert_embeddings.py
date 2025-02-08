import gc
from contextlib import nullcontext
from typing import Any

import torch
from langchain import text_splitter
from langchain_text_splitters import Language

from utils.lm_components import LmComponents, GraphCodeBertConf
from utils.system_logger import logger


class PowershellTextSplitter(text_splitter.RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any) -> None:
        separators = self.get_separators_for_language(Language.POWERSHELL)
        super().__init__(separators=separators, **kwargs)


def tensor_info(description: str, t: torch.Tensor) -> None:
    logger.info(
        f'{description} tensor=(dtype:{t.dtype}, shape:{tuple(t.shape)}, device:{t.device}, is_cuda:{t.is_cuda})'
    )


def run_gc() -> None:
    """ Run Python's Garbage Collector and (if CUDA is available) clean up GPU cache. """
    try:
        gc.collect()
    except Exception as e:
        logger.error(f'ERROR while gc.collect(): {e}')

    try:
        if torch.cuda.is_available():
            logger.info(torch.cuda.memory_summary())
            torch.cuda.empty_cache()
            logger.info('Cleaned CUDA cache')
            logger.info(torch.cuda.memory_summary())
    except Exception as e:
        logger.error(f'ERROR while trying to clean CUDA cache: {e}')


def _chunk_text_with_overlap(text_body: str, max_length: int = 512, overlap: int = 50) -> list[str]:
    splitter = PowershellTextSplitter(chunk_size=max_length, chunk_overlap=overlap, keep_separator=False)
    return splitter.split_text(text_body)


def _compute_chunk_embeddings(ml_components: LmComponents, chunk_of_code: str, remove_batch_dim: bool) -> torch.Tensor:
    # max sequence length of tokens the model can process. for instance 512
    max_length = GraphCodeBertConf.max_position_embeddings
    if len(chunk_of_code) > max_length:
        logger.warning(
            f'compute_code_embeddings: chunk_of_code will be truncated from {len(chunk_of_code)} to {max_length}')

    try:
        # padding=True: Pads sequences to the length of the longest sequence in the batch.
        # padding='max_length': Pads sequences to the max_length specified, regardless of the actual input length.
        padding_strategy = 'max_length' if ml_components.device_name == 'xla' else True

        # inputs is a dict with keys: [input_ids, attention_mask]
        inputs = ml_components.tokenizer(
            chunk_of_code, return_tensors='pt', max_length=max_length, truncation=True, padding=padding_strategy
        )

        input_ids = inputs['input_ids'].to(device=ml_components.tensor_device)
        attention_mask = inputs['attention_mask'].to(device=ml_components.tensor_device)

        # Choose the appropriate context based on CUDA availability
        autocast_context = torch.cuda.amp.autocast if torch.cuda.is_available() else nullcontext
        with torch.no_grad(), autocast_context():
            # model's input and output parameters
            # https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertForPreTraining
            # do not use named arguments (input_ids=input_ids, attention_mask=attention_mask) as it is not supported by XLA
            outputs = ml_components.llm(input_ids, attention_mask)

        if ml_components.device_name == 'xla':
            # BertModelWrapper returns `last_hidden_state` so no further extraction is needed
            embeddings = outputs
        else:
            # The last layer's hidden states is at the last index (-1) of the hidden_states
            hidden_states = outputs.hidden_states
            embeddings = hidden_states[-1]

        # `embeddings` has the shape (batch_size, sequence_length, hidden_size)
        # to access embeddings for the [CLS] token:
        cls_embeddings = embeddings[:, 0, :]

        # torch.cuda.amp.autocast will likely change the dtype of the hidden states to float16
        cls_embeddings = cls_embeddings.to(dtype=torch.float32)
    except Exception as e:
        logger.exception(f'ERROR during embeddings computation: "{e}" for "{chunk_of_code}"')
        run_gc()
        cls_embeddings = torch.zeros(size=(ml_components.config.hidden_size,), device=ml_components.tensor_device)

    if cls_embeddings.shape[0] == 1 and remove_batch_dim:
        # `squeeze(0)` removes batch dimension: [1, sequence_length, hidden_size] -> [sequence_length, hidden_size]
        cls_embeddings = cls_embeddings.squeeze(dim=0)

    # tensor_info('bert_embeddings._compute_chunk_embeddings', cls_embeddings)
    return cls_embeddings


def _aggregate_embeddings(ml_components: LmComponents, embeddings_list: list[torch.Tensor]) -> torch.Tensor:
    if not embeddings_list:
        # hidden_size stands for the size of individual token
        return torch.zeros(size=(ml_components.config.hidden_size,), device=ml_components.tensor_device)

    with torch.no_grad():
        # Stack embeddings and compute the mean
        stacked_embeddings = torch.stack(embeddings_list)
        aggregated_embedding = torch.mean(stacked_embeddings, dim=0)
    # tensor_info('bert_embeddings._aggregate_embeddings', aggregated_embedding)
    return aggregated_embedding


def compute_bert_embeddings(ml_components: LmComponents, text_body: str, remove_batch_dim: bool = False) -> torch.Tensor:
    # max sequence length of tokens the model can process. for instance 512
    max_length = GraphCodeBertConf.max_position_embeddings

    text_body_chunks = _chunk_text_with_overlap(text_body, max_length=max_length, overlap=0)
    chunk_embeddings = [_compute_chunk_embeddings(ml_components, chunk, remove_batch_dim=remove_batch_dim)
                        for chunk in text_body_chunks]
    aggregated_embedding = _aggregate_embeddings(ml_components, chunk_embeddings)
    return aggregated_embedding
