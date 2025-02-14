import os
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers.tokenization_utils_base import BatchEncoding

from loraproj.classifier.lora_classifier import instantiate_tokenizer
from loraproj.classifier.lora_configuration import ModelConf


class LoraFolderLoader:
    def __init__(self):
        self.model_conf = ModelConf()
        self.label: list[int] = list()
        self.file_name: list[str] = list()
        self.fqfn: list[str] = list()
        self.text_body: list[str] = list()
        self.input_ids: list[int] = list()
        self.attention_mask: list[int] = list()
        self.token_type_ids: list[int] = list()

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(data={
            'file_name': self.file_name,
            'fqfn': self.fqfn,
            'text_body': self.text_body,
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'token_type_ids': self.token_type_ids,
            'label': self.label
        })

    def _process_text_file(self, args: tuple[str, str, int]) -> tuple[str, str, str, list[int], list[int], list[int], int]:
        """Helper function to process text file (embeddings computation)"""
        folder_path, file_name, label = args
        fqfn = os.path.join(folder_path, file_name)

        tokenizer = instantiate_tokenizer()
        with open(fqfn) as f:
            body = f.read()
            tokenized_output: BatchEncoding = tokenizer(body, truncation=True, padding='max_length', max_length=512)
            input_ids: list[int] = tokenized_output['input_ids']
            attention_mask: list[int] = tokenized_output['attention_mask']
            token_type_ids: list[int] = tokenized_output.get('token_type_ids')

        return file_name, fqfn, body, input_ids, attention_mask, token_type_ids, label

    def read(self, folder_path: str, labels: list[str] = ['0', '1']) -> None:
        """Read test files from the given folder path and process them in parallel."""
        tasks: list[tuple[str, str, int]] = list()

        # Prepare the task list for multiprocessing
        for label in tqdm(labels, desc='Creating tasks', leave=True):
            label_folder = os.path.join(folder_path, label)
            if not os.path.isdir(label_folder):
                # Skip if the folder does not exist
                continue

            for file_name in os.listdir(label_folder):
                if file_name.endswith('.ps1'):
                    tasks.append((label_folder, file_name, int(label)))

        # Use multiprocessing to process images in parallel
        num_workers = min(cpu_count(), len(tasks))
        with Pool(processes=num_workers) as pool:
            for file_name, fqfn, body, input_ids, attention_mask, token_type_ids, label in tqdm(
                pool.imap_unordered(self._process_text_file, tasks), total=len(tasks), desc='Text files processing'
            ):
                self.file_name.append(file_name)
                self.fqfn.append(fqfn)
                self.text_body.append(body)
                self.input_ids.append(input_ids)
                self.attention_mask.append(attention_mask)
                self.token_type_ids.append(token_type_ids)
                self.label.append(label)
