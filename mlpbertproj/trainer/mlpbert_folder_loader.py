import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.auto import tqdm

from mlpbertproj.classifier.mlpbert_configuration import ModelConf
from utils.lm_core import instantiate_ml_components
from utils.bert_embeddings import compute_bert_embeddings


class MlpBertFolderLoader:
    def __init__(self):
        self.model_conf = ModelConf()
        self.label: list[int] = list()
        self.file_name: list[str] = list()
        self.fqfn: list[str] = list()
        self.text_body: list[str] = list()
        self.text_embeddings: list[np.ndarray] = list()

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(data={
            'file_name': self.file_name,
            'fqfn': self.fqfn,
            'text_body': self.text_body,
            'text_embeddings': self.text_embeddings,
            'label': self.label
        })

    def _process_text_file(self, args: tuple[str, str, int]) -> tuple[str, str, str, np.ndarray | None, int]:
        """Helper function to process text file (embeddings computation)"""
        folder_path, file_name, label = args
        fqfn = os.path.join(folder_path, file_name)

        ml_components = instantiate_ml_components()
        with open(fqfn) as f:
            body = f.read()
            pt_embeddings: torch.Tensor = compute_bert_embeddings(ml_components, body, remove_batch_dim=True)

        np_embeddings = pt_embeddings.cpu().numpy()
        return file_name, fqfn, body, np_embeddings, label

    def read(self, folder_path: str, labels: list[str] = ['0', '1'], extensions: tuple[str, ...] = ('.ps1', )) -> None:
        """Read text files from the given folder path and process them in parallel."""
        tasks: list[tuple[str, str, int]] = list()

        # Prepare the task list for multiprocessing
        for label in tqdm(labels, desc='Creating tasks', leave=True):
            label_folder = os.path.join(folder_path, label)
            if not os.path.isdir(label_folder):
                continue  # Skip if the folder does not exist

            for file_name in os.listdir(label_folder):
                if file_name.endswith(extensions):
                    tasks.append((label_folder, file_name, int(label)))

        # Use multiprocessing to process images in parallel
        num_workers = min(cpu_count(), len(tasks))
        with Pool(processes=num_workers) as pool:
            for file_name, fqfn, body, embeddings, label in tqdm(
                pool.imap_unordered(self._process_text_file, tasks), total=len(tasks), desc='Text files processing'
            ):
                if embeddings is not None:
                    self.file_name.append(file_name)
                    self.fqfn.append(fqfn)

                    self.text_body.append(body)
                    self.text_embeddings.append(embeddings)
                    self.label.append(label)
