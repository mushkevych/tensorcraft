import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm
from tqdm.auto import tqdm

from trainer.lm_core import instantiate_ft
from xplainproj.classifier.textcode_features import script_attributes_log_scale, ratio_of_comments_to_code, \
    is_64base_content_present, compute_ft_embeddings
from xplainproj.classifier.xplain_configuration import ModelConf


class XplainFolderLoader:
    def __init__(self):
        self.model_conf = ModelConf()
        self.file_name: list[str] = list()
        self.fqfn: list[str] = list()
        self.text_body: list[str] = list()
        self.label: list[int] = list()

        self.longest_code_line_length: list[float] = list()
        self.median_code_line_length: list[float] = list()
        self.lines_of_code: list[float] = list()
        self.code_size_in_bytes: list[float] = list()
        self.ratio_of_comments_to_code: list[float] = list()
        self.is_64base_content_present: list[float] = list()
        self.file_name_embedding: list[np.ndarray] = []

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(data={
            'file_name': self.file_name,
            'fqfn': self.fqfn,
            'text_body': self.text_body,
            'label': self.label,

            'longest_code_line_length': self.longest_code_line_length,
            'median_code_line_length': self.median_code_line_length,
            'lines_of_code': self.lines_of_code,
            'code_size_in_bytes': self.code_size_in_bytes,
            'ratio_of_comments_to_code': self.ratio_of_comments_to_code,
            'is_64base_content_present': self.is_64base_content_present,
            'file_name_embedding': self.file_name_embedding,
        })

    def _process_text_file(self, args: tuple[str, str, int]) -> tuple[
        str, ndarray, str, str, float, float, float, float, float, bool, int]:
        """Helper function to process text file (embeddings computation)"""
        folder_path, file_name, label = args
        fqfn = os.path.join(folder_path, file_name)

        ft_model = instantiate_ft()
        file_name_embeddings: np.ndarray = compute_ft_embeddings(file_name, ft_model)
        with open(fqfn) as f:
            body = f.read()
            longest_code_line_length, median_code_line_length, lines_of_code, code_size_in_bytes = \
                script_attributes_log_scale(body)
            ratio_of_comments_to_code_value = ratio_of_comments_to_code(body)
            is_64base_content_present_value = is_64base_content_present(body)

        return (
            file_name,
            file_name_embeddings,
            fqfn,
            body,
            longest_code_line_length,
            median_code_line_length,
            lines_of_code,
            code_size_in_bytes,
            ratio_of_comments_to_code_value,
            is_64base_content_present_value,
            label
        )

    def read(self, folder_path: str, labels: list[str] = ['0', '1']) -> None:
        """Read test files from the given folder path and process them in parallel."""
        tasks: list[tuple[str, str, int]] = list()

        # Prepare the task list for multiprocessing
        for label in tqdm(labels, desc='Creating tasks', leave=True):
            label_folder = os.path.join(folder_path, label)
            if not os.path.isdir(label_folder):
                continue  # Skip if the folder does not exist

            for file_name in os.listdir(label_folder):
                if file_name.endswith('.ps1'):
                    tasks.append((label_folder, file_name, int(label)))

        # Use multiprocessing to process text bodies in parallel
        num_workers = min(cpu_count(), len(tasks))
        with Pool(processes=num_workers) as pool:
            for tuple_results in tqdm(pool.imap_unordered(self._process_text_file, tasks), total=len(tasks), desc='Test files processing'):
                (
                    file_name,
                    file_name_embeddings,
                    fqfn,
                    body,
                    longest_code_line_length,
                    median_code_line_length,
                    lines_of_code,
                    code_size_in_bytes,
                    ratio_of_comments_to_code_value,
                    is_64base_content_present_value,
                    label
                ) = tuple_results

                self.file_name.append(file_name)
                self.fqfn.append(fqfn)
                self.text_body.append(body)
                self.label.append(label)

                self.longest_code_line_length.append(longest_code_line_length)
                self.median_code_line_length.append(median_code_line_length)
                self.lines_of_code.append(lines_of_code)
                self.code_size_in_bytes.append(code_size_in_bytes)
                self.ratio_of_comments_to_code.append(ratio_of_comments_to_code_value)
                self.is_64base_content_present.append(is_64base_content_present_value)
                self.file_name_embedding.append(file_name_embeddings)
