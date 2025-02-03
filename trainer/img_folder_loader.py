import os
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm

from utils.image_toolbox import resize_crop_and_pad, resize_and_pad
from imgproj.classifier.img_configuration import ModelConf


class ImgFolderLoader:
    def __init__(self):
        self.model_conf = ModelConf()  # Assuming ModelConf is defined elsewhere
        self.label: list[int] = list()
        self.file_name: list[str] = list()
        self.fqfn: list[str] = list()
        self.image_greyscale: list[np.ndarray] = list()
        self.image_dim: list[tuple[int, int]] = []  # (Height, Width)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(data={
            'file_name': self.file_name,
            'fqfn': self.fqfn,
            'img_grey': self.image_greyscale,
            'img_height': [dim[0] for dim in self.image_dim] ,
            'img_width': [dim[1] for dim in self.image_dim] ,
            'label': self.label
        })

    def _process_image(self, args: tuple[str, str, int]) -> tuple[str, str, np.ndarray|None, int, int, int]:
        """Helper function to process each image (grayscale conversion and resizing)"""
        folder_path, file_name, label = args
        fqfn = os.path.join(folder_path, file_name)

        image_grey = cv2.imread(fqfn, cv2.IMREAD_GRAYSCALE)
        scaled_image_grey = None
        img_height, img_width = (-1, -1)
        if image_grey is not None:
            resized_img = resize_crop_and_pad(image_grey, self.model_conf.intermediary_image_size)
            scaled_image_grey = resize_and_pad(resized_img, self.model_conf.image_size)
            img_height, img_width = scaled_image_grey.shape[:2]
        return file_name, fqfn, scaled_image_grey, img_height, img_width, label

    def read(self, folder_path: str, labels: list[str] = ['0', '1']) -> None:
        """Read image files from the given folder path and process them in parallel."""
        tasks: list[tuple[str, str, int]] = list()

        # Prepare the task list for multiprocessing
        for label in tqdm(labels, desc='Creating tasks', leave=True):
            label_folder = os.path.join(folder_path, label)
            if not os.path.isdir(label_folder):
                continue  # Skip if the folder does not exist

            for file_name in os.listdir(label_folder):
                if file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                    tasks.append((label_folder, file_name, int(label)))

        # Use multiprocessing to process images in parallel
        num_workers = min(cpu_count(), len(tasks))
        with Pool(processes=num_workers) as pool:
            for file_name, fqfn, scaled_image_grey, img_height, img_width, label in tqdm(
                    pool.imap_unordered(self._process_image, tasks), total=len(tasks), desc='Image processing'
            ):
                if scaled_image_grey is not None:
                    self.file_name.append(file_name)
                    self.fqfn.append(fqfn)
                    self.image_greyscale.append(scaled_image_grey)
                    self.image_dim.append((img_height, img_width))
                    self.label.append(label)
