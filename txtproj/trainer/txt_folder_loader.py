import os
from collections import Counter

import pandas as pd
from bs4 import BeautifulSoup

from utils.text_nlp_helper import process_text, vectorize
from txtproj.classifier.txt_configuration import ModelConf


class TextFolderLoader:
    def __init__(self):
        self._word_counter: Counter = Counter()
        self.X: list[str] = []
        self.label: list[int] = []
        self.file_names: list[str] = []
        self._vocabulary: list[str] = []

    @property
    def vocabulary(self) -> list[str]:
        """Return the sorted list of top `ModelConf.vocabulary_size` words found in the documents."""
        if not self._vocabulary:
            most_common_words = [word for word, _ in self._word_counter.most_common(ModelConf.vocabulary_size)]
            self._vocabulary = sorted(most_common_words)
        return self._vocabulary

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(data={
            'file_name': self.file_names,
            'text': self.X,
            'text_tfidf': vectorize(self.vocabulary, self.X).tolist(),
            'label': self.label
        })

    def read(
        self, folder_path: str, labels: list[str] = ['0', '1'], positive_labels: list[str] = ['1'], extensions: tuple[str, ...] = ('.txt', )
    ) -> None:
        """Read Textual files from the given folder path and builds the vocabulary."""
        for label in labels:
            label_folder = os.path.join(folder_path, label)
            if not os.path.isdir(label_folder):
                # Skip if the folder does not exist
                continue

            for filename in os.listdir(label_folder):
                if filename.endswith(extensions):
                    self.file_names.append(filename)
                    file_path = os.path.join(label_folder, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        document = file.read()

                        # Parse HTML content and extract text using BeautifulSoup
                        soup = BeautifulSoup(document, features='html.parser')
                        extracted_text = soup.get_text(separator=' ')  # Extract text and separate by spaces
                        processed_document = process_text(extracted_text)  # Process the extracted text

                        self.X.append(processed_document)
                        self.label.append(int(label))

                        if label in positive_labels:
                            # Update the word counter with words from this document
                            self._word_counter.update(processed_document.split())
