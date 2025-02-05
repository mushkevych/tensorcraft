import base64
import math
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

import fasttext
import numpy as np

REGEX_BASE64_STR = re.compile(r"""["'\s]?(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?["'\s]?""")
REGEX_DELIMITERS = re.compile(r"[_\- .\\]")

BASE64_DECODED_STR_LENGTH_THRESHOLD = 32
BASE64_ENCODED_STR_LENGTH_THRESHOLD = 44


def is_64base_content_present(text_body: str) -> bool:
    for line in text_body:
        matches = REGEX_BASE64_STR.findall(line)
        for match in matches:
            if len(match) < BASE64_ENCODED_STR_LENGTH_THRESHOLD:
                continue

            try:
                unquoted_string = match.replace("'", "").replace('"', '').strip()
                decoded_bytes = base64.b64decode(unquoted_string, validate=True)
                if len(decoded_bytes) >= BASE64_DECODED_STR_LENGTH_THRESHOLD:
                    return True
            except Exception:
                # If there's an error in decoding, skip this token.
                pass
    return False


def script_attributes(text_body: str) -> tuple[int, int, int, int]:
    """
    computes several attributes of the script
    :return: tuple[longest_code_line_length, median_code_line_length, lines_of_code, code_size_in_bytes]
    """
    line_lengths: list[int] = list()
    for line in text_body:
        hash_index = line.find('#')
        if hash_index == -1:
            line_lengths.append(len(line))

        if hash_index != -1:
            line_lengths.append(hash_index)

    if line_lengths:
        return max(line_lengths), int(statistics.median(line_lengths)), len(line_lengths), sum(line_lengths)
    else:
        return 0, 0, 0, 0


def script_attributes_log_scale(text_body: str) -> tuple[float, float, float, float]:
    """ script attributes are scaled using a logarithm """
    longest_code_line_length, median_code_line_length, lines_of_code, code_size_in_bytes = script_attributes(
        text_body)
    # +1 is a handler of possible 0 value
    return (math.log2(longest_code_line_length + 1),
            math.log2(median_code_line_length + 1),
            math.log2(lines_of_code + 1),
            math.log2(code_size_in_bytes + 1))


def ratio_of_comments_to_code(text_body: str) -> float:
    num_code_lines: int = 1
    num_comment_lines: int = 0

    for line in text_body:
        clean_line = line.strip()
        if not clean_line:
            # skip empty lines
            continue

        if clean_line.startswith('#') or clean_line.startswith('<#'):
            num_comment_lines += 1
        elif '#' in clean_line:
            num_comment_lines += 1  # this is a trailing comment
            num_code_lines += 1
        else:
            num_code_lines += 1

    if num_comment_lines:
        log_comment = math.log2(num_comment_lines)
    else:
        log_comment = -1.0

    log_code = math.log2(num_code_lines)
    if log_code == 0:
        log_code = 1

    ratio = round(log_comment / log_code, ndigits=2)
    return ratio


def _tokenize_text(file_name: str) -> list[str]:
    tokens: list[str] = REGEX_DELIMITERS.split(file_name)
    # Filter out any empty tokens resulting from consecutive delimiters
    return [token for token in tokens if token]


def compute_ft_embeddings(text: str, model: fasttext.FastText) -> np.ndarray:
    """Generate an embedding for the input text using a FastText model and custom tokenizer."""
    if text:
        embeddings: list[np.ndarray] = list()

        for token in _tokenize_text(text):
            embeddings.append(model.get_word_vector(token))
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.get_dimension())
