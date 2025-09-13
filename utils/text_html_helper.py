import re
from typing import Optional
from bs4 import BeautifulSoup
from unidecode import unidecode

# This regex pattern finds and captures sequences of Japanese-Chinese-Korean Ideographs,
# Japanese Hiragana/Katakana, and Korean Hangul characters.
# The parentheses (...) make it a capturing group, which is key for re.split().
LOGOGRAM_PATTERN = re.compile(
    r'([\u4e00-\u9fff'  # JCK Unified Ideographs (most common Chinese characters)
    r'\u3040-\u309f'    # Japanese Hiragana
    r'\u30a0-\u30ff'    # Japanese Katakana
    r'\uac00-\ud7a3'    # Korean Hangul Syllables
    r']+)'
)

# This pattern now finds lines containing ONLY horizontal whitespace.
# The ^ and $ anchors match the start and end of a line due to the re.MULTILINE flag.
WHITESPACE_ONLY_LINE_RE = re.compile(
    r'^[ \t\u00A0\u2000-\u200A\u202F\u205F\u3000]+$', re.MULTILINE
)

# Whitespace pattern includes a range of common typographic spaces
# from the Unicode General Punctuation block, plus the Ideographic Space.
COMPREHENSIVE_HORIZONTAL_WHITESPACE_RE = re.compile(
    r'['
    r' \t\u00A0'            # Standard space, tab, non-breaking space
    r'\u2000-\u200A'        # En Quad, Em Quad, various typographic spaces
    r'\u202F\u205F\u3000'   # Narrow No-Break Space, Medium Math Space, Ideographic Space
    r']+'
)
VERTICAL_WHITESPACE_RE = re.compile(r'[\n\r]+')


def normalize_whitespace_with_newlines(text: str) -> str:
    """
    Normalizes whitespace in a string using pre-compiled regex patterns.
    """
    # Step 1: Find lines containing only whitespace and remove their content.
    # This turns a newline followed by a whitespace-only line into two consecutive newlines.
    text_step1 = WHITESPACE_ONLY_LINE_RE.sub('', text)

    # Step 2: Collapse consecutive vertical whitespace (now includes the newly empty lines).
    text_step2 = VERTICAL_WHITESPACE_RE.sub('\n', text_step1)

    # Step 3: Collapse all remaining horizontal whitespace within lines of text.
    text_step3 = COMPREHENSIVE_HORIZONTAL_WHITESPACE_RE.sub(' ', text_step2)

    # Step 4: Strip any leading/trailing whitespace from the final result.
    normalized_text = text_step3.strip()
    return normalized_text


def clean_text_selective(text: Optional[str], clean_html: bool = True, logogram_selectiveness: bool = True) -> Optional[str]:
    """
    Cleans a string by removing HTML and normalizing whitespace.
    :param logogram_selectiveness: if True then transliterates characters to ASCII BUT preserves logograms (e.g., Chinese, Japanese).
            otherwise: transliterates everything to characters to ASCII.
    """
    if not text:
        return text

    # HTML Cleaning
    if clean_html:
        soup = BeautifulSoup(text, features=['html.parser'])
        text = soup.get_text(separator=' ')

    if not logogram_selectiveness:
        # Global Transliteration
        decoded_text = unidecode(text)
    else:
        # Selective Transliteration
        # Split the string by logograms, keeping the logograms in the list
        parts = LOGOGRAM_PATTERN.split(text)

        # Process each part: apply unidecode only to the non-logogram parts
        processed_parts = []
        for part in parts:
            # If the part is a logogram (it will match the LOGOGRAM_PATTERN pattern), keep it as is.
            # Otherwise, apply unidecode transliteration.
            if LOGOGRAM_PATTERN.fullmatch(part):
                processed_parts.append(part)
            else:
                processed_parts.append(unidecode(part))

        # Join the parts back together
        decoded_text = ''.join(processed_parts)

    # Normalize whitespace
    normalized_text = normalize_whitespace_with_newlines(decoded_text)
    return normalized_text


if __name__ == '__main__':
    fixtures: list[str] = [
        "Café   <B>olé!</b> &amp; other strange characters like—(’你好’)",
        "Weird «КАФЕ» chars — “double quotes” and ‘single quotes’.",
        "Café olé—你好, this is a test with scripts like かたかな.",
        """line 1
        line 2
        
        line 4   
        """,
        "A short\u2003message\u3000from our sponsor.",
        "\n\n  Hello\t\tWorld  \n\r\nThis is a test. \n ",
    ]
    for fixture_str in fixtures:
        cleaned_html = clean_text_selective(fixture_str, clean_html=True)

        print(f'Original HTML: {fixture_str}')
        print(f'Cleaned HTML:  {cleaned_html}')
        print('-' * 20)
