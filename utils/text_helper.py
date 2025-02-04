import re

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.language import Language
from spacy.tokens import Doc

nlp = spacy.load('en_core_web_sm')

REGEX_ALPHANUM = re.compile(r"\b(?=[^\W\d_]*\d)(?=\d*[^\W\d_])[^\W_]{4,}\b")
REGEX_LONG_WORD = re.compile(r'[A-Za-z0-9/_\-+]{16,}')
REGEX_PATH = re.compile(r'/([a-zA-Z0-9_-]+/?)+')


@Language.component('text_tokenizer')
def text_tokenizer(doc: Doc) -> Doc:
    processed_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            # Remove stopwords and punctuation, special characters
            continue

        # Perform lemmatization and cast to lowercase
        lemma = token.lemma_.lower()

        # Replace common text patterns with placeholders
        if token.like_num:
            lemma = '<num>'
        elif token.like_url:
            lemma = '<url>'
        elif token.like_email:
            lemma = '<email>'
        elif REGEX_LONG_WORD.match(token.text):
            lemma = '<longword>'
        elif REGEX_PATH.match(token.text):
            lemma = '<path>'
        elif REGEX_ALPHANUM.match(token.text):
            lemma = '<alphanum>'

        # Add processed token to the list
        if lemma.strip():  # Check if the lemma is not an empty string
            processed_tokens.append(lemma)

    # Create a new spaCy Doc with the processed tokens
    if processed_tokens:
        return Doc(doc.vocab, words=processed_tokens)
    else:
        return Doc(doc.vocab, words=['<empty>'])  # Handle case where all tokens are removed

# Add the custom tokenizer component to the pipeline
nlp.add_pipe('text_tokenizer', last=True)


def process_text(text: str) -> str:
    # Apply the spaCy pipeline to the text
    doc = nlp(text)

    # Join tokens back into a string
    processed_text = ' '.join([token.text for token in doc])

    return processed_text


def vectorize(vocabulary: list[str], text: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(vocabulary=vocabulary, lowercase=True, dtype=np.float32)

    X_tfidf = vectorizer.fit_transform(text)

    # Convert the sparse matrix to an array
    text_vectors = X_tfidf.toarray()
    return text_vectors


if __name__ == '__main__':
    texts = [
        "",
        "Hello, world! This is a simple test case.",
        "The price of the item is $19.99, and the order ID is 987654321.",
        """
        Once upon a time, in a faraway land, there existed a kingdom full of mystery and wonder.
        The people in this land spoke in poetic verses, recounting tales of old.
        """,
        """<html><body><p>
        This is an HTML-like string with tags, URL https://www.example.com and an email support@example.com... 
        </p></body></html>""",
        """# a8f7s9d8f7g9h0a8d7s6f5g4h3j2k1l0m9n8b7v6c5x4z3a2s1d0f9g
        def example_function(x): return x * 2""",
    ]

    for text in texts:
        print(f'Original: {text}')
        print(f'Processed: {process_text(text)}')
