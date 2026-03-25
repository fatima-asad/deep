"""
Utility functions shared across deep learning notebooks.
"""

import re
import string
import numpy as np


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace from text."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_sequences(texts: list, tokenizer, max_len: int) -> np.ndarray:
    """
    Convert a list of text strings to zero-padded integer sequences.

    Args:
        texts: List of raw text strings.
        tokenizer: A fitted Keras Tokenizer instance.
        max_len: Maximum sequence length (sequences are truncated/padded).

    Returns:
        NumPy array of shape (len(texts), max_len).
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")


def load_json_dataset(filepath: str) -> list:
    """Load a JSON-lines dataset file and return a list of records."""
    import json

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
