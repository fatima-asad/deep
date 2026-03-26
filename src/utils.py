"""Shared utility functions for deep learning NLP notebooks."""

import json
import string

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def clean_text(text: str) -> str:
    """Lowercase and remove punctuation from a string.

    Parameters
    ----------
    text:
        Raw input string.

    Returns
    -------
    str
        Cleaned string with all characters lowercased and punctuation
        replaced by spaces.
    """
    text = text.lower()
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return text.translate(translator)


def build_sequences(
    texts,
    vocab_size: int = 1000,
    padding: str = "pre",
):
    """Tokenize texts and build padded n-gram input/output sequence pairs.

    Each sentence is split into overlapping n-gram sequences where the
    last token is the prediction target and the preceding tokens form the
    input sequence.

    Parameters
    ----------
    texts:
        Iterable of raw text strings (e.g. a pandas Series or list).
    vocab_size:
        Maximum vocabulary size passed to ``Tokenizer(num_words=...)``.
    padding:
        Padding strategy forwarded to ``pad_sequences`` (``"pre"`` or
        ``"post"``).

    Returns
    -------
    X_padded : np.ndarray, shape (n_samples, max_len)
        Padded input sequences.
    y : np.ndarray, shape (n_samples,)
        Integer output tokens (next-word indices).
    tokenizer : Tokenizer
        Fitted Keras tokenizer.
    max_len : int
        Length that all sequences were padded to.
    """
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    X, Y = [], []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            X.append(sequence[:i])
            Y.append(sequence[i])

    max_len = max(len(x) for x in X)
    X_padded = pad_sequences(X, maxlen=max_len, padding=padding)
    y = np.array(Y)

    return X_padded, y, tokenizer, max_len


def load_json_dataset(filepath: str) -> list:
    """Load a JSON or JSON-lines dataset from disk.

    Supports both a single top-level JSON array and newline-delimited JSON
    (one record per line).

    Parameters
    ----------
    filepath:
        Path to the ``.json`` file.

    Returns
    -------
    list
        A list of records (dicts).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try standard JSON first (single array / object)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        pass

    # Fall back to newline-delimited JSON
    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
