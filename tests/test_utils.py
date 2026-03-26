"""Unit tests for src/utils.py helper functions."""

import json
import os
import tempfile

import numpy as np
import pytest

from src.utils import build_sequences, clean_text, load_json_dataset


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_lowercases_text(self):
        assert clean_text("Hello World") == "hello world"

    def test_removes_punctuation(self):
        result = clean_text("Hello, World!")
        assert "," not in result
        assert "!" not in result

    def test_replaces_punctuation_with_space(self):
        # Punctuation characters become spaces, so words stay separated
        result = clean_text("one,two")
        assert "one" in result
        assert "two" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_already_clean(self):
        assert clean_text("hello world") == "hello world"

    def test_all_punctuation(self):
        result = clean_text("!@#$%")
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# build_sequences
# ---------------------------------------------------------------------------

class TestBuildSequences:
    TEXTS = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "the fox and the dog",
    ]

    def test_returns_four_values(self):
        result = build_sequences(self.TEXTS, vocab_size=50)
        assert len(result) == 4

    def test_X_is_2d_array(self):
        X, y, _, _ = build_sequences(self.TEXTS, vocab_size=50)
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2

    def test_y_is_1d_array(self):
        X, y, _, _ = build_sequences(self.TEXTS, vocab_size=50)
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

    def test_X_and_y_same_length(self):
        X, y, _, _ = build_sequences(self.TEXTS, vocab_size=50)
        assert len(X) == len(y)

    def test_max_len_matches_X_columns(self):
        X, y, _, max_len = build_sequences(self.TEXTS, vocab_size=50)
        assert X.shape[1] == max_len

    def test_tokenizer_fits_vocab(self):
        _, _, tokenizer, _ = build_sequences(self.TEXTS, vocab_size=50)
        assert len(tokenizer.word_index) > 0

    def test_pre_padding(self):
        X, _, _, _ = build_sequences(self.TEXTS, vocab_size=50, padding="pre")
        # Pre-padding: first element of shortest sequence rows should be 0
        assert X[0][0] == 0

    def test_post_padding(self):
        X, _, _, max_len = build_sequences(self.TEXTS, vocab_size=50, padding="post")
        # Post-padding: last element of a row that is shorter than max_len must be 0
        # At least one row must be padded (since not all sequences are max length)
        has_trailing_zero = any(row[-1] == 0 for row in X)
        assert has_trailing_zero


# ---------------------------------------------------------------------------
# load_json_dataset
# ---------------------------------------------------------------------------

class TestLoadJsonDataset:
    def _write(self, content: str) -> str:
        """Write content to a temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def test_loads_json_array(self):
        data = [{"a": 1}, {"b": 2}]
        path = self._write(json.dumps(data))
        try:
            result = load_json_dataset(path)
            assert result == data
        finally:
            os.unlink(path)

    def test_loads_json_lines(self):
        lines = [{"x": 1}, {"x": 2}, {"x": 3}]
        content = "\n".join(json.dumps(r) for r in lines)
        path = self._write(content)
        try:
            result = load_json_dataset(path)
            assert result == lines
        finally:
            os.unlink(path)

    def test_single_object_wrapped_in_list(self):
        obj = {"key": "value"}
        path = self._write(json.dumps(obj))
        try:
            result = load_json_dataset(path)
            assert isinstance(result, list)
            assert result[0] == obj
        finally:
            os.unlink(path)

    def test_returns_list(self):
        path = self._write(json.dumps([]))
        try:
            result = load_json_dataset(path)
            assert isinstance(result, list)
        finally:
            os.unlink(path)
