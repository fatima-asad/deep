"""Reusable model-building functions for deep learning NLP tasks."""

import numpy as np
from tensorflow.keras.layers import Dense, Embedding, GRU, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_rnn_model(
    vocab_size: int,
    embed_dim: int = 100,
    rnn_units: int = 100,
    rnn_type: str = "gru",
    activation: str = "relu",
) -> Sequential:
    """Build a simple RNN-based language model for next-word prediction.

    Parameters
    ----------
    vocab_size:
        Size of the vocabulary (number of unique tokens + 1 for the padding
        index).
    embed_dim:
        Dimensionality of the token embedding vectors.
    rnn_units:
        Number of units in the recurrent layer.
    rnn_type:
        Which recurrent layer to use: ``"gru"``, ``"lstm"``, or ``"rnn"``.
    activation:
        Activation function for the recurrent layer.

    Returns
    -------
    Sequential
        Compiled Keras model ready for training.

    Raises
    ------
    ValueError
        If an unrecognised ``rnn_type`` is supplied.

    Examples
    --------
    >>> model = build_rnn_model(vocab_size=501, embed_dim=100, rnn_units=128)
    >>> model.summary()
    """
    rnn_type = rnn_type.lower()
    rnn_classes = {"gru": GRU, "lstm": LSTM, "rnn": SimpleRNN}
    if rnn_type not in rnn_classes:
        raise ValueError(
            f"Unknown rnn_type '{rnn_type}'. Choose from: {list(rnn_classes)}."
        )

    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embed_dim),
            rnn_classes[rnn_type](units=rnn_units, activation=activation),
            Dense(units=vocab_size, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_next_word(model, tokenizer, seed_text: str, max_len: int) -> str:
    """Predict the single most likely next word following ``seed_text``.

    Parameters
    ----------
    model:
        Trained Keras model built with :func:`build_rnn_model`.
    tokenizer:
        Fitted ``tensorflow.keras.preprocessing.text.Tokenizer``.
    seed_text:
        The input text whose next word should be predicted.
    max_len:
        Sequence length the model was trained on; used for padding.

    Returns
    -------
    str
        The predicted next word, or an empty string if the token cannot be
        found in the tokenizer's index.
    """
    sequence = tokenizer.texts_to_sequences([seed_text])
    padded = pad_sequences(sequence, maxlen=max_len, padding="pre")
    predicted_index = np.argmax(model.predict(padded, verbose=0), axis=-1)[0]

    index_word = {v: k for k, v in tokenizer.word_index.items()}
    return index_word.get(predicted_index, "")


def generate_text(
    model,
    tokenizer,
    seed_text: str,
    max_len: int,
    n_words: int = 10,
) -> str:
    """Generate ``n_words`` new words appended to ``seed_text``.

    Parameters
    ----------
    model:
        Trained Keras model built with :func:`build_rnn_model`.
    tokenizer:
        Fitted ``tensorflow.keras.preprocessing.text.Tokenizer``.
    seed_text:
        Starting text for generation.
    max_len:
        Sequence length the model was trained on.
    n_words:
        Number of words to generate.

    Returns
    -------
    str
        ``seed_text`` followed by the generated words.
    """
    result = seed_text
    for _ in range(n_words):
        next_word = predict_next_word(model, tokenizer, result, max_len)
        if not next_word:
            break
        result = f"{result} {next_word}"
    return result
