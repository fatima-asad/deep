# Deep Learning NLP Project

A collection of deep learning and natural language processing experiments using Python, TensorFlow, and Keras.

## Directory Structure

```
deep/
├── notebooks/                  # Jupyter notebooks (one topic per file)
│   ├── mnist_classification.ipynb    # MNIST digit classification
│   ├── sarcasm_detection.ipynb       # Sarcasm detection dataset exploration
│   ├── text_generation_rnn.ipynb     # Text generation with SimpleRNN / LSTM
│   ├── nlp_rnn_fundamentals.ipynb    # NLP fundamentals with RNNs
│   ├── poem_generation_gru.ipynb     # Poem generation with GRU
│   └── ocr_pipeline.ipynb            # OCR / image-text extraction
├── src/
│   └── utils.py                # Shared helper functions
├── requirements.txt
└── README.md
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/mnist_classification.ipynb` | MNIST digit classification using a dense neural network |
| `notebooks/sarcasm_detection.ipynb` | Sarcasm detection dataset exploration |
| `notebooks/text_generation_rnn.ipynb` | Text generation using SimpleRNN / LSTM on a quotes dataset |
| `notebooks/nlp_rnn_fundamentals.ipynb` | NLP fundamentals with RNNs |
| `notebooks/poem_generation_gru.ipynb` | **Poem generation** using a GRU-based language model |
| `notebooks/ocr_pipeline.ipynb` | OCR / image text extraction using Tesseract |

## Shared Utilities (`src/utils.py`)

Common helpers reused across notebooks:

| Function | Description |
|----------|-------------|
| `clean_text(text)` | Lowercase and remove punctuation from a string |
| `build_sequences(texts, vocab_size, padding)` | Tokenize texts and build padded n-gram input/output pairs |
| `load_json_dataset(filepath)` | Load a JSON or newline-delimited JSON file into a list of records |

**Example:**
```python
from src.utils import clean_text, build_sequences

lines = ["The quick brown fox", "jumps over the lazy dog"]
lines_clean = [clean_text(l) for l in lines]
X, y, tokenizer, max_len = build_sequences(lines_clean, vocab_size=100)
```

## Getting Started

### Prerequisites

- Python 3.8+
- [Jupyter Notebook](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fatima-asad/deep.git
   cd deep
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

## Highlighted Project: Poem Generation (GRU)

`notebooks/poem_generation_gru.ipynb` trains a GRU-based language model on a poem dataset to generate new poem-like text.

**Model architecture:**
- Embedding layer (vocab size 500, 100 dimensions)
- GRU layer (100 units, ReLU activation)
- Dense softmax output layer

**Example usage:**
```python
seed = "the cut"
generated = generate(gru_model, tokenizer, seed, max_len, n_words=10)
print(generated)
```

## Tech Stack

- **TensorFlow / Keras** – model building and training
- **Pandas / NumPy** – data processing
- **scikit-learn** – utilities and preprocessing
- **OpenCV / Tesseract** – image processing and OCR

## License

This project is open source and available under the [MIT License](LICENSE).
