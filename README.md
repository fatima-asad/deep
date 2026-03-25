# Deep Learning NLP Project

A collection of deep learning and natural language processing experiments using Python, TensorFlow, and Keras.

## Projects

| Notebook | Description |
|----------|-------------|
| `Untitled8.ipynb` | MNIST digit classification using a dense neural network |
| `Untitled25.ipynb` | Sarcasm detection dataset exploration |
| `Untitled80.ipynb` | Text generation using SimpleRNN on a quotes dataset |
| `Untitled83.ipynb` | NLP fundamentals with RNNs |
| `Untitled84.ipynb` | **Poem generation** using a GRU-based language model |
| `Untitled86.ipynb` | OCR / image text extraction using Tesseract |

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

`Untitled84.ipynb` trains a GRU-based language model on a poem dataset to generate new poem-like text.

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
