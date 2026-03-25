# deep

A collection of deep learning experiments using TensorFlow/Keras, covering natural language processing, text classification, and optical character recognition (OCR).

## Features

- **Text Processing** – Tokenization, sequence padding, and embedding layers for NLP tasks
- **Recurrent Neural Networks** – SimpleRNN, LSTM, and GRU models for sequence modeling
- **Poem & Sarcasm Detection** – Text classification experiments on poetry and sarcasm datasets
- **OCR Pipeline** – Image-to-text extraction using OpenCV and Tesseract

## Setup

### Prerequisites

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for OCR notebooks)

### Installation

```bash
git clone https://github.com/fatima-asad/deep.git
cd deep
pip install -r requirements.txt
```

### Running Notebooks

```bash
jupyter notebook
```

Open any notebook from the `notebooks/` directory to explore the experiments.

## Project Structure

```
deep/
├── notebooks/          # Jupyter notebooks for experiments
│   ├── text_rnn.ipynb       # RNN model for text sequences
│   ├── poem_analysis.ipynb  # Poem processing and modeling
│   ├── sarcasm_detection.ipynb  # Sarcasm classification
│   └── ocr_pipeline.ipynb   # OCR with OpenCV + Tesseract
├── src/                # Reusable utility modules
│   └── utils.py        # Common helper functions
├── requirements.txt    # Python dependencies
└── README.md
```

## Dependencies

| Package | Purpose |
|---------|---------|
| TensorFlow / Keras | Model building and training |
| NumPy / Pandas | Data manipulation |
| Matplotlib | Visualization |
| OpenCV | Image processing |
| Pytesseract | OCR text extraction |
| scikit-learn | Data preprocessing utilities |