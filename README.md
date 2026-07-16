# Sentiment & Emotion Analysis API

A production-ready FastAPI application for **simultaneous sentiment** (positive/negative) **and emotion** (6-class multi-label) analysis on text. Built with PyTorch, featuring custom tokenization, model architecture, and comprehensive training pipeline.

## 🎯 Features

- **Dual-task Learning**: Performs sentiment and emotion classification in a single forward pass
- **Multi-label Emotion Detection**: Classifies text into 6 emotion categories (joy, sadness, anger, fear, surprise, neutral)
- **Binary Sentiment Classification**: Determines if text expresses positive or negative sentiment
- **REST API**: FastAPI-powered endpoints with built-in health checks and CORS support
- **Custom Tokenizer**: Vocabulary-based text encoding with configurable preprocessing
- **Docker Support**: Containerized deployment with health checks
- **Production Logging**: Comprehensive request/response logging and error tracking
- **Configurable Thresholds**: Fine-tuned emotion detection thresholds

## 📊 Project Structure

```
.
├── app/
│   ├── main.py                 # FastAPI application setup
│   ├── routes.py              # API endpoints
│   ├── config.py              # API configuration & constants
│   ├── inference/
│   │   ├── loader.py          # Model & artifact loading
│   │   └── predict.py         # Prediction & postprocessing
│   ├── schemas/
│   │   └── request.py         # Request/response schemas
│   └── core/
│       └── logging_config.py   # Logging setup
├── ml/
│   ├── models/
│   │   ├── models.py          # Model architecture
│   │   ├── encoder.py         # Mean pooling encoder
│   │   ├── train.py           # Training pipeline
│   │   ├── evaluate.py        # Model evaluation
│   │   ├── tokenizer.py       # Text encoding & vocab (training + serving)
│   │   ├── focal_loss.py      # Focal loss implementation
│   │   ├── training_utils.py  # Shared training/validation setup
│   │   ├── validation.py      # Validation logic
│   │   ├── test.py            # Test-split evaluation
│   │   └── tune_threshold.py  # Per-emotion threshold tuning
│   ├── datasets/
│   │   ├── sentiment_dataset.py    # Sentiment data loader
│   │   └── emotion_dataset.py      # Emotion data loader
│   ├── data/
│   │   ├── raw/                # Original datasets
│   │   ├── cleaned/            # Preprocessed GoEmotions data
│   │   └── processed/          # Train/val/test splits
│   ├── checkpoints/           # Model checkpoints
│   │   ├── best_model.pt
│   │   └── last_model.pt
│   ├── artifacts/             # vocab.pkl & thresholds.json
│   └── notebooks/             # EDA & analysis notebooks
├── scripts/
│   ├── data_preprocessing.py       # Text cleaning + GoEmotions filtering
│   ├── build_sentiment_dataset.py  # Rebuilds the Sentiment140 split from raw
│   └── build.py                    # Builds the vocabulary
├── config.py                  # Global configuration
├── utils.py                   # Project utilities
├── Dockerfile                 # Docker image definition
├── pyproject.toml            # Project metadata & dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- PyTorch (CPU or CUDA)
- pip or uv package manager

### Installation

1. **Clone or setup the project**:
```bash
cd "Sentiment Analysis"
```

2. **Create and activate virtual environment**:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # On Windows PowerShell
```

3. **Install dependencies**:
```bash
pip install -e .
# OR using uv (faster):
uv pip install --system .
```

### Running the API

Start the FastAPI development server:
```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

**Interactive API Documentation**: Visit `http://localhost:8000/docs` (Swagger UI)

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Returns API status and version.

**Response**:
```json
{
  "status": "healthy",
  "message": "API is healthy",
  "version": "1.0.0"
}
```

### Sentiment & Emotion Prediction
```bash
POST /predict
Content-Type: application/json

{
  "text": "I am feeling happy and excited!"
}
```

**Response**:
```json
{
  "sentiment": {
    "probability": 0.85,
    "label": "positive"
  },
  "emotions": {
    "joy": 0.92,
    "sadness": 0.05,
    "anger": 0.02,
    "fear": 0.08,
    "surprise": 0.45,
    "neutral": 0.15
  },
  "active_emotions": ["joy", "surprise"]
}
```

## 🤖 Model Architecture

The `EmotionsSentimentModel` is a dual-task neural network:

- **Embedding Layer**: Maps tokens to 128-dimensional embeddings
- **Encoder**: Mean pooling encoder that creates sentence representations
- **Sentiment Head**: Linear layer outputting sentiment probability (sigmoid activation)
- **Emotion Head**: Linear layer outputting 6 emotion scores (sigmoid activation)

### Key Parameters
- Embedding dimension: 128
- Vocabulary size: 30,000
- Max sequence length: 100
- Number of emotions: 6

## 📚 Training

### Data Sources

The model is trained on two datasets:

- **Sentiment task — [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)** (1.6M tweets, Kaggle: `kazanova/sentiment140`; original source: Stanford, Go et al. 2009). The raw `training.1600000.processed.noemoticon.csv` file is preprocessed by `scripts/build_sentiment_dataset.py`: it keeps only the `sentence` and `sentiment` columns, remaps labels `4 → 1` (binary `0` = negative, `1` = positive), and cleans the text (strips `@mentions` and URLs, replaces non-letters with spaces, collapses whitespace, lowercases).
- **Emotion task — GoEmotions** (Google Research), filtered down to 6 labels: joy, sadness, anger, fear, surprise, neutral. Multi-label — a single example can carry more than one emotion. The label distribution is heavily imbalanced: `neutral` has ~14.7k positive examples vs. `fear`'s ~829, which is why the loss function below weights per class.

### Training Approach

`EmotionsSentimentModel` is a single shared encoder with two task heads: a binary sentiment head trained with `BCEWithLogitsLoss`, and a multi-label emotion head trained with a custom focal loss. The focal loss uses a per-class `alpha` tensor (`[1.85, 2.18, 1.86, 4.63, 2.63, 0.26]`, inverse-frequency weights in the emotion order above) instead of a flat weight — this was the single biggest improvement to emotion-detection quality, since it keeps the model paying attention to rare classes like `fear` instead of collapsing toward the majority `neutral` class.

Training uses early stopping (patience of 3 epochs with no improvement in validation macro F1); the checkpoint with the best macro F1 is saved as `best_model.pt`. After training, per-emotion decision thresholds are tuned on the validation set via an F1 sweep (`ml/models/tune_threshold.py`) and saved to `ml/artifacts/thresholds.json`, which is then loaded at both validation and inference time. The full pipeline runs in this order: **train → tune thresholds → validate**.

### Training Pipeline

Run the training script:
```bash
python ml/models/train.py
```

**Hyperparameters** (configurable in `ml/models/training_utils.py`):
- Batch size: Configurable
- Learning rate: Configurable
- Epochs: Configurable
- Loss function: Focal Loss for class imbalance handling

### Data Preprocessing

Preprocess raw data:
```bash
python scripts/data_preprocessing.py
```

**Text cleaning operations**:
- URL removal
- Mention/hashtag removal
- Lowercasing
- Special character removal
- Whitespace normalization

## ⚙️ Configuration

### Emotion Detection Thresholds
Configured in `config.py` (adjustable):
```python
EMOTION_THRESHOLD = {
    "joy": 0.35,
    "sadness": 0.35,
    "anger": 0.35,
    "fear": 0.30,
    "surprise": 0.35,
    "neutral": 0.50,
}
```

### Tokenizer Parameters
```python
TOKENIZER_PARAMETERS = {
    "pad_token": "<PAD>",
    "unk_token": "<UNK>",
    "max_vocab_size": 30000,
    "max_seq_len": 100,
    "lowercase": True
}
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t emotion-reader:latest .
```

### Run Container
```bash
docker run -p 8000:8000 emotion-reader:latest
```

The container includes:
- Health checks every 30 seconds
- CPU-optimized PyTorch installation
- Uvicorn server on port 8000
- All necessary model checkpoints and artifacts

## 📦 Dependencies

**Core Dependencies**:
- `fastapi>=0.124.0` - Web framework
- `uvicorn>=0.40.0` - ASGI server
- `torch` - Deep learning framework (CPU by default)
- `pandas>=2.3.3` - Data manipulation
- `scikit-learn>=1.8.0` - ML utilities
- `pydantic>=2.12.5` - Data validation

**Development Dependencies**:
- `matplotlib>=3.10.8` - Visualization
- `ipykernel>=7.1.0` - Jupyter support

See `pyproject.toml` for complete dependency list.

## 🔧 Development & Testing

### Run Tests
```bash
python tests/test.py
```

### Available Notebooks
- [EDA](ml/notebooks/eda.ipynb) - Exploratory Data Analysis
- [Data Separation](ml/notebooks/data_separation.ipynb) - Train/val/test split logic
- [Label Selection](ml/notebooks/label_selection.ipynb) - Emotion label analysis

## 🎓 Model Evaluation

The model is evaluated on multiple metrics:
- **F1 Score** - Harmonic mean of precision and recall
- **Accuracy** - Overall correctness
- **Precision** - False positive rate
- **Recall** - False negative rate

Evaluation code: `ml/models/evaluate.py`

### Results

On the validation split, the sentiment head reaches **~78% accuracy / ~0.77 F1**, and the emotion head reaches **~0.46 macro F1 / ~0.66 micro F1** — competitive with published BERT baselines on GoEmotions for a from-scratch (non-pretrained) encoder.

## ⚠️ Known Limitations

- **Distribution shift**: the sentiment head was trained on 2009-era tweets (Sentiment140). It performs well on informal, social-media-style phrasing (`"i love this so much best day ever"` → 0.92 positive) but degrades on idiomatic general English (`"i feel so good i could jump for joy"` was misclassified negative in controlled testing).
- **Preprocessing consistency**: `encode_text` expects input cleaned the same way the training data was (lowercase, punctuation stripped). `predict_text()` in `app/inference/predict.py` applies `clean_text()` to the input before encoding, so this is handled automatically at inference time rather than left to the caller.
- **Fixed-length padding**: `encode_text` pads every sequence to 100 tokens. This is consistent between training and inference, but it dilutes the signal for short inputs.
- **Rare-class ceiling**: `fear` has only ~829 training examples; performance on rare emotions is fundamentally data-limited, not just a loss-function problem.

## 📝 Logging & Monitoring

- **Request logging**: All HTTP requests/responses logged with timestamps
- **Model inference logging**: Predictions and processing times tracked
- **Error logging**: Comprehensive error messages for debugging

Configure logging in `app/core/logging_config.py`

## 🌐 CORS Configuration

CORS is enabled for all origins (configurable in `app/main.py`):
```python
allow_origins=["*"]
allow_methods=["*"]
allow_headers=["*"]
```

Modify for production security requirements.

## 📈 Performance Optimization

- **Mean pooling encoder** - Efficient sentence representation
- **Focal loss** - Handles class imbalance in emotion detection
- **Batch inference** - Support for batch processing
- **Model caching** - Artifacts loaded once on startup
- **CPU-optimized** - PyTorch CPU build by default

## 🤝 Contributing

This is an end-to-end sentiment analysis project. Contributions for improvements welcome!

## 📄 License

[Specify your license here]

## ✨ Project Highlights

- ✅ Complete ML pipeline from data preprocessing to API
- ✅ Production-ready FastAPI server
- ✅ Dual-task learning (sentiment + emotion)
- ✅ Docker containerization
- ✅ Comprehensive logging and error handling
- ✅ Configurable thresholds and parameters
- ✅ RESTful API with Swagger documentation

---

**Questions?** Check the notebooks in `ml/notebooks/` for detailed analysis and model development insights.
