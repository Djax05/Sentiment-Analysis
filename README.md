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
│   │   ├── tokenizer.py       # Text encoding
│   │   ├── focal_loss.py      # Focal loss implementation
│   │   ├── training_utils.py  # Training utilities
│   │   ├── validation.py      # Validation logic
│   │   └── test.py            # Testing utilities
│   ├── datasets/
│   │   ├── sentiment_dataset.py    # Sentiment data loader
│   │   └── emotion_dataset.py      # Emotion data loader
│   ├── preprocessing/
│   │   ├── text_encoder.py    # Text encoding functions
│   │   └── vocab.py           # Vocabulary management
│   ├── data/
│   │   ├── raw/               # Original datasets
│   │   ├── cleaned/           # Preprocessed data
│   │   ├── processed/         # Train/val/test splits
│   │   └── appended/          # Combined datasets
│   ├── checkpoints/           # Model checkpoints
│   │   ├── best_model.pt
│   │   └── last_model.pt
│   ├── artifacts/             # Serialized vocab & models
│   └── notebooks/             # EDA & analysis notebooks
├── scripts/
│   ├── data_preprocessing.py   # Data cleaning pipeline
│   └── build.py               # Build utilities
├── tests/
│   └── test.py                # Unit tests
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

### Datasets

The model is trained on two datasets:
- **Sentiment Dataset**: Binary sentiment classification (positive/negative)
  - Training split used for sentiment head
- **GoEmotions Dataset**: 6-class multi-label emotion classification
  - Training split used for emotion head

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
