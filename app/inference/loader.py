import torch
from ml.preprocessing.vocab import load_vocab
from ml.models.models import EmotionsSentimentModel
from app.core.logging_config import get_logger
from config import CHECKPOINT


MODEL_PATH = CHECKPOINT / "best_model.pt"

_model = None
_vocab = None

logger = get_logger(__name__)


def load_artifacts():
    logger.info("Loading Sentiment Model...")
    global _model, _vocab

    if _vocab is None:
        _vocab = load_vocab()

    if _model is None:
        _model = EmotionsSentimentModel(vocab_size=len(_vocab))
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        _model.load_state_dict(state_dict)
        _model.eval()

    logger.info("Sentiment Model Loaded Successfully")
    return _model, _vocab
