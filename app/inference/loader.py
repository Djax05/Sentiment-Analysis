import torch
from ml.preprocessing.vocab import load_vocab
from ml.models.models import EmotionsSentimentModel
from config import CHECKPOINT


MODEL_PATH = CHECKPOINT / "best_model.pt"

_model = None
_vocab = None


def load_artifacts():
    global _model, _vocab

    if _vocab is None:
        _vocab = load_vocab()

    if _model is None:
        _model = EmotionsSentimentModel(vocab_size=len(_vocab))
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        _model.load_state_dict(state_dict)
        _model.eval()

    return _model, _vocab
