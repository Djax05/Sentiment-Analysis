from fastapi import APIRouter, HTTPException
from .inference.predict import postprocess_outputs
from .inference.loader import load_artifacts
from .schemas.request import (
    TextRequest,
    PredictResponse,
    HealthResponse
)

router = APIRouter()


@router.get("/", response_model=HealthResponse)
def health_check():
    model, vocab = load_artifacts()
    if model is None or vocab is None:
        return HTTPException(
            status_code=500,
            detail="Model not loaded"
            )

    else:
        return HealthResponse(
            status="healthy",
            message=f"API is healthy and length of vocab is {len(vocab)}",
            version="1.0.0"
        )


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: TextRequest):
    return postprocess_outputs(req.text)
