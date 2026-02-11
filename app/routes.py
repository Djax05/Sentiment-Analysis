from fastapi import APIRouter
from .inference.predict import postprocess_outputs
from .schemas.request import (
    TextRequest,
    PredictResponse,
    HealthResponse
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():

    return HealthResponse(
        status="healthy",
        message="API is healthy",
        version="1.0.0"
    )


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: TextRequest):
    return postprocess_outputs(req.text)
