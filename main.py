from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.schemas.request import TextRequest
from app.inference.model import load_model, predict
from config import CHECKPOINT


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(CHECKPOINT / "best_model.pt")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "API is alive"}


@app.post("/predict")
def predict_endpoint(request: TextRequest):

    result = predict(app.state.model, request.text)
    return result
