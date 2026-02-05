from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.inference.loader import load_artifacts

from .config import settings
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n" + "="*10)
    print("App is loading...")
    print("="*10)

    model, vocab = load_artifacts()

    print("App ready to serve requests")

    yield

    print("\n Shutting Down App...")

app = FastAPI(
    title="Sentiment and Emotion Analysis API",
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, tags=["Sentiment and Emotion Analysis"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "API is alive",
        "docs": "/docs",
        "health": "/health"
        }
