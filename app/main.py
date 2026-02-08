import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.inference.loader import load_artifacts
from .core.logging_config import setup_logging, get_logger

from .config import settings
from .routes import router


setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, vocab = load_artifacts()

    logger.info("App ready to serve requests")

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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} | "
        f"{response.status_code} | {duration:.2f}ms"
    )

    return response

app.include_router(router, tags=["Sentiment and Emotion Analysis"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "API is alive",
        "docs": "/docs",
        "health": "/health"
        }
