FROM python:3.12-slim

ENV PYHTONNUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY app ./app
COPY ml/checkpoints ./ml/checkpoints
COPY ml/models ./ml/models
COPY ml/preprocessing ./ml/preprocessing
COPY ml/artifacts ./ml/artifacts

RUN uv pip install --system .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m" ,"uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]