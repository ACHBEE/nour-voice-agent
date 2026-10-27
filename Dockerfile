# One image for ingestion, the RAG gRPC service and the voice agent: same code, different
# commands (see docker-compose.yml). Models are baked in so containers start without network
# access to model hubs, and nothing is downloaded on the first call.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

RUN useradd --create-home --uid 1000 app && mkdir /app && chown app:app /app
WORKDIR /app

COPY requirements.txt .
# BuildKit cache mount: downloaded wheels survive requirement changes, but stay out of the image.
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

USER app

# Voice models (VAD, turn detector, noise cancellation): depend only on installed packages.
RUN python -m livekit.agents download-files

# Embedding model (multilingual-e5-small ONNX): depends only on these three files.
COPY --chown=app:app rag_service/__init__.py rag_service/config.py rag_service/embeddings.py rag_service/
RUN python -c "from rag_service.embeddings import get_embedder; get_embedder()"

COPY --chown=app:app . .
ENV HF_HUB_OFFLINE=1

EXPOSE 50051
CMD ["python", "-m", "rag_service.server"]
