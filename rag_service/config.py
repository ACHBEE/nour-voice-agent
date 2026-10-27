"""RAG service settings, from the environment (.env.local / .env in the repo root)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://nour:nour_dev@localhost:5433/nour")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
# Weight of the full-text list in RRF (vector list = 1). Chosen by evals/run_evals.py.
RRF_KEYWORD_WEIGHT = float(os.getenv("RRF_KEYWORD_WEIGHT", "1.0"))
# Persistent model cache (fastembed defaults to the OS temp dir, which gets cleaned).
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", str(ROOT / ".cache" / "fastembed"))
RAG_GRPC_ADDR = os.getenv("RAG_GRPC_ADDR", "localhost:50051")  # where clients connect
RAG_GRPC_PORT = int(os.getenv("RAG_GRPC_PORT", "50051"))  # where the server listens
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
