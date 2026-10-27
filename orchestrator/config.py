"""Orchestrator settings (env-overridable). Database / gRPC settings live in rag_service.config."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
HISTORY_TURNS = 8  # messages of conversation history the LLM sees per call
