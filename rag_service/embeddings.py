"""Multilingual sentence embeddings on CPU via fastembed (ONNX Runtime, no PyTorch).

Default model: intfloat/multilingual-e5-small (384-d, MIT), loaded from the official repo's
ONNX export. e5 models are trained with asymmetric prefixes, so queries get "query: " and
documents get "passage: "; skipping them measurably hurts retrieval.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from rag_service.config import EMBED_CACHE_DIR, EMBED_MODEL

logger = logging.getLogger("rag.embeddings")

# Models fastembed doesn't ship but whose HF repos include an ONNX export.
_CUSTOM_MODELS = {
    "intfloat/multilingual-e5-small": {
        "pooling": PoolingType.MEAN,
        "normalization": True,
        "sources": ModelSource(hf="intfloat/multilingual-e5-small"),
        "dim": 384,
        "model_file": "onnx/model.onnx",
        "license": "mit",
    },
}
_E5_PREFIXES = ("query: ", "passage: ")


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        supported = {m["model"] for m in TextEmbedding.list_supported_models()}
        if model_name not in supported and model_name in _CUSTOM_MODELS:
            TextEmbedding.add_custom_model(model=model_name, **_CUSTOM_MODELS[model_name])
        self._model = TextEmbedding(model_name=model_name, cache_dir=EMBED_CACHE_DIR)
        self._prefixes = _E5_PREFIXES if "e5" in model_name.lower() else ("", "")
        self.dim = len(self.embed_query("dim probe"))
        logger.info("loaded embedding model %s (dim=%d)", model_name, self.dim)

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([self._prefixes[0] + text]))).tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed([self._prefixes[1] + t for t in texts])]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
