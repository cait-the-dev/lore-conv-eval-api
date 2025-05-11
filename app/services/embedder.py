from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.EMBEDDER_MODEL_NAME)


def compute(messages: List[dict]):
    """Return mean pooled embedding for entire conversation."""
    texts = [m["text"] for m in messages]
    model = _load_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.mean(axis=0)
