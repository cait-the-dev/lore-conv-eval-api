from pathlib import Path
from typing import List

import joblib
from transformers import pipeline

from app.config import get_settings

_LABELS = [
    "Self‑Efficacy",
    "Growth Mindset",
    "Social Worth",
    "Self‑Compassion",
    "Coping Ability",
]


class _ZeroShot:
    def __init__(self):
        self._clf = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            truncation=True,
        )

    def predict(self, messages: List[dict]):
        text = "\n".join(m["text"] for m in messages)[0:4096]
        result = self._clf(text, _LABELS, multi_label=True)
        return dict(zip(result["labels"], result["scores"]))


def _load_linear_head():
    settings = get_settings()
    path = Path(settings.CLASSIFIER_MODEL_DIR, "classifier.joblib")
    if path.exists():
        return joblib.load(path)
    return None


_linear = _load_linear_head()
_zero = _ZeroShot()


def predict(messages: List[dict]):
    """Return mapping {facet: confidence 0‑1}."""
    if _linear is not None:
        emb = compute_embedding(messages)
        probs = _linear.predict_proba([emb])[0]
        return {label: float(prob) for label, prob in zip(_linear.classes_, probs)}
    return _zero.predict(messages)


from services.embedder import (
    compute as compute_embedding,
)
