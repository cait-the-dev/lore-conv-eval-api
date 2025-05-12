from __future__ import annotations

import joblib
from pathlib import Path
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer

from app.schemas import ConversationMessage

_MODEL_PATH = Path("models/belief_classifier/logreg.joblib")
_SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if _MODEL_PATH.exists():
    _clf = joblib.load(_MODEL_PATH)
else:
    _clf = None


def predict(messages: List[ConversationMessage]) -> Tuple[Dict[str, float], list]:
    if _clf is None:
        return {}, []
    txt = "\n".join(m.text for m in messages)
    emb = _SBERT.encode([txt], normalize_embeddings=True)
    proba = _clf.predict_proba(emb)[0]
    return {lbl: float(p) for lbl, p in zip(_clf.classes_, proba)}, []
