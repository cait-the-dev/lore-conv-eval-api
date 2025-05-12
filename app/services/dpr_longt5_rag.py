from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.schemas import ConversationMessage
from app.services import vector_store
from app.services.roberta_multitask import predict as _roberta_predict

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_tok = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
_gen = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-tglobal-base").to(_DEVICE)


def _json_from(text: str) -> Dict:
    """Extract first JSON object from `text`; return {{}} on failure."""
    try:
        return json.loads(re.search(r"\{.*\}", text, flags=re.S).group(0))
    except Exception:
        return {}


def predict(
    messages: List[ConversationMessage],
) -> Tuple[Dict[str, float], list[dict]]:
    """Return (facet→confidence, evidence‑spans) using RAG‑Lite.

    Falls back to RoBERTa predictor if vector store unavailable *or* retrieval
    yields zero context.
    """
    latest = messages[-1].text
    retrieved: List[str] = vector_store.search(latest, k=5)

    if not retrieved:
        return _roberta_predict(messages)

    prompt = (
        "<belief_extraction>\nCONTEXT:\n"
        + "\n".join(retrieved + [latest])
        + "\n</belief_extraction>"
    )

    inputs = _tok(prompt, return_tensors="pt").to(_DEVICE)
    out_ids = _gen.generate(**inputs, max_length=128)
    decoded = _tok.decode(out_ids[0], skip_special_tokens=True)

    jd = _json_from(decoded)
    logits = {b["facet"]: b.get("confidence", 0.5) for b in jd.get("beliefs", [])}
    spans = [
        {"facet": b["facet"], "text": b["evidence"]} for b in jd.get("beliefs", [])
    ]
    return logits, spans
