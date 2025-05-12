from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline,
)

from app.schemas import ConversationMessage
from labeling_pipeline.rules import FACETS

_MODEL_DIR = Path(os.getenv("ROBERTA_MODEL_DIR", "models/roberta_multitask"))
_DEVICE = 0 if torch.cuda.is_available() else -1


class _ZeroShot:
    def __init__(self) -> None:
        self.pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=_DEVICE,
        )
        self.labels = list(FACETS.keys())

    def predict(
        self, messages: List[ConversationMessage]
    ) -> Tuple[Dict[str, float], list[dict]]:
        joined = "\n".join(m.text for m in messages)[:4096]
        res = self.pipe(joined, self.labels, multi_label=True)
        logits = {lbl: float(scr) for lbl, scr in zip(res["labels"], res["scores"])}
        return logits, []


if _MODEL_DIR.exists():
    _tok = AutoTokenizer.from_pretrained(_MODEL_DIR)
    _qa_model = AutoModelForQuestionAnswering.from_pretrained(_MODEL_DIR)
    _qa_model.eval().to(_DEVICE if _DEVICE >= 0 else "cpu")
    _fallback = None
else:
    _tok = _qa_model = None
    _fallback = _ZeroShot()


@torch.inference_mode()
def predict(messages: List[ConversationMessage]) -> Tuple[Dict[str, float], list[dict]]:
    if _fallback:
        return _fallback.predict(messages)

    context = "\n".join(m.text for m in messages)[:5120]
    logits: Dict[str, float] = {}
    spans: list[dict] = []

    for facet in FACETS.keys():
        query = f"What evidence shows the user's {facet.lower()}?"
        enc = _tok(query, context, truncation=True, return_tensors="pt").to(
            _qa_model.device
        )

        out = _qa_model(**enc)
        start_idx = int(out.start_logits.argmax())
        end_idx = int(out.end_logits.argmax())
        span_ids = enc["input_ids"][0][start_idx : end_idx + 1]
        span_text = _tok.decode(span_ids, skip_special_tokens=True).strip()
        conf = torch.softmax(out.start_logits + out.end_logits, dim=-1).max().item()

        logits[facet] = conf
        spans.append({"facet": facet, "text": span_text})

    return logits, spans
