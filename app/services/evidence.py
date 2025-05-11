from functools import lru_cache
from typing import List, Dict

from transformers import pipeline


@lru_cache(maxsize=1)
def _qa():
    return pipeline("question-answering", model="distilroberta-base")


def add_spans(beliefs: List[Dict], messages: List[dict]):
    qa = _qa()
    context = " ".join(m["text"] for m in messages)[0:4096]
    for b in beliefs:
        question = f"What in this text shows {b['facet'].lower()}?"
        try:
            ans = qa(question=question, context=context)
            b["evidence"] = ans.get("answer", "")
        except Exception:
            pass
    return beliefs
