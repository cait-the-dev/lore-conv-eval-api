from typing import Dict, List

_THRESHOLD = 0.35
_PILLAR_MAP = {
    "Self‑Efficacy": "Coping",
    "Coping Ability": "Coping",
    "Growth Mindset": "Growth",
    "Social Worth": "Connectedness",
    "Self‑Compassion": "Emotional Regulation",
}


def map_logits(logits: Dict[str, float], messages):
    beliefs: List[Dict] = []
    for facet, score in logits.items():
        if score >= _THRESHOLD:
            beliefs.append(
                {
                    "facet": facet,
                    "belief_id": facet.replace(" ", "_").upper(),
                    "confidence": round(score, 3),
                    "evidence": "",
                }
            )
    return beliefs


def rollup(beliefs: List[Dict]):
    tally: Dict[str, List[float]] = {}
    for b in beliefs:
        pillar = _PILLAR_MAP.get(b["facet"], "Other")
        tally.setdefault(pillar, []).append(b["confidence"])
    return {pill: round(sum(vals) / len(vals), 3) for pill, vals in tally.items()}
