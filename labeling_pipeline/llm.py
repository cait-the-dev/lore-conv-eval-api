from __future__ import annotations

import csv, json, os, re, time, pathlib, sys
from typing import Dict, List, Any, Optional

import openai
from tqdm import tqdm
from pydantic import BaseModel, ValidationError, field_validator

from .config import DATA, LLM_CSV, OPENAI_KEY
from .rules import FACETS

_api_key: Optional[str] = OPENAI_KEY or os.getenv("OPENAI_API_KEY")

if not _api_key:
    print(
        "\033[91m‼︎ OpenAI API-key not found — set `OPENAI_KEY` in config "
        "or export OPENAI_API_KEY in your shell.\033[0m",
        file=sys.stderr,
    )
    sys.exit(1)

openai.api_key = _api_key

MODEL = "gpt-4o"
TEMPERATURE = 0.1
MAX_TOKENS = 120
SEED = 42
MAX_CALLS = 1_000

_examples: Dict[str, List[str]] = {
    # Agency & Self-belief
    "Self-Efficacy": ["I can handle stressful days.", "I managed to fix it myself."],
    "Autonomy": ["I decided to live at home.", "I chose a new doctor."],
    "Growth Mindset": ["I’m still learning new skills.", "I can improve over time."],
    "Locus of Control": [
        "It’s up to me to change.",
        "That problem is beyond my control.",
    ],
    # Emotional Resilience
    "Optimism": [
        "I’m hopeful things will work out.",
        "I try to look on the bright side.",
    ],
    "Emotional Regulation": [
        "I took deep breaths to calm myself.",
        "Grounding exercises helped me relax.",
    ],
    "Self-Compassion": [
        "I’m being kind to myself today.",
        "I forgave myself for the mistake.",
    ],
    # Social & Belonging
    "Social Connectedness": [
        "I talked to my granddaughter.",
        "I’ve felt lonely this week.",
    ],
    "Contribution": [
        "I love to volunteer at the shelter.",
        "Helping others makes me happy.",
    ],
    # Purpose & Meaning
    "Values Alignment": [
        "Spending time with family matters to me.",
        "I value honesty above all.",
    ],
    "Purpose": ["My goal is to finish a 10 K.", "I’m aiming to mentor young people."],
    # Safety & Health
    "Physical Safety": [
        "I’m worried about falling at night.",
        "I feel safe with the alarm system.",
    ],
    "Health Agency": [
        "I always take my meds on time.",
        "I scheduled a check-up next week.",
    ],
    # Help Seeking
    "Support Utilization": [
        "I asked my therapist for advice.",
        "I joined a local support group.",
    ],
}
missing = [f for f in FACETS if f not in _examples]
if missing:
    print("⚠️  Provide examples for new facets:", ", ".join(missing), file=sys.stderr)

_glossary = "\n".join(
    f"- **{facet}** → {' | '.join(_examples.get(facet, [])[:2])}" for facet in FACETS
)
SYSTEM = f"""
You are a senior annotation specialist at Lore Health.

Task ──
Given ONE *user* sentence (never StoryBot) pick the SINGLE most-salient
belief facet or `null` if none clearly apply.

Output schema (STRICT JSON on one line):
{{
  "facet": "<FacetName or null>",
  "evidence": "<short evidence span>",
  "confidence": <float 0-1>
}}

Guidelines ──
• Choose from the facet glossary below.
• Quote the minimal evidence span.
• confidence ≥ 0.80 → high certainty.
• If nothing matches → facet=null, evidence="", confidence=0.0

Facet Glossary ──
{_glossary}
""".strip()


class FacetAnswer(BaseModel):
    facet: str | None
    evidence: str
    confidence: float

    @field_validator("facet")
    @classmethod
    def facet_known(cls, v):
        if v is not None and v not in FACETS:
            raise ValueError(f"unknown facet {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def conf_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("confidence outside 0-1")
        return v


_JS_RE = re.compile(r"\{.*\}", re.S)


def _safe_parse(raw: str) -> dict | None:
    try:
        return json.loads(_JS_RE.search(raw).group(0))
    except Exception:
        return None


def _validate_output(raw_json: dict, original: str) -> FacetAnswer | None:
    """pydantic + evidence-must-match check"""
    try:
        ans = FacetAnswer(**raw_json)
    except ValidationError:
        return None
    if ans.evidence and ans.evidence.lower() not in original.lower():
        return None
    return ans


def _heuristic_facet(sentence: str) -> str | None:
    for facet, pat in FACETS.items():
        if pat.search(sentence):
            return facet
    return None


def _chat_call(
    text: str, retries: int = 3, backoff: float = 1.6
) -> Dict[str, Any] | None:
    delay = 0.5
    for attempt in range(1, retries + 1):
        try:
            resp = openai.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                seed=SEED,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text.strip()},
                ],
            )
            return _safe_parse(resp.choices[0].message.content)

        except openai.RateLimitError as e:
            if attempt == retries:
                print("⚠️  Rate-limit hit - giving up.", file=sys.stderr)
                return None
            time.sleep(delay)
            delay *= backoff

        except openai.AuthenticationError as e:
            print(f"\033[91m‼︎ Authentication failed: {e}\033[0m", file=sys.stderr)
            sys.exit(1)

        except Exception as e:
            print(f"⚠️  OpenAI error: {e}", file=sys.stderr)
            return None

    return None


def run(max_calls: int = MAX_CALLS) -> None:
    if not openai.api_key:
        raise RuntimeError(
            "OPENAI_KEY is not set – please export it or add it to `.env`."
        )

    rows: List[Dict[str, Any]] = []
    calls = kept = fallback = dropped = errors = 0

    conversations = json.load(open(DATA / "conversations.json"))

    with tqdm(total=max_calls, desc="LLM calls") as bar:
        for conv in conversations:
            for msg in conv["messages_list"]:
                if msg["ref_user_id"] == 1:
                    continue
                if calls >= max_calls:
                    break

                raw = _chat_call(msg["message"])
                calls += 1

                if raw is None:
                    errors += 1
                    bar.update(1)
                    bar.set_postfix(kept=kept, fb=fallback, drop=dropped, err=errors)
                    continue

                ans = _validate_output(raw, msg["message"])
                if ans:
                    h_fac = _heuristic_facet(msg["message"])
                    if h_fac and h_fac != ans.facet:
                        ans.facet = h_fac
                        ans.evidence = ""
                        ans.confidence = 0.55
                        fallback += 1
                    else:
                        kept += 1

                    if ans.facet:
                        rows.append(
                            {
                                "conversation": msg["message"],
                                "facet": ans.facet,
                                "answer_text": ans.evidence,
                                "confidence": ans.confidence,
                            }
                        )
                else:
                    dropped += 1

                bar.update(1)
                bar.set_postfix(kept=kept, fb=fallback, drop=dropped, err=errors)

            if calls >= max_calls:
                break

    pathlib.Path(LLM_CSV).parent.mkdir(parents=True, exist_ok=True)

    with open(LLM_CSV, "w", newline="") as f:
        fieldnames = ["conversation", "facet", "answer_text", "confidence"]
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"✅ LLM harvesting complete ▸ "
        f"{kept}+{fallback}={kept+fallback} valid / "
        f"{dropped} dropped / {errors} errors "
        f"(total calls {calls})\n"
        f"   → {LLM_CSV}"
    )


if __name__ == "__main__":
    run()
