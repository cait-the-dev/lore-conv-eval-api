from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_DIR = DATA / "label_runs"

HEURISTIC_CSV = OUT_DIR / "heuristic.csv"
LLM_CSV = OUT_DIR / "llm.csv"
MERGED_CSV = DATA / "belief_labeled.csv"

POOL_FILE = OUT_DIR / "pool_unlabeled.csv"
QUERY_BATCH = OUT_DIR / "qa_batch.csv"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
