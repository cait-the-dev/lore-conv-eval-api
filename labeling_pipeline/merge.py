import pandas as pd
from .config import HEURISTIC_CSV, LLM_CSV, MERGED_CSV

THRESH = 0.75


def run():
    h = pd.read_csv(HEURISTIC_CSV)
    l = pd.read_csv(LLM_CSV)
    l = l[l.confidence >= THRESH]

    joined = pd.concat([h, l]).drop_duplicates(subset=["conversation"], keep="first")
    joined.to_csv(MERGED_CSV, index=False)
    print(f"✅ merged {len(joined)} rows → {MERGED_CSV}")


if __name__ == "__main__":
    run()
