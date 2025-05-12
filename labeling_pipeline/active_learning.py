import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import entropy
from .config import DATA, MERGED_CSV, POOL_FILE, QUERY_BATCH

BATCH = 200


def run():
    gold = pd.read_csv(MERGED_CSV)
    pool = pd.read_json(DATA / "conversations.json")
    pool["text"] = pool.messages_list.apply(
        lambda lst: [m["message"] for m in lst if m["ref_user_id"] != 1]
    ).explode("text")
    pool = pool.drop_duplicates("text")
    pool = pool.loc[~pool.text.isin(gold.conversation)]
    pool.to_csv(POOL_FILE, index=False)

    vec = TfidfVectorizer(max_features=20_000)
    X_train = vec.fit_transform(gold.conversation)
    clf = LogisticRegression(max_iter=1000).fit(X_train, gold.facet)

    X_pool = vec.transform(pool.text)
    proba = clf.predict_proba(X_pool)
    ent = entropy(proba.T)

    idx = np.argsort(-ent)[:BATCH]
    batch = pool.iloc[idx]
    batch.to_csv(QUERY_BATCH, index=False)
    print(f"❓ wrote {len(batch)} high-entropy rows → {QUERY_BATCH}")


if __name__ == "__main__":
    run()
