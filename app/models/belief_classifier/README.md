## Purpose
Stores the lightweight linear head that maps conversation embeddings to belief facets.We keep the final classifier.joblib under Git‑LFS so the repo clone stays <10 MB.

## Training quick‑start

```
poetry run python train_classifier.py \
  --data ../../data/conversations_labeled.csv \
  --output classifier.joblib
```

The script expects a CSV with columns embedding (JSON list) and one column per facet (1/0). It will:

Parse embeddings into numpy arrays.

Fit a One‑Vs‑Rest LogisticRegression model.

Persist the artifact as classifier.joblib in this folder.

Once committed, Git‑LFS will store it as a pointer, keeping the repo lean.