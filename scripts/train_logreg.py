import joblib, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/belief_labeled.csv")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
X = sbert.encode(df["conversation"].tolist(), normalize_embeddings=True)
y = df["facet"]
clf = LogisticRegression(max_iter=1000).fit(X, y)
clf.classes_ = y.unique()
joblib.dump(clf, "models/belief_classifier/logreg.joblib")
