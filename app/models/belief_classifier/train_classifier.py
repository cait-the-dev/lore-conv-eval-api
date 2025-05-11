import argparse, joblib, json, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FACETS = [
    "Self-Efficacy",
    "Growth Mindset",
    "Social Worth",
    "Self-Compassion",
    "Coping Ability",
]


def main(data_path: str, output: str):
    df = pd.read_csv(data_path)
    X = df["embedding"].apply(json.loads).tolist()
    y = df[FACETS]
    clf = make_pipeline(
        StandardScaler(), OneVsRestClassifier(LogisticRegression(max_iter=1000))
    )
    clf.fit(X, y)
    joblib.dump(clf, output)
    print("Saved", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="classifier.joblib")
    main(**vars(parser.parse_args()))
