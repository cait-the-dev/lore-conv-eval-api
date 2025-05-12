from __future__ import annotations
import argparse, json, os, pathlib, uuid
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from qdrant_client import QdrantClient, models

DATA_FILE = pathlib.Path("data/conversations.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
BATCH = 128


def load_sentences() -> List[Dict[str, Any]]:
    """Return a flat list of message dicts with conv_id attached."""
    records: List[Dict[str, Any]] = []
    convos = json.load(open(DATA_FILE))
    for conv in convos:
        cid = conv["ref_conversation_id"]
        for m in conv["messages_list"]:
            records.append(
                dict(
                    conv_id=cid,
                    speaker=m["screen_name"],
                    text=m["message"],
                )
            )
    return records


def encode_texts(records: List[Dict[str, Any]]) -> List[List[float]]:
    model = SentenceTransformer(MODEL_NAME)
    texts = [r["text"] for r in records]
    return model.encode(texts, normalize_embeddings=True, batch_size=BATCH)


def rebuild_collection(
    records: List[Dict[str, Any]],
    vectors: List[List[float]],
    collection: str,
    mode: str,
    url: str,
    api_key: str | None,
) -> None:
    cli = QdrantClient(url=url, api_key=api_key)

    cli.recreate_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    )

    if mode == "rag":
        payload = [{"text": r["text"]} for r in records]
        ids = list(range(len(records)))

        cli.upload_collection(
            collection_name=collection,
            ids=ids,
            vectors=vectors,
            payload=payload,
            parallel=4,
        )

    elif mode == "context":
        points = []
        for vec, r in zip(vectors, records):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=r,
                )
            )
        cli.upload_collection(
            collection, vectors=None, payload=points, ids=None, parallel=4
        )

    else:
        raise ValueError(f"Unknown mode {mode!r} (choose 'rag' or 'context')")

    print(f"✅  {len(records)} vectors written → collection “{collection}” ({mode})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build Qdrant index from conversations.json"
    )
    ap.add_argument(
        "--collection", required=True, help="Collection name to create/recreate"
    )
    ap.add_argument(
        "--mode",
        choices=["rag", "context"],
        default="rag",
        help="'rag' = minimal payload | 'context' = conv_id / speaker payload",
    )
    ap.add_argument(
        "--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    ap.add_argument("--qdrant-key", default=os.getenv("QDRANT_API_KEY"))
    args = ap.parse_args()

    print("🔄  Loading sentences …")
    recs = load_sentences()

    print(f"🔄  Encoding {len(recs)} sentences with {MODEL_NAME} …")
    vecs = encode_texts(recs)

    print(f"🔄  Rebuilding collection “{args.collection}” ({args.mode}) …")
    rebuild_collection(
        records=recs,
        vectors=vecs,
        collection=args.collection,
        mode=args.mode,
        url=args.qdrant_url,
        api_key=args.qdrant_key,
    )


if __name__ == "__main__":
    main()
