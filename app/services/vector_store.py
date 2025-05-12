from __future__ import annotations
import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
_SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if BACKEND == "qdrant":
    from qdrant_client import QdrantClient

    _qc = QdrantClient(host=os.getenv("QDRANT_HOST", "qdrant"), port=6333)
    _COLL = "belief_ctx"

    def search(query: str, k: int = 5) -> List[str]:
        vec = _SBERT.encode(query, normalize_embeddings=True)
        hits = _qc.search(
            collection_name=_COLL,
            query_vector=vec.tolist(),
            limit=k,
            with_payload=True,
        )
        return [h.payload["text"] for h in hits]

else:
    import pickle, faiss
    from pathlib import Path

    _IDX_DIR = Path("models/rag_index")
    _texts = (
        pickle.loads((_IDX_DIR / "texts.pkl").read_bytes()) if _IDX_DIR.exists() else []
    )
    _index = (
        faiss.read_index((_IDX_DIR / "faiss.index").as_posix())
        if _IDX_DIR.exists()
        else None
    )

    def search(query: str, k: int = 5) -> List[str]:
        if _index is None:
            return []
        vec = _SBERT.encode(query, normalize_embeddings=True)
        _, ids = _index.search(np.asarray([vec]), k)
        return [_texts[i] for i in ids[0] if i != -1]
