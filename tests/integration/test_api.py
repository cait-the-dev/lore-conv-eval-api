import json
import random
import pathlib

from fastapi.testclient import TestClient

from app.main import app
from app.adapters.conversation_adapter import from_dataset

DATA_PATH = pathlib.Path("data/conversations.json")
DATASET = json.load(DATA_PATH.open())
SAMPLES = random.sample(DATASET, k=min(5, len(DATASET)))

client = TestClient(app)


def test_conversation_samples():
    """Ensure /beliefs responds and returns >=1 belief for sample entries."""
    for entry in SAMPLES:
        req_payload = from_dataset(entry).model_dump(mode="json")
        resp = client.post("/beliefs", json=req_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == req_payload["user_id"]
        assert body["conv_id"] == req_payload["conv_id"]
        assert body["beliefs"], "No beliefs extracted for sample conversation"
