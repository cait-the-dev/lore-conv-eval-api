# Conversational Evaluation API

Turns raw StoryBot ↔ user dialogues into **structured self-belief vectors** that power Lore’s resilience-building platform.

## ✨ Why this matters
* Extracts identity signals → feeds contextual bandits & recommender
* Provides explainable evidence spans for clinical validation
* Built with privacy & online/offline feature parity

## ⚡ Quick start

```bash
git clone https://github.com/<you>/lore-conv-eval-api.git
cd lore-conv-eval-api
cp .env.example .env        # edit creds
make docker                 # builds & runs via docker-compose
curl -X POST http://localhost:8000/beliefs \
     -H "Content-Type: application/json" \
     -d @data/sample_conversations.json | jq
```

🏗️ Architecture
<!-- add your image -->

🧑‍💻 Local dev
bash
Copy
Edit
make dev            # uvicorn reload + hot-watch
make test           # pytest, coverage, mypy
📜 API contract
See schemas.py – response includes per-facet
confidence, evidence span, and pillar rollups.

📈 Observability
Prometheus metrics at /metrics:

belief_latency_ms

belief_drift_kl

belief_confidence_bucket{facet}

🛡️ Security & compliance
TLS-only, JWT auth, AES-256 at rest

No PII in logs; Right-to-Delete cascade script in scripts/