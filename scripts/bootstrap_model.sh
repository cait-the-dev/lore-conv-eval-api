set -euo pipefail

python - <<'PY'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("Embedding model cached at", model.cache_folder)
PY

if [ ! -f models/evidence_extractor/model/model.onnx ]; then
  echo "[bootstrap] Converting evidence extractor to ONNX..."
  cd models/evidence_extractor
  ./download_model.sh distilroberta-base
  cd -
else
  echo "[bootstrap] ONNX model already present, skipping."
fi