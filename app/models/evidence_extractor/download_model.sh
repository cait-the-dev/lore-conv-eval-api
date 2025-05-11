set -euo pipefail
MODEL_NAME=${1:-distilroberta-base}
TARGET_DIR="model"
mkdir -p "$TARGET_DIR"

python - <<'PY'
from huggingface_hub import snapshot_download
import os, sys
model = sys.argv[1]
path = snapshot_download(model_name=model, cache_dir="./hf-cache", local_dir="model", local_dir_use_symlinks=False)
print("Downloaded to", path)
PY $MODEL_NAME

optimum-cli export onnx --model "$TARGET_DIR" "$TARGET_DIR"

echo "ONNX model ready under $TARGET_DIR"