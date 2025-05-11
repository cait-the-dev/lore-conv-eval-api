PurposeStores the fine‑tuned QA model (DistilRoBERTa) converted to ONNX for faster CPU inference.

We don’t ship 400 MB of weights in the main repo; instead, run download_model.sh to pull and convert.

Usage

cd models/evidence_extractor
./download_model.sh

The script will:

huggingface-cli download distilroberta-base (or your fine‑tuned variant).

Convert to ONNX using optimum-cli export onnx.

Place the result in model/ subdir, which is Git‑LFS tracked.