#!/bin/bash
# Download every model and dataset the examples use.
#
#   setup/prefetch.sh [dest_dir]        (default: $HPCW_SHARED)
#
# Needs internet, so run it on a login node. Models go to <dest>/hf_hub
# (a Hugging Face hub cache), datasets to <dest>/data. About 26 GB in total.
set -euo pipefail

setup=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set +u
source "$setup/env.sh"
set -u
dest=${1:-$HPCW_SHARED}
mkdir -p "$dest/hf_hub" "$dest/data"
export HF_HUB_CACHE=$dest/hf_hub HF_HUB_OFFLINE=0

models=(
    Qwen/Qwen3-0.6B      # 1.2 GB  chat (03), LoRA fine-tuning (04)
    Qwen/Qwen3-1.7B      # 3.4 GB  chat and vLLM (03)
    Qwen/Qwen3-4B        # 8.0 GB  "will it fit on my GPU?" exercise (03)
    Qwen/Qwen3.5-0.8B    # 1.8 GB  2026 generation, hybrid linear attention (03, 04)
    Qwen/Qwen3.5-2B      # 4.6 GB  (03)
)
for m in "${models[@]}"; do
    echo ">>> $m"
    hf download "$m" --quiet > /dev/null
done

# Only the fp16 weights of SDXL-Turbo (7 GB instead of 40 GB).
echo ">>> stabilityai/sdxl-turbo (fp16)"
hf download stabilityai/sdxl-turbo --quiet --include "model_index.json" --include "*/config.json" \
    --include "scheduler/*" --include "tokenizer*/*" --include "*/*.fp16.safetensors" > /dev/null

echo ">>> b-mc2/sql-create-context"
hf download --repo-type dataset b-mc2/sql-create-context --quiet > /dev/null

echo ">>> Tiny Shakespeare"
curl -fsSL -o "$dest/data/shakespeare.txt" \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# CIFAR-10: 60k PNGs in parquet files -> one 180 MB uint8 array file, which is
# what you want on a cluster file system (one big file instead of many small ones).
echo ">>> CIFAR-10"
hf download --repo-type dataset uoft-cs/cifar10 --quiet > /dev/null
python - "$dest/data/cifar10.npz" <<'EOF'
import io, sys
import numpy as np, pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import snapshot_download

root = snapshot_download("uoft-cs/cifar10", repo_type="dataset")
arrays = {}
for split in ("train", "test"):
    table = pq.read_table(f"{root}/plain_text/{split}-00000-of-00001.parquet").to_pydict()
    arrays[f"x_{split}"] = np.stack([np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
                                     for img in table["img"]])
    arrays[f"y_{split}"] = np.array(table["label"], dtype=np.int64)
np.savez(sys.argv[1], **arrays)
print({k: v.shape for k, v in arrays.items()})
EOF

du -sh "$dest/hf_hub" "$dest/data"
