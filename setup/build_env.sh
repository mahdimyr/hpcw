#!/bin/bash
# Build a workshop Python environment from the Alliance wheelhouse.
#
#   setup/build_env.sh [main|vllm] [target_dir]
#
#   main  PyTorch + Hugging Face stack, ~1 min, ~1.5 GB   (default)
#   vllm  vLLM inference engine,        ~3 min, ~6 GB
#
# Runs fine on a login node: wheels come from CVMFS, nothing is compiled.
# Default target: /scratch/$USER/hpcw/venv (or venv-vllm).
set -euo pipefail

kind=${1:-main}
case $kind in
    main) default_target=/scratch/$USER/hpcw/venv ;;
    vllm) default_target=/scratch/$USER/hpcw/venv-vllm ;;
    *) echo "usage: $0 [main|vllm] [target_dir]" >&2; exit 1 ;;
esac
target=${2:-$default_target}
setup=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -e $target ]]; then
    echo "$target already exists; remove it first if you want to rebuild." >&2
    exit 1
fi

set +u  # module scripts reference unset variables
source "$setup/modules.sh" "$kind"
set -u

echo ">>> Creating virtual environment in $target"
virtualenv --no-download "$target"
source "$target/bin/activate"
pip install --no-index --upgrade pip
echo ">>> Installing packages from requirements-$kind.txt"
pip install --no-index -r "$setup/requirements-$kind.txt"

echo ">>> Checking the environment"
python - <<'EOF'
import importlib, torch
print(f"python ok, torch {torch.__version__} (built for CUDA {torch.version.cuda})")
for name in ("transformers", "vllm", "peft", "trl", "diffusers"):
    try:
        print(f"{name:13s} {importlib.import_module(name).__version__}")
    except ImportError:
        pass
EOF
du -sh "$target"

echo
echo "Done. To use this environment instead of the shared one, run (and add to ~/.bashrc):"
if [[ $kind == vllm ]]; then
    echo "    export HPCW_VENV_VLLM=$target"
else
    echo "    export HPCW_VENV=$target"
fi
