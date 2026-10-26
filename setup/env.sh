# Workshop environment -- source this file, don't execute it:
#
#     source setup/env.sh          # PyTorch + Hugging Face environment
#     source setup/env.sh vllm     # vLLM environment (example 03)
#
# It loads the modules, activates a Python virtual environment and points every
# cache (models, compiled kernels, ...) at /scratch instead of your home directory.
# Every variable can be overridden by exporting it before sourcing this file.

_hpcw_kind=${1:-main}
_hpcw_setup=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Read-only area prepared by the instructor: Python environments, models, datasets.
export HPCW_SHARED=${HPCW_SHARED:-/scratch/tus210/hpcw-shared}
# Your own working area: outputs, checkpoints and caches.
export HPCW_WORK=${HPCW_WORK:-/scratch/$USER/hpcw}
export HPCW_DATA=${HPCW_DATA:-$HPCW_SHARED/data}
mkdir -p "$HPCW_WORK"

source "$_hpcw_setup/modules.sh" "$_hpcw_kind"

# Python environment: the shared one unless you built your own with
# setup/build_env.sh and exported HPCW_VENV (or HPCW_VENV_VLLM) to point at it.
if [[ $_hpcw_kind == vllm ]]; then
    _hpcw_venv=${HPCW_VENV_VLLM:-$HPCW_SHARED/venv-vllm}
else
    _hpcw_venv=${HPCW_VENV:-$HPCW_SHARED/venv}
fi
if [[ -f $_hpcw_venv/bin/activate ]]; then
    source "$_hpcw_venv/bin/activate"
else
    echo "env.sh: no Python environment at $_hpcw_venv -- build one with setup/build_env.sh" >&2
fi

# Hugging Face: models were pre-downloaded into the shared hub cache, so jobs run
# offline and 30 people don't download the same files at once. To use a model
# that isn't there, see "Bring your own model" in the README.
export HF_HOME=${HF_HOME:-$HPCW_WORK/hf_home}
if [[ -z ${HF_HUB_CACHE:-} && -d $HPCW_SHARED/hf_hub ]]; then
    export HF_HUB_CACHE=$HPCW_SHARED/hf_hub
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
fi
export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}

# Progress bars are nice in a terminal but flood the log files of batch jobs.
if [[ ! -t 1 ]]; then
    export TQDM_DISABLE=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
fi

# JIT-compiled GPU kernels (torch.compile, Triton, vLLM) are cached on disk so the
# second run starts much faster. Keep these caches out of your home quota.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HPCW_WORK/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HPCW_WORK/cache/inductor}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HPCW_WORK/cache/vllm}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$HPCW_WORK/cache/matplotlib}

unset _hpcw_kind _hpcw_setup _hpcw_venv
