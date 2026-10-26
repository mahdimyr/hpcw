# Software modules every workshop environment is built on.
# Sourced by env.sh and build_env.sh -- usage: source setup/modules.sh [main|vllm]
#
# StdEnv/2023 is the Alliance (Compute Canada) software stack served from CVMFS.
# Python packages are installed from its pre-built "wheelhouse" with
# `pip install --no-index`, so no compiling and no internet access is needed.
# Some Python packages come from modules instead of pip: pyarrow (arrow module,
# needed by `datasets`) and cv2 (opencv module, needed by vLLM).

module -q load StdEnv/2023
module -q load python/3.12 arrow/21.0.0
if [[ ${1:-main} == vllm ]]; then
    module -q load opencv/4.14.0
fi
