#!/bin/bash
# One-time preparation of the shared workshop area (instructor only).
#
#   setup/instructor_setup.sh [shared_dir]     (default: /scratch/$USER/hpcw-shared)
#
# Builds both Python environments, downloads models and datasets, and makes the
# directory readable by everyone. Takes ~10 minutes; run it on a login node a few
# days before the workshop, then run every example once to warm up and verify.
# If you use a different directory, update HPCW_SHARED in setup/env.sh.
set -euo pipefail

setup=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
shared=${1:-/scratch/$USER/hpcw-shared}
mkdir -p "$shared"

[[ -d $shared/venv ]] || "$setup/build_env.sh" main "$shared/venv"
[[ -d $shared/venv-vllm ]] || "$setup/build_env.sh" vllm "$shared/venv-vllm"
HPCW_SHARED=$shared "$setup/prefetch.sh" "$shared"

# Attendees need to read the shared area and traverse its parent directories.
chmod -R go+rX "$shared"
chmod o+x "$(dirname "$shared")"
echo "Shared area ready: $shared"
