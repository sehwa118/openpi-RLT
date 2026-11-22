#!/usr/bin/env bash
# Keep the existing JAX/training environment unchanged. Run from any directory.
set -euo pipefail
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate sehwa
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [ ! -x .venv/bin/python ]; then
    echo 'The existing project .venv must be available first.' >&2
    exit 1
fi
if [ ! -x .venv-lingbot-touch/bin/python ]; then
    uv venv --python .venv/bin/python .venv-lingbot-touch
fi
uv pip install --python .venv-lingbot-touch/bin/python 'torch==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
# Reuse installed non-Torch project dependencies; isolated torch site-packages comes first.
PROJECT_SITE="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
LINGBOT_SITE="$(.venv-lingbot-touch/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "$PROJECT_SITE" > "$LINGBOT_SITE/openpi_project_dependencies.pth"
CUDA_VISIBLE_DEVICES="" .venv-lingbot-touch/bin/python -c 'import torch, numpy; print("Extraction environment:", torch.__version__, "NumPy", numpy.__version__); assert torch.__version__ == "2.7.1+cu128"'
