#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run from an activated, working HP vendor Python environment.
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'Activate the HP CUDA-enabled Python environment first'
assert torch.cuda.is_bf16_supported(), 'BF16 is required'
print('Preserving', torch.__version__, torch.cuda.get_device_name(0))
PY
python -m venv --system-site-packages .venv
.venv/bin/python - <<'PY' > hp-torch-constraints.txt
import importlib.metadata as m
for name in ['torch', 'torchvision', 'torchaudio']:
    try: print(name + '==' + m.version(name))
    except m.PackageNotFoundError: pass
PY
if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -c hp-torch-constraints.txt -r requirements-hp.txt
else
  .venv/bin/python -m pip install -c hp-torch-constraints.txt -r requirements-hp.txt
fi
.venv/bin/python -m ipykernel install --user --name assemlens-hp --display-name 'Python (AssemLens HP)'
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
printf '\nSelect Python (AssemLens HP) in VS Code. FFmpeg is also required.\n'
