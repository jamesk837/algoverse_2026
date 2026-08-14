#!/usr/bin/env bash
# EC2 environment for the videophy2_auto judge. 
set -euo pipefail

VENV="${VENV:-$HOME/venvs/videophy2}"

if [ -z "${SKIP_APT:-}" ]; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-dev build-essential git \
    ffmpeg libgl1 libglib2.0-0
fi

[ -d "$VENV" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

"$PY" -m pip install -q torch
# --no-deps: the tokenizers<0.14 pin has no wheel for modern Python
"$PY" -m pip install -q --no-deps transformers==4.28.1
"$PY" -m pip install -q decord sentencepiece protobuf boto3 pandas

# relax the runtime version table that --no-deps just bypassed
"$PY" - <<'PATCH'
import importlib.util, re
from pathlib import Path
spec = importlib.util.find_spec("transformers")
if spec is None or not spec.origin:
    raise SystemExit("transformers did not install")
p = Path(spec.origin).parent / "dependency_versions_table.py"
s = p.read_text(encoding="utf-8")
for pkg in ("tokenizers", "huggingface-hub"):
    s = re.sub(rf'^(\s*)"{pkg}":.*$', rf'\1"{pkg}": "{pkg}",', s, flags=re.M)
p.write_text(s, encoding="utf-8")
print(f"patched {p}")
PATCH

# torch>=2.6 flipped weights_only; .bin loading under transformers 4.28 needs it off
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
cat > "$SITE/sitecustomize.py" <<'SHIM'
try:
    import torch
    if not getattr(torch.load, "_weights_only_shim", False):
        _orig = torch.load

        def _load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig(*a, **kw)

        _load._weights_only_shim = True
        torch.load = _load
except Exception:
    pass
SHIM
echo "wrote $SITE/sitecustomize.py"

"$PY" - <<'VERIFY'
import importlib, sys
ok = True

def check(label, fn):
    global ok
    try:
        print(f"  PASS  {label}: {fn()}")
    except Exception as e:
        ok = False
        print(f"  FAIL  {label}: {type(e).__name__}: {e}")

print("-" * 70)
import torch
check("torch", lambda: f"{torch.__version__}, cuda={torch.cuda.is_available()}")
check("gpu", lambda: torch.cuda.get_device_name(0))
check("decord", lambda: getattr(importlib.import_module("decord"), "__version__", "imported"))
check("sentencepiece", lambda: "imported" if importlib.import_module("sentencepiece") else "")
check("boto3", lambda: importlib.import_module("boto3").__version__)
check("pandas", lambda: importlib.import_module("pandas").__version__)

import transformers
if transformers.__version__ == "4.28.1":
    print(f"  PASS  transformers: {transformers.__version__}")
else:
    ok = False
    print(f"  FAIL  transformers is {transformers.__version__}, must be 4.28.1")

if getattr(torch.load, "_weights_only_shim", False):
    print("  PASS  torch.load shim active")
else:
    ok = False
    print("  FAIL  torch.load shim not active; sitecustomize.py did not load")

try:
    import boto3
    ident = boto3.client("sts", region_name="us-east-1").get_caller_identity()
    print(f"  PASS  aws credentials: {ident['Arn']}")
except Exception as e:
    ok = False
    print(f"  FAIL  aws credentials: {type(e).__name__}: {e}")

print("-" * 70)
print("READY" if ok else "FAILED -- fix the lines above")
sys.exit(0 if ok else 1)
VERIFY

cat <<EOF

activate with:  source $VENV/bin/activate
EOF
