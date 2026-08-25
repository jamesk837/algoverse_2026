#!/usr/bin/env bash
# EC2 environment for the videophy2_auto judge. 
set -euo pipefail

VENV="${VENV:-$HOME/venvs/videophy2}"

PYBIN="${PYTHON:-python3}"
command -v "$PYBIN" >/dev/null || { echo "no such interpreter: $PYBIN"; exit 1; }
# Fail here, not five minutes into a Rust build. These stacks pin packages whose
# newest wheels are 3.12: numpy<2 (vila) stops at 3.12, and transformers 4.28.1's
# tokenizers needs PyO3, which caps at 3.13. torch sets the 3.10 floor.
"$PYBIN" - <<'VERCHECK' || exit 1
import sys
lo, hi = (3, 10), (3, 12)
v = sys.version_info[:2]
if not (lo <= v <= hi):
    print(f"  FAIL  {sys.executable} is Python {v[0]}.{v[1]}; these pins need "
          f"{lo[0]}.{lo[1]}-{hi[0]}.{hi[1]}.")
    print("        Install one and re-run with:  PYTHON=python3.12 $0")
    raise SystemExit(1)
print(f"  using Python {v[0]}.{v[1]} at {sys.executable}")
VERCHECK

if [ -z "${SKIP_APT:-}" ]; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-dev build-essential git \
    ffmpeg libgl1 libglib2.0-0
fi

[ -d "$VENV" ] || "$PYBIN" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

"$PY" -m pip install -q torch
# --no-deps: the tokenizers<0.14 pin has no wheel for modern Python
"$PY" -m pip install -q --no-deps transformers==4.28.1
# --no-deps skipped transformers' own requirements, so they go in by hand. Colab
# had all of these preinstalled, which is why this was never needed there and a
# clean venv dies on `import transformers` with ModuleNotFoundError: packaging.
# Two are deliberately unpinned against what 4.28.1 asks for:
#   tokenizers  -- 4.28.1 wants <0.14, which has no wheel for modern Python and
#     falls back to a Rust build that fails (PyO3 caps at 3.13). A current
#     tokenizers only has to exist to satisfy the import-time version check:
#     videophy2 uses the sentencepiece-based LlamaTokenizer, never the fast one.
#   huggingface-hub -- capped below 1.0 because transformers 4.28 imports
#     HfFolder and friends at module scope, and 1.0 removed them. Nothing here
#     touches the Hub at runtime (the model is a local path), but the import
#     still has to resolve.
# The version-table patch below is what stops the relaxed pins being rejected.
"$PY" -m pip install -q filelock "huggingface-hub<1.0" packaging pyyaml regex   requests tqdm tokenizers
# mplug_owl_video's own imports, which are not transformers' dependencies:
# modeling_mplug_owl.py does `import einops` at module scope, and
# processing_mplug_owl.py does `from PIL import Image`. Its flash_attn import
# is wrapped in a try/except -- the "install flash-attn first." line it prints
# is expected and harmless.
"$PY" -m pip install -q einops pillow
# the judge loads with device_map={"": "cpu"}, which transformers routes
# through accelerate. Held below 1.0: transformers 4.28 imports helpers from
# accelerate.utils that the 1.x line moved, and --no-deps means nothing pins it.
"$PY" -m pip install -q 'accelerate<1.0'
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
check("accelerate", lambda: importlib.import_module("accelerate").__version__)
check("einops", lambda: importlib.import_module("einops").__version__)
check("PIL", lambda: importlib.import_module("PIL").__version__)
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
