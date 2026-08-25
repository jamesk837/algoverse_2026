#!/usr/bin/env bash
# EC2 environment for the vila_ewm (WorldModelBench) judge. 
set -euo pipefail

VENV="${VENV:-$HOME/venvs/vila}"

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
VILA_DIR="${VILA_DIR:-$HOME/VILA}"

if [ -z "${SKIP_APT:-}" ]; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-dev build-essential git \
    ffmpeg libgl1 libglib2.0-0
fi

[ -d "$VILA_DIR" ] || git clone --depth 1 https://github.com/NVlabs/VILA.git "$VILA_DIR"

# VILA hardcodes attn_implementation="flash_attention_2" in its own source
python3 - "$VILA_DIR" <<'REWRITE'
import sys
from pathlib import Path
n = 0
for p in (Path(sys.argv[1]) / "llava").rglob("*.py"):
    s = p.read_text(encoding="utf-8", errors="replace")
    if "flash_attention_2" in s:
        p.write_text(s.replace("flash_attention_2", "sdpa"), encoding="utf-8")
        n += 1
print(f"rewrote flash_attention_2 -> sdpa in {n} file(s)")
REWRITE

[ -d "$VENV" ] || "$PYBIN" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

"$PY" -m pip install -q torch
# --no-deps: VILA's pyproject pins gradio==3.35.2, which drags pydantic<2 and
# makes the resolve impossible. gradio/bitsandbytes/flash-attn are demo-or-kernel
# deps that inference never touches.
"$PY" -m pip install -q -e "$VILA_DIR" --no-deps
"$PY" -m pip install -q transformers==4.46.0 accelerate==0.34.2
"$PY" -m pip install -q boto3 pandas sentencepiece protobuf einops timm \
  opencv-python pillow safetensors loguru hydra-core omegaconf termcolor \
  shortuuid einops-exts pytorchvideo decord openpyxl markdown2 scikit-learn httpx
"$PY" -m pip install -q deepspeed
"$PY" -m pip install -q git+https://github.com/bfshi/scaling_on_scales.git
# last, so nothing above can bump it back up
"$PY" -m pip install -q 'numpy<2'

SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"

# `pip install -e` can fail quietly on this repo
echo "$VILA_DIR" > "$SITE/vila_path.pth"

# llava/train/sequence_parallel/* and llava/model/language_model/q*llama.py carry
# unconditional top-level `from flash_attn import ...`; ps3 is imported the same way.
cat > "$SITE/sitecustomize.py" <<'STUBS'
import importlib.machinery
import sys
import types

_ATTRS = {}


class _RaisingModule(types.ModuleType):
    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        key = (self.__name__, attr)
        if key not in _ATTRS:
            def _raise(*a, **kw):
                raise RuntimeError(
                    f"{key[0]}.{key[1]} was actually CALLED. The stub is only safe "
                    f"if this path is never taken -- the model should be on sdpa.")
            _ATTRS[key] = type(attr, (), {"__init__": _raise, "__call__": _raise})
        return _ATTRS[key]


def _install(root, submodules=()):
    names = set()
    for n in [root] + [f"{root}.{s}" for s in submodules]:
        parts = n.split(".")
        for i in range(1, len(parts) + 1):
            names.add(".".join(parts[:i]))
    for name in sorted(names, key=lambda s: s.count(".")):
        mod = _RaisingModule(name)
        mod.__path__ = []
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = mod
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, mod)


if "flash_attn" not in sys.modules:
    _install("flash_attn", ["bert_padding", "flash_attn_interface", "layers.rotary",
                            "ops.triton.rotary", "modules.mha"])
if "ps3" not in sys.modules:
    _install("ps3")
STUBS
echo "wrote $SITE/sitecustomize.py"

"$PY" - <<'VERIFY'
import importlib, sys
ok = True

def check(label, fn):
    global ok
    try:
        print(f"  PASS  {label}: {fn()}")
        return True
    except Exception as e:
        ok = False
        print(f"  FAIL  {label}: {type(e).__name__}: {e}")
        return False

print("-" * 70)
import torch
check("torch", lambda: f"{torch.__version__}, cuda={torch.cuda.is_available()}")
check("gpu", lambda: torch.cuda.get_device_name(0))
check("transformers", lambda: importlib.import_module("transformers").__version__)
check("boto3", lambda: importlib.import_module("boto3").__version__)
check("pandas", lambda: importlib.import_module("pandas").__version__)
check("s2wrapper", lambda: "imported" if importlib.import_module("s2wrapper") else "")

import numpy as np
if np.__version__.startswith("2."):
    ok = False
    print(f"  FAIL  numpy is {np.__version__}, must be <2")
else:
    print(f"  PASS  numpy: {np.__version__}")

# canary for the numpy ABI mismatch
check("deepspeed", lambda: importlib.import_module("deepspeed").__version__)

if "flash_attn" in sys.modules and "ps3" in sys.modules:
    print("  PASS  flash_attn + ps3 stubs active")
else:
    ok = False
    print("  FAIL  stubs not active; sitecustomize.py did not load")

if check("import llava", lambda: importlib.import_module("llava").__file__):
    from transformers.utils import is_flash_attn_2_available
    if is_flash_attn_2_available():
        ok = False
        print("  FAIL  is_flash_attn_2_available() is True -- the model would route "
              "onto an FA2 path and hit the raising stubs")
    else:
        print("  PASS  is_flash_attn_2_available(): False (model stays on sdpa)")

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
vila_ewm decodes at temperature 0.7 -- every score is a single stochastic draw.
EOF
