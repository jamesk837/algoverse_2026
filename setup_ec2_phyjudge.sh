#!/usr/bin/env bash
# EC2 environment for the phyjudge_9b judge. 

VENV="${VENV:-$HOME/venvs/phyjudge}"

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
"$PY" -m pip install -q transformers accelerate peft pyyaml boto3 pandas
# separate requirements on purpose: the qwen-vl-utils[decord] extra fails to
# resolve and silently takes the whole package with it
"$PY" -m pip install -q qwen-vl-utils
"$PY" -m pip install -q decord
# nothing here quantizes, and transformers rejects torchao<0.16 at import
"$PY" -m pip uninstall -y -q torchao 2>/dev/null || true

"$PY" - <<'VERIFY'
import importlib, importlib.util, inspect, os, shutil, sys
ok = True
warn = []

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

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1e9
    print(f"  ----  gpu: {p.name}, {gb:.1f} GB")
    if gb < 35:
        ok = False
        print(f"  FAIL  {gb:.1f} GB is too small. Below ~35 GB accelerate offloads and")
        print("        the PEFT adapter copy into meta-device layers silently no-ops,")
        print("        so you score bare Qwen3.5. Use p4d/p4de/p5.")
    elif "A100" not in p.name and "H100" not in p.name:
        warn.append(f"gpu is {p.name}, not an A100/H100 -- watch the load log for "
                    "'Some parameters are on the meta device'")
else:
    ok = False
    print("  FAIL  no CUDA device")

free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
if free_gb < 60:
    ok = False
    print(f"  FAIL  {free_gb:.0f} GB free. The Qwen3.5-9B base pull is ~19.3 GB plus "
          "the HF cache; size the EBS volume up.")
else:
    print(f"  PASS  disk: {free_gb:.0f} GB free")

if importlib.util.find_spec("torchao") is None:
    print("  PASS  torchao: absent")
else:
    ok = False
    print("  FAIL  torchao is importable; transformers will reject it at import")

check("transformers", lambda: importlib.import_module("transformers").__version__)
check("peft", lambda: importlib.import_module("peft").__version__)
check("decord", lambda: getattr(importlib.import_module("decord"), "__version__", "imported"))
check("yaml", lambda: importlib.import_module("yaml").__version__)
check("boto3", lambda: importlib.import_module("boto3").__version__)
check("pandas", lambda: importlib.import_module("pandas").__version__)

if check("qwen_vl_utils", lambda: "imported"
         if importlib.import_module("qwen_vl_utils").process_vision_info else ""):
    from qwen_vl_utils import process_vision_info
    if "return_video_metadata" in inspect.signature(process_vision_info).parameters:
        print("  PASS  process_vision_info supports return_video_metadata (real fps)")
    else:
        warn.append("process_vision_info has no return_video_metadata; the harness "
                    "falls back to infer.prepare_inputs and Qwen3-VL assumes fps=24 "
                    "on a compressed time axis. Do not run shuffle/reverse/freeze here.")

if os.environ.get("HF_TOKEN"):
    print("  PASS  HF_TOKEN set")
else:
    warn.append("HF_TOKEN unset; phyjudge pulls its base model from the Hub, not S3")

try:
    import boto3
    ident = boto3.client("sts", region_name="us-east-1").get_caller_identity()
    print(f"  PASS  aws credentials: {ident['Arn']}")
except Exception as e:
    ok = False
    print(f"  FAIL  aws credentials: {type(e).__name__}: {e}")

print("-" * 70)
for w in warn:
    print(f"  WARN  {w}")
print("READY" if ok else "FAILED -- fix the lines above")
sys.exit(0 if ok else 1)
VERIFY

cat <<EOF

activate with:  source $VENV/bin/activate
one model load per process -- exit and rerun rather than reloading in place.
EOF
