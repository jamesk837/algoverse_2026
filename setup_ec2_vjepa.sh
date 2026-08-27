#!/usr/bin/env bash
# EC2 environment for the frozen V-JEPA 2.1 embedding job (embed_vjepa.py).
#
# Independent of the three judge venvs -- nothing here shares a pin with them,
# so it coexists on the same box. SKIP_APT=1 if you lack sudo.
# SKIP_MODEL_CHECK=1 skips the real forward pass (which pulls ~1.2 GB).

VENV="${VENV:-$HOME/venvs/vjepa}"

if [ -z "${SKIP_APT:-}" ]; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-dev build-essential git \
    ffmpeg libgl1 libglib2.0-0
fi

[ -d "$VENV" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

# torch into the venv, not inherited, so the AMI's torch cannot leak in.
# timm and einops are what vjepa2's hubconf declares.
"$PY" -m pip install -q torch torchvision
"$PY" -m pip install -q timm einops
"$PY" -m pip install -q opencv-python-headless numpy boto3 pandas

"$PY" - <<'VERIFY'
import importlib, os, shutil, sys, time
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
    print(f"  ----  gpu: {p.name}, {p.total_memory/1e9:.1f} GB")
else:
    ok = False
    print("  FAIL  no CUDA device; embed_vjepa.py needs one")

free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
if free_gb < 20:
    ok = False
    print(f"  FAIL  {free_gb:.0f} GB free; the hub checkpoint plus the torch.hub "
          "cache needs headroom")
else:
    print(f"  PASS  disk: {free_gb:.0f} GB free")

for mod in ("timm", "einops", "cv2", "numpy", "boto3", "pandas"):
    check(mod, lambda m=mod: getattr(importlib.import_module(m), "__version__", "imported"))

try:
    import boto3
    ident = boto3.client("sts", region_name="us-east-1").get_caller_identity()
    print(f"  PASS  aws credentials: {ident['Arn']}")
except Exception as e:
    ok = False
    print(f"  FAIL  aws credentials: {type(e).__name__}: {e}")

# The real check: the source does not tell you whether the preprocessor hands
# back (C,T,H,W) or a list of views, nor whether 18,432 tokens fit. One forward
# pass answers both and gives the per-clip cost that decides the instance type.
if os.environ.get("SKIP_MODEL_CHECK"):
    warn.append("SKIP_MODEL_CHECK set; the processor contract and VRAM headroom "
                "are unverified")
elif torch.cuda.is_available():
    try:
        import numpy as np
        processor = torch.hub.load("facebookresearch/vjepa2",
                                   "vjepa2_preprocessor",
                                   crop_size=384, trust_repo=True)
        encoder, _ = torch.hub.load("facebookresearch/vjepa2",
                                    "vjepa2_1_vit_large_384", trust_repo=True)
        encoder = encoder.cuda().eval()
        for q in encoder.parameters():
            q.requires_grad = False

        buf = np.random.randint(0, 255, (64, 480, 640, 3), dtype=np.uint8)
        clip = processor(buf)
        if isinstance(clip, (list, tuple)):
            clip = clip[0]
            warn.append("preprocessor returned a list of views; embed_vjepa takes "
                        "view 0, confirm that is what you want")
        if clip.shape[0] != 3 and clip.shape[1] == 3:
            clip = clip.permute(1, 0, 2, 3)
            warn.append("preprocessor returned (T,C,H,W); embed_vjepa permutes it")
        print(f"  PASS  preprocessor -> {tuple(clip.shape)} {clip.dtype}")

        torch.cuda.reset_peak_memory_stats()
        x = clip.unsqueeze(0).cuda()
        t0 = time.time()
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = encoder(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9

        vec = tokens.float().mean(dim=1).squeeze(0).cpu().numpy().astype("float16")
        if vec.shape != (1024,):
            ok = False
            print(f"  FAIL  pooled to {vec.shape}, expected (1024,)")
        else:
            print(f"  PASS  encoder -> {tuple(tokens.shape)} -> pooled (1024,) fp16")
            print(f"  ----  {dt:.2f}s and {peak:.1f} GB peak for one clip")
            print(f"  ----  ~{dt*10560/3600:.1f} gpu-hours for all 10.5k embeddings "
                  "(decode and s3 not included)")
    except torch.cuda.OutOfMemoryError as e:
        ok = False
        print(f"  FAIL  OOM on a single 64-frame 384px clip: {e}")
        print("        18,432 tokens through a ViT-L does not fit here. Size up.")
    except Exception as e:
        ok = False
        print(f"  FAIL  model check: {type(e).__name__}: {e}")

print("-" * 70)
for w in warn:
    print(f"  WARN  {w}")
print("READY" if ok else "FAILED -- fix the lines above")
sys.exit(0 if ok else 1)
VERIFY

cat <<EOF

activate with:  source $VENV/bin/activate

  python build_split.py --dry-run     # check the join and the PC balance first
  python build_split.py               # upload splits/videophy2_train/split_v1.json
  python embed_vjepa.py --limit 10    # smoke test before the full run
  python embed_vjepa.py               # resumable; rerun after any failure
  python embed_vjepa.py --consolidate  # pack into one npz per split
EOF
