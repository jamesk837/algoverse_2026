"""Verify the predictor error cache, without touching a live run.

Read-only S3 GETs + CPU-only video decode -- no GPU, no torch, safe to run
alongside a live predict_vjepa.py process (different tmp dir, no writes).

Needs embed_vjepa.py and predict_vjepa.py in the same directory (or on
sys.path) and AWS creds in the environment.

    python verify_cache.py --scan          # structural check of every object
    python verify_cache.py --decode 5      # old vs new read_clip on 5 random clips
    python verify_cache.py --decode-worst  # the exact clip that caused the OOM
    python verify_cache.py                 # both
"""
import argparse, io, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import embed_vjepa as E
import predict_vjepa as PV

BUCKET = E.BUCKET
TMP = Path("./tmp_verify")


def old_read_clip(path):
    """The pre-fix implementation: decode the WHOLE clip into RAM, verbatim,
    for comparison only. Never call this on a clip you don't already know is
    small -- it is the thing that OOM'd the box."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def scan_cache():
    """Every cached object: does it parse, does it have err_ arrays, are the
    values finite. An OOM SIGKILL cannot produce a partial object -- the npz
    buffer is built fully in memory before put_object ever runs -- but this
    checks it rather than asserts it."""
    keys = E.list_keys(PV.ERR_PREFIX + "/")
    print(f"{len(keys)} objects under {PV.ERR_PREFIX}/")

    def check(key):
        try:
            body = E._ensure_s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
            with np.load(io.BytesIO(body), allow_pickle=False) as z:
                errs = [k for k in z.files if k.startswith("err_")]
                if not errs:
                    return key, "no err_ arrays"
                for ek in errs:
                    a = z[ek]
                    if a.size == 0:
                        return key, f"{ek} is empty"
                    if not np.all(np.isfinite(a)):
                        return key, f"{ek} has NaN/Inf"
                return key, None
        except Exception as e:
            return key, f"{type(e).__name__}: {e}"

    bad = []
    with ThreadPoolExecutor(max_workers=E.CACHE_READ_WORKERS) as pool:
        for key, why in pool.map(check, keys):
            if why:
                bad.append((key, why))

    print(f"{len(bad)} bad of {len(keys)}")
    for key, why in bad[:30]:
        print(f"  {key}  -  {why}")
    if len(bad) > 30:
        print(f"  ... and {len(bad) - 30} more")
    return bad


def _compare(stem, variant="clean"):
    doc = E.load_split()
    key = E.video_key(doc, stem, variant)
    local = E.download_video(key)
    try:
        old = old_read_clip(local)
        new = E.read_clip(local)
        old_sel = [old[i] for i in E.frame_indices(len(old))]
        same = (len(old_sel) == len(new)
               and all(np.array_equal(a, b) for a, b in zip(old_sel, new)))
        print(f"  {stem[:45]:45s} n_frames={len(old):5d}  "
              f"identical={same}")
        return same
    finally:
        local.unlink(missing_ok=True)


def decode_check(n=5, seed=0):
    doc = E.load_split()
    stems = random.Random(seed).sample(sorted(doc["clips"]), n)
    ok = all(_compare(s) for s in stems)
    print("ALL IDENTICAL" if ok else "MISMATCH FOUND -- do not trust the fix")
    return ok


def decode_worst():
    """The specific clip that OOM-killed the box three times. If old and new
    read_clip agree HERE, they agree everywhere -- this is the largest file
    in the corpus by a 5x margin."""
    keys = E.list_keys("datasets/implausibench/ImplausiBench/real/")
    key = max(keys, key=lambda k: E._ensure_s3().head_object(
        Bucket=BUCKET, Key=k)["ContentLength"])
    stem = Path(key).stem
    print(f"largest real-split clip: {key}")
    local = E.download_video(key)
    try:
        old = old_read_clip(local)
        new = E.read_clip(local)
        old_sel = [old[i] for i in E.frame_indices(len(old))]
        same = (len(old_sel) == len(new)
               and all(np.array_equal(a, b) for a, b in zip(old_sel, new)))
        print(f"  n_frames={len(old)}  identical={same}")
        return same
    finally:
        local.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--decode", type=int, nargs="?", const=5)
    ap.add_argument("--decode-worst", action="store_true")
    a = ap.parse_args()
    TMP.mkdir(exist_ok=True)
    if not (a.scan or a.decode or a.decode_worst):
        a.scan, a.decode = True, 5
    if a.scan:
        scan_cache()
    if a.decode:
        decode_check(a.decode)
    if a.decode_worst:
        decode_worst()
