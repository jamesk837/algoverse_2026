import argparse
import hashlib
import io
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import cv2
import numpy as np

BUCKET = "nickb-aarj"
SPLIT_KEY = "splits/videophy2_train/split_v1.json"

HUB_REPO = "facebookresearch/vjepa2"
HUB_MODEL = "vjepa2_1_vit_large_384"
HUB_PREPROCESSOR = "vjepa2_preprocessor"
CROP_SIZE = 384
NUM_FRAMES = 64
EMBED_DIM = 1024

# 2-frame tubelet: 64 frames -> 32 moments. 384 crop / 16px patches -> 24x24.
N_TEMPORAL = 32
N_SPATIAL = 576

# vjepa2 main ships a localhost placeholder in src/hub/backbones.py
WEIGHTS_PLACEHOLDER_URL = "http://localhost:8300"
WEIGHTS_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

EMB_PREFIX = f"embeddings/{HUB_MODEL}/train_t{N_TEMPORAL}"
PACK_PREFIX = f"embeddings/{HUB_MODEL}/packs_t{N_TEMPORAL}"
TMP_DIR = Path("./tmp_embed")

SPLITS = ("train", "val", "cal")

META_VARIANTS = "__variants__"
META_FRAMES = "__n_frames__"

try:
    from google.colab import userdata
    os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
except Exception:
    pass
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


s3 = None


CACHE_READ_WORKERS = 32


def _ensure_s3():
    global s3
    if s3 is None:
        from botocore.config import Config
        s3 = boto3.client("s3", config=Config(
            max_pool_connections=CACHE_READ_WORKERS + 8,
            retries={"max_attempts": 5, "mode": "standard"}))
    return s3


def safe_local_name(name, max_len=80):
    stem, ext = os.path.splitext(name)
    if len(stem) <= max_len:
        return name
    return f"{stem[:max_len]}_{hashlib.sha256(stem.encode()).hexdigest()[:16]}{ext}"


def load_split():
    body = _ensure_s3().get_object(Bucket=BUCKET, Key=SPLIT_KEY)["Body"].read()
    doc = json.loads(body)
    print(f"split v{doc['version']} ({doc['created_utc']}), "
          f"{len(doc['clips'])} clips, seed {doc['seed']}")
    return doc


def variants_needed(entry, superficial):
    if entry["split"] == "train":
        return ["clean"] + list(entry["train_perturbations"])
    if entry["split"] == "val":
        return ["clean"] + list(superficial)
    return ["clean"]


def video_key(doc, stem, variant):
    if variant == "clean":
        return doc["clips"][stem]["source_key"]
    return f"{doc['attacks_prefix']}{stem}/{variant}.mp4"


def emb_key(stem):
    return f"{EMB_PREFIX}/{safe_local_name(stem + '.npz')}"


def fmt_secs(s):
    s = int(s)
    if s >= 3600:
        return f"{s//3600}h{(s%3600)//60:02d}m"
    if s >= 60:
        return f"{s//60}m{s%60:02d}s"
    return f"{s}s"


def read_clip(path):
    """Decode sequentially and return BGR frames; embed() does the conversion."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {path}")
    return frames


def frame_indices(n_frames, k=NUM_FRAMES):
    return np.linspace(0, n_frames - 1, k).round().astype(int)


def patch_hub_base_url():
    import sys
    import torch

    fixed = []
    for path in Path(torch.hub.get_dir()).glob("*vjepa2*/src/hub/backbones.py"):
        src = path.read_text(encoding="utf-8")
        if WEIGHTS_PLACEHOLDER_URL in src:
            path.write_text(src.replace(WEIGHTS_PLACEHOLDER_URL,
                                        WEIGHTS_BASE_URL), encoding="utf-8")
            fixed.append(str(path))

    # fix anything already imported too, so the session needs no restart
    for name, mod in list(sys.modules.items()):
        if getattr(mod, "VJEPA_BASE_URL", None) == WEIGHTS_PLACEHOLDER_URL:
            mod.VJEPA_BASE_URL = WEIGHTS_BASE_URL
            fixed.append(f"sys.modules[{name!r}]")

    for f in fixed:
        print(f"  patched base url in {f}")
    return fixed


def load_encoder(device="cuda"):
    import torch

    print(f"loading {HUB_MODEL} from torch.hub ...")
    # the preprocessor pulls the repo but no weights, so patch in between
    processor = torch.hub.load(HUB_REPO, HUB_PREPROCESSOR,
                               crop_size=CROP_SIZE, trust_repo=True)
    patch_hub_base_url()

    print("  fetching weights (~5.2 GB on a cold cache) ...")
    encoder, _predictor = torch.hub.load(HUB_REPO, HUB_MODEL, trust_repo=True)
    del _predictor

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    n = sum(p.numel() for p in encoder.parameters())
    print(f"  encoder on {device}, {n/1e6:.0f}M params, frozen")
    return processor, encoder


def embed(processor, encoder, frames, device="cuda"):
    import torch

    idx = frame_indices(len(frames))
    # the preprocessor normalizes with ImageNet mean/std, which assumes RGB
    buf = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx])
    assert buf.shape[0] == NUM_FRAMES and buf.dtype == np.uint8, buf.shape

    clip = processor(buf)
    if isinstance(clip, (list, tuple)):
        clip = clip[0]
    if clip.ndim != 4:
        raise RuntimeError(f"preprocessor returned ndim={clip.ndim}, expected "
                           f"4 (C,T,H,W); shape {tuple(clip.shape)}")
    if clip.shape[0] != 3 and clip.shape[1] == 3:
        clip = clip.permute(1, 0, 2, 3)

    x = clip.unsqueeze(0).to(device)

    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            tokens = encoder(x)
    if isinstance(tokens, (list, tuple)):
        tokens = tokens[-1]
    if tokens.ndim != 3:
        raise RuntimeError(f"encoder returned ndim={tokens.ndim}, expected "
                           f"3 (B,N,D); shape {tuple(tokens.shape)}")

    seq = pool_tokens(tokens).cpu().numpy().astype(np.float16)
    if seq.shape != (N_TEMPORAL, EMBED_DIM):
        raise RuntimeError(f"pooled to {seq.shape}, expected "
                           f"({N_TEMPORAL}, {EMBED_DIM})")
    return seq, len(frames), int(tokens.shape[1])


def pool_tokens(tokens):
    """(1, T*S, D) -> (T, D): mean over the spatial tokens of each moment.

    Assumes the encoder flattens tokens temporal-major. A spatial-major layout
    reshapes just as cleanly, so verify_temporal_axis() is what checks it.
    """
    import torch

    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise RuntimeError(f"expected (1, N, D) tokens, got "
                           f"{tuple(tokens.shape)}")
    n, d = int(tokens.shape[1]), int(tokens.shape[2])
    if d != EMBED_DIM:
        raise RuntimeError(f"encoder width {d}, expected {EMBED_DIM}")
    if n != N_TEMPORAL * N_SPATIAL:
        raise RuntimeError(
            f"encoder returned {n} tokens, expected {N_TEMPORAL}*{N_SPATIAL}="
            f"{N_TEMPORAL * N_SPATIAL}. Fix N_TEMPORAL/N_SPATIAL to match the "
            f"actual grid rather than letting the reshape regroup tokens.")

    # float32 first: averaging 576 bf16 values accumulates real error
    return tokens.float().reshape(N_TEMPORAL, N_SPATIAL, EMBED_DIM).mean(dim=1)


def _cos_rows(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=1)


def verify_temporal_axis(processor, encoder, doc, device="cuda", n_clips=3):
    """Embed a clip and its reverse variant. If axis 0 is time, aligning the
    two back-to-front beats aligning them front-to-front."""
    stems = [s for s in sorted(doc["clips"])
             if doc["clips"][s]["split"] == "train"][:n_clips]
    flipped_all, ident_all = [], []

    print(f"\nverifying the temporal axis on {len(stems)} clips ...")
    for stem in stems:
        try:
            seqs = {}
            for variant in ("clean", "reverse"):
                local = download_video(video_key(doc, stem, variant))
                try:
                    frames = read_clip(local)
                finally:
                    local.unlink(missing_ok=True)
                seqs[variant], _, _ = embed(processor, encoder, frames, device)
        except Exception as e:
            print(f"  skip {stem[:40]}: {type(e).__name__}: {e}")
            continue

        clean, rev = seqs["clean"], seqs["reverse"]
        flipped = float(_cos_rows(clean, rev[::-1]).mean())
        ident = float(_cos_rows(clean, rev).mean())
        flipped_all.append(flipped)
        ident_all.append(ident)
        print(f"  {stem[:44]:44s} flipped {flipped:.4f}  identity {ident:.4f}")

    if not flipped_all:
        print("  FAIL  no clip could be embedded; cannot verify")
        return False
    f, i = float(np.mean(flipped_all)), float(np.mean(ident_all))
    ok = f > i
    print(f"  mean flipped {f:.4f} vs identity {i:.4f}  ->  "
          + ("PASS, axis 0 is time" if ok else
             "FAIL, axis 0 is NOT time, do not trust the reshape"))
    return ok


def compare_to_mean_pooled(stems=None, limit=20, old_prefix=None):
    """new.mean(axis=0) against the old mean-pooled cache, relative to scale.

    Both average the same 18432 tokens, so this catches a bad reshape or a
    changed preprocessor. It cannot catch a time/space swap: a mean is
    order-invariant.
    """
    old_prefix = old_prefix or f"embeddings/{HUB_MODEL}/train"
    doc = load_split()
    stems = stems or [s for s in sorted(doc["clips"])][:limit]

    worst, checked = 0.0, 0
    for stem in stems:
        new, _ = read_cached(stem)
        try:
            body = _ensure_s3().get_object(
                Bucket=BUCKET,
                Key=f"{old_prefix}/{safe_local_name(stem + '.npz')}",
            )["Body"].read()
        except Exception:
            continue
        with np.load(io.BytesIO(body)) as z:
            old = {k: z[k] for k in z.files
                   if k not in (META_VARIANTS, META_FRAMES)}
        for variant, seq in new.items():
            if variant not in old or seq.ndim != 2:
                continue
            ref = old[variant].astype(np.float32)
            d = float(np.abs(seq.astype(np.float32).mean(axis=0) - ref).max())
            worst = max(worst, d / max(float(np.abs(ref).max()), 1e-6))
            checked += 1

    if not checked:
        print("no overlapping variants found between the two caches")
        return None
    # bf16 autocast puts the floor on cross-session agreement near 3e-4
    print(f"compared {checked} variants; largest relative "
          f"|new.mean(0) - old| = {worst:.2e}"
          + ("  PASS" if worst < 5e-3 else "  FAIL, the reshape changed what "
                                           "is being averaged"))
    return worst


def read_many(stems, workers=None, every=250):
    workers = workers or CACHE_READ_WORKERS
    done, lock, n = 0, threading.Lock(), len(stems)
    t0 = time.perf_counter()

    def one(stem):
        nonlocal done
        got = read_cached(stem)
        with lock:
            done += 1
            if done % every == 0 or done == n:
                el = time.perf_counter() - t0
                print(f"    {done}/{n}  {el:.0f}s elapsed, "
                      f"~{(n-done)*el/done:.0f}s left", flush=True)
        return got

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(stems, pool.map(one, stems)))


def list_cached_keys():
    keys = set()
    paginator = _ensure_s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=EMB_PREFIX):
        for obj in page.get("Contents", []):
            keys.add(obj["Key"])
    return keys


def read_cached(stem):
    """-> ({variant: (T,D) array}, {variant: n_frames}); empty if nothing cached."""
    try:
        body = _ensure_s3().get_object(Bucket=BUCKET, Key=emb_key(stem))
        body = body["Body"].read()
    except Exception:
        return {}, {}
    with np.load(io.BytesIO(body)) as z:
        names = list(z[META_VARIANTS]) if META_VARIANTS in z else []
        counts = list(z[META_FRAMES]) if META_FRAMES in z else []
        frames = {str(n): int(c) for n, c in zip(names, counts)}
        vectors = {k: z[k] for k in z.files
                   if k not in (META_VARIANTS, META_FRAMES)}
    return vectors, frames


def write_cached(stem, vectors, frames):
    names = sorted(vectors)
    payload = dict(vectors)
    payload[META_VARIANTS] = np.array(names, dtype="<U64")
    payload[META_FRAMES] = np.array([frames.get(n, -1) for n in names],
                                    dtype=np.int32)
    buf = io.BytesIO()
    np.savez(buf, **payload)
    key = emb_key(stem)
    _ensure_s3().put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    return key


def download_video(key):
    local = TMP_DIR / "video" / safe_local_name(Path(key).name)
    local.parent.mkdir(parents=True, exist_ok=True)
    _ensure_s3().download_file(BUCKET, key, str(local))
    return local


def plan(doc, splits, limit=None, workers=CACHE_READ_WORKERS):
    """-> [(stem, [variants still needed])], already-cached ones excluded.

    One LIST says which clips have a record at all; only those get read, and
    those reads go out in parallel. A fresh run does zero GETs.
    """
    superficial = doc["superficial_variants"]
    stems = [s for s in sorted(doc["clips"])
             if doc["clips"][s]["split"] in splits]

    existing = list_cached_keys()
    to_read = [s for s in stems if emb_key(s) in existing]
    contents = {}
    if to_read:
        print(f"  reading {len(to_read)} cached records "
              f"({workers} threads) ...", flush=True)
        contents = {s: got[0] for s, got in read_many(to_read, workers).items()}

    todo, cached, total, partial = [], 0, 0, False
    for stem in stems:
        if limit and len(todo) >= limit:
            partial = True
            break
        needed = variants_needed(doc["clips"][stem], superficial)
        total += len(needed)
        have = contents.get(stem, {})
        missing = [v for v in needed if v not in have]
        cached += len(needed) - len(missing)
        if missing:
            todo.append((stem, missing))
    return todo, cached, total, partial


def run(splits=SPLITS, limit=None, device="cuda", dry_run=False, push_to_s3=True):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    doc = load_split()

    print(f"\nchecking cache for splits={list(splits)} ...")
    todo, cached, total, partial = plan(doc, splits, limit)

    pending = sum(len(v) for _, v in todo)
    scope = f"first {total} scanned" if partial else f"{total} called for"
    print(f"  {scope}, {cached} already cached")
    print(f"  {len(todo)} clips to touch, {pending} embeddings to compute")
    if not push_to_s3:
        print("  push_to_s3=False: nothing will be uploaded")
    if dry_run:
        print("\ndry run; no model loaded")
        return {}
    if not todo:
        print("\nnothing to do.")
        return {}

    processor, encoder = load_encoder(device)

    results = {}
    done, failed, mismatches = 0, 0, []
    print("\nthe first clip prints per variant so the shapes can be checked\n")
    t_start = time.perf_counter()

    for n, (stem, missing) in enumerate(todo, 1):
        entry = doc["clips"][stem]

        vectors, frame_counts = read_cached(stem)
        # clean first: its frame count is what the others are checked against
        ordered = (["clean"] if "clean" in missing else []) + \
                  [v for v in missing if v != "clean"]

        # the GPU is idle through every download and the objects are independent
        def _fetch(v, _stem=stem):
            try:
                return download_video(video_key(doc, _stem, v))
            except Exception as exc:
                return exc

        with ThreadPoolExecutor(max_workers=min(len(ordered), 8)) as pool:
            fetched = dict(zip(ordered, pool.map(_fetch, ordered)))

        ok = 0
        for variant in ordered:
            try:
                local = fetched[variant]
                if isinstance(local, Exception):
                    raise local
                try:
                    frames = read_clip(local)
                finally:
                    local.unlink(missing_ok=True)

                vec, n_frames, n_tokens = embed(processor, encoder, frames, device)
                vectors[variant] = vec
                frame_counts[variant] = n_frames

                ref = frame_counts.get("clean")
                if variant != "clean" and ref is not None and ref != n_frames:
                    mismatches.append((stem, variant, ref, n_frames))
                    print(f"  WARN {stem[:40]} {variant}: {n_frames} frames "
                          f"vs clean {ref}")

                if n == 1:
                    print(f"    {variant:42s} {n_frames} frames -> {n_tokens} "
                          f"tokens -> {vec.shape} {vec.dtype}")
                done += 1
                ok += 1
            except Exception as e:
                failed += 1
                print(f"  FAILED {stem[:40]} {variant}: {type(e).__name__}: {e}")

        results[stem] = vectors
        if vectors and push_to_s3:
            write_cached(stem, vectors, frame_counts)

        elapsed = time.perf_counter() - t_start
        eta = (pending - done) * elapsed / done if done else 0
        print(f"[{n:>4}/{len(todo)}] {entry['split']:<5} pc={entry['pc']} "
              f"{ok}/{len(ordered)}  {done:>5}/{pending}  "
              f"eta {fmt_secs(eta)}  {stem[:44]}")

    print(f"\ndone. {done} embedded, {failed} failed, "
          f"{len(mismatches)} frame-count mismatches"
          + ("" if push_to_s3 else " (nothing uploaded)"))
    if failed:
        print("  check the log for FAILED; a clean exit is not a clean run")
    for m in mismatches[:10]:
        print(f"  mismatch {m[0]} {m[1]}: clean={m[2]} variant={m[3]}")
    return results


def consolidate(splits=SPLITS):
    """Collapse the per-clip npz files into one array per split. Rows are
    (stem, variant); a perturbed row carries its clean clip's label."""
    doc = load_split()
    superficial = doc["superficial_variants"]

    for split in splits:
        stems = sorted(s for s, e in doc["clips"].items() if e["split"] == split)
        X, meta_stem, meta_variant, pc, sa, joint = [], [], [], [], [], []
        missing = 0

        print(f"{split}: reading {len(stems)} records ...", flush=True)
        fetched = read_many(stems)

        for stem in stems:
            entry = doc["clips"][stem]
            have = fetched[stem][0]
            for variant in variants_needed(entry, superficial):
                if variant not in have:
                    missing += 1
                    continue
                X.append(have[variant])
                meta_stem.append(stem)
                meta_variant.append(variant)
                pc.append(entry["pc"])
                sa.append(-1 if entry["sa"] is None else entry["sa"])
                joint.append(-1 if entry["joint"] is None else entry["joint"])

        if not X:
            print(f"{split}: nothing cached, skipping")
            continue

        key = f"{PACK_PREFIX}/{split}.npz"
        local = TMP_DIR / Path(key).name
        local.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            local,
            X=np.stack(X).astype(np.float16),
            stem=np.array(meta_stem, dtype="<U160"),
            variant=np.array(meta_variant, dtype="<U64"),
            pc=np.array(pc, dtype=np.int8),
            sa=np.array(sa, dtype=np.int8),
            joint=np.array(joint, dtype=np.int8),
        )
        try:
            _ensure_s3().upload_file(str(local), BUCKET, key)
        finally:
            local.unlink(missing_ok=True)

        print(f"{split}: {len(X)} rows x {N_TEMPORAL} x {EMBED_DIM} -> "
              f"s3://{BUCKET}/{key}"
              + (f"  ({missing} uncached rows skipped)" if missing else ""))
        for name, c in sorted(Counter(meta_variant).items()):
            print(f"    {c:>6}  {name}")


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(description="cache frozen V-JEPA 2.1 embeddings")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--limit", type=int, default=None,
                    help="only touch the first N clips that need work")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what is cached and what is missing")
    ap.add_argument("--no-push", action="store_true",
                    help="compute but upload nothing (smoke tests)")
    ap.add_argument("--consolidate", action="store_true",
                    help="pack the per-clip npz files into one per split")
    ap.add_argument("--verify-axis", action="store_true",
                    help="confirm axis 0 of the pooled tensor is time")
    ap.add_argument("--compare-mean", action="store_true",
                    help="check new.mean(0) against the old mean-pooled cache")
    args = ap.parse_args()

    if args.consolidate:
        consolidate(splits=args.splits)
    elif args.compare_mean:
        compare_to_mean_pooled(limit=args.limit or 20)
    elif args.verify_axis:
        proc, enc = load_encoder(args.device)
        raise SystemExit(0 if verify_temporal_axis(
            proc, enc, load_split(), args.device) else 1)
    else:
        run(splits=args.splits, limit=args.limit, device=args.device,
            dry_run=args.dry_run, push_to_s3=not args.no_push)
