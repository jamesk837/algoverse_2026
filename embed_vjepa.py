import argparse
import hashlib
import io
import json
import os
import threading
import time
from collections import Counter, defaultdict
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

# Rebound by select_dataset(). The default is the videophy2_train split cache,
# so every key written before 2026-08-22 keeps its name.
EMB_PREFIX = f"embeddings/{HUB_MODEL}/train_t{N_TEMPORAL}"
PACK_PREFIX = f"embeddings/{HUB_MODEL}/packs_t{N_TEMPORAL}"
TMP_DIR = Path("./tmp_embed")

SPLITS = ("train", "val", "cal")

# The phase-1 judge-audit corpora. These have no split file: no train/val/cal,
# no perturbation assignment, and (for ImplausiBench) no human labels at all.
# The point of embedding them is that the three judges scored these exact clips,
# so putting a probe on the same representations is what finally lets a probe
# delta and a judge delta be compared on one video.
#
# source_prefix mirrors attack_suite.DATASET_PREFIXES and attacks_prefix mirrors
# attack_suite.dest_key -- the alias in the middle of "attacks/<alias>/" is
# attack_suite's dataset name, which is "test", not "videophy2_test".
BENCHMARK_DATASETS = {
    "videophy2_test": {
        "source_prefix": "datasets/videophy2_test/",
        "attacks_prefix": "attacks/test/",
        "metadata_prefix": "datasets/videophy2_test/_metadata/",
    },
    "implausibench_real": {
        "source_prefix": "datasets/implausibench/ImplausiBench/real/",
        "attacks_prefix": "attacks/implausibench_real/",
        "metadata_prefix": None,
    },
    "implausibench_implausible": {
        "source_prefix": "datasets/implausibench/ImplausiBench/implausible/",
        "attacks_prefix": "attacks/implausibench_implausible/",
        "metadata_prefix": None,
    },
}

VIDEO_SUFFIXES = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

# Mirrors build_split.SUPERFICIAL_VARIANTS. The split file carries this list, so
# the training path still reads it from the doc and never touches this constant;
# a synthesized benchmark doc has no file to read it from. Keep the two in sync.
SUPERFICIAL_VARIANTS = (
    "photometric",
    "caption_echo_rubric_vocab",
    "caption_echo_score_anchor_positive",
    "caption_echo_authoritative_claim",
    "caption_echo_score_anchor_negative",
    "caption_echo_control_irrelevant",
)

# also from build_split, for reading a benchmark's metadata CSV
ID_COLUMN = "video_url"
LABEL_COLUMNS = ("pc", "sa", "joint")

# already rendered by attack_suite, only the embeddings are missing. Kept out
# of split_v1.json because a rebuild reshuffles train/val/cal and would
# invalidate the cache.
TEMPORAL_VARIANTS = ("shuffle", "reverse", "freeze")

# Which splits get the full variant set. val has carried it since 2026-08-19;
# cal was added 2026-08-21 after the mentor read n=336 as half the held-out
# set -- val and cal are 10% each, so evaluating on val alone does use half of
# what exists. Purely additive: the stem -> split assignment is untouched and
# the run stays resumable per (clip, variant). `--variant-splits val`
# reproduces the pre-2026-08-21 cache.
HELD_OUT_SPLITS = ("val", "cal")

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


def list_keys(prefix):
    keys = []
    paginator = _ensure_s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def stem_from_url(url):
    """must match attack_suite.dest_key and build_split.stem_from_url"""
    return Path(str(url).split("?")[0].split("/")[-1]).stem


def select_dataset(dataset=None):
    """Point the per-clip cache at one corpus. None keeps the training split's
    prefix, so nothing already written moves."""
    global EMB_PREFIX
    EMB_PREFIX = (f"embeddings/{HUB_MODEL}/train_t{N_TEMPORAL}" if dataset is None
                  else f"embeddings/{HUB_MODEL}/{dataset}_t{N_TEMPORAL}")
    return EMB_PREFIX


def list_source_videos(prefix):
    return sorted(k for k in list_keys(prefix)
                  if not k.endswith("/") and "/_metadata/" not in k
                  and Path(k).suffix.lower() in VIDEO_SUFFIXES)


def benchmark_labels(cfg):
    """stem -> {pc, sa, joint} from the corpus metadata CSV, or {} if it has
    none. ImplausiBench ships no human physics scores at all, and the
    embeddings are worth having either way, so anything missing here warns and
    returns empty instead of raising."""
    import csv as _csv

    prefix = cfg.get("metadata_prefix")
    if not prefix:
        return {}
    csvs = [k for k in list_keys(prefix) if k.lower().endswith(".csv")]
    if not csvs:
        print(f"  no CSV under {prefix}; clips carry no human labels")
        return {}

    key = sorted(csvs)[0]
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    local = TMP_DIR / Path(key).name
    _ensure_s3().download_file(BUCKET, key, str(local))
    try:
        with open(local, encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(_csv.DictReader(fh))
    finally:
        local.unlink(missing_ok=True)

    if not rows or ID_COLUMN not in rows[0]:
        print(f"  {key} has no '{ID_COLUMN}' column "
              f"({sorted(rows[0]) if rows else 'empty'}); no labels")
        return {}

    def as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    labels = {}
    for row in rows:
        stem = stem_from_url(row[ID_COLUMN])
        if not stem or stem in labels:
            continue          # first occurrence wins, as in build_split
        labels[stem] = {c: as_int(row.get(c)) for c in LABEL_COLUMNS}
    print(f"  {len(rows)} metadata rows -> {len(labels)} labelled stems "
          f"({Path(key).name})")
    return labels


def restrict_to_rendered(doc):
    """Drop clips with no rendered variant at all.

    A benchmark corpus is defined by the attack_suite sweep, not by what sits in
    the dataset folder: videophy2_test holds 1638 clips but only 450 were ever
    rendered, and a clip with no variants cannot contribute a within-clip delta,
    which is the only thing these embeddings are for.

    Deliberately "zero renders", not "all nine". A clip missing one variant is
    still a perfectly good clip -- it stays, the gap is reported by
    check_sources(), and build_eval() masks that variant. This drops only clips
    that were never in the sweep.
    """
    have = rendered_variants(doc)
    keep = {s: e for s, e in doc["clips"].items() if have.get(s)}
    dropped = len(doc["clips"]) - len(keep)
    if dropped:
        print(f"  {dropped} clips have no rendered variant and are not part of "
              f"the curated benchmark; {len(keep)} remain")
    if not keep:
        raise RuntimeError(
            f"no clip under {doc['attacks_prefix']} has any rendered variant; "
            "run attack_suite for this dataset first, or pass "
            "rendered_only=False to embed the clean clips anyway")
    doc["clips"] = keep
    return doc


def load_benchmark(dataset, rendered_only=True):
    """Synthesize a split-file-shaped doc for a benchmark corpus.

    Everything downstream -- plan(), run(), consolidate(), check_sources(),
    variants_needed() -- reads the doc and nothing else, so matching its shape
    is what makes the benchmark path free of forked code. Two fields do the
    work: every clip's `split` is the dataset name, and `held_out_splits` is
    that same name, which is what makes variants_needed() hand back the full
    clean + 6 superficial + 3 temporal set for every clip.
    """
    cfg = BENCHMARK_DATASETS[dataset]
    keys = list_source_videos(cfg["source_prefix"])
    if not keys:
        raise RuntimeError(
            f"no videos under s3://{BUCKET}/{cfg['source_prefix']}")
    labels = benchmark_labels(cfg)

    clips, dupes = {}, 0
    for key in keys:
        stem = Path(key).stem
        if stem in clips:
            dupes += 1
            continue
        lab = labels.get(stem, {})
        clips[stem] = {"split": dataset, "source_key": key,
                       "pc": lab.get("pc"), "sa": lab.get("sa"),
                       "joint": lab.get("joint")}

    n_lab = sum(1 for c in clips.values() if c["pc"] is not None)
    print(f"benchmark {dataset}: {len(clips)} clips "
          f"({n_lab} with a human PC label), attacks under "
          f"{cfg['attacks_prefix']}")
    if dupes:
        print(f"  WARN {dupes} source keys share a stem and were dropped; "
              "attacks/ is keyed by stem, so they are indistinguishable")
    if not n_lab:
        print("  no human labels: the pack's pc/sa/joint are all -1. Deltas "
              "are within-clip and need no label, but MAE is not computable.")

    doc = {
        "version": f"benchmark:{dataset}",
        "created_utc": "-",
        "seed": None,
        "dataset": dataset,
        "source_prefix": cfg["source_prefix"],
        "attacks_prefix": cfg["attacks_prefix"],
        "superficial_variants": list(SUPERFICIAL_VARIANTS),
        "val_temporal_variants": list(TEMPORAL_VARIANTS),
        "held_out_splits": [dataset],
        "clips": clips,
    }
    return restrict_to_rendered(doc) if rendered_only else doc


def load_doc(dataset=None, rendered_only=True):
    """The one entry point: the training split, or a benchmark corpus."""
    select_dataset(dataset)
    if dataset is None:
        return load_split()
    if dataset not in BENCHMARK_DATASETS:
        raise ValueError(f"dataset must be one of "
                         f"{list(BENCHMARK_DATASETS)} or None")
    return load_benchmark(dataset, rendered_only=rendered_only)


def load_split():
    body = _ensure_s3().get_object(Bucket=BUCKET, Key=SPLIT_KEY)["Body"].read()
    doc = json.loads(body)
    print(f"split v{doc['version']} ({doc['created_utc']}), "
          f"{len(doc['clips'])} clips, seed {doc['seed']}")
    return doc


def variants_needed(entry, superficial, temporal=TEMPORAL_VARIANTS,
                    held_out=HELD_OUT_SPLITS):
    if entry["split"] == "train":
        return ["clean"] + list(entry["train_perturbations"])
    if entry["split"] in held_out:
        return ["clean"] + list(superficial) + list(temporal)
    return ["clean"]


def temporal_variants(doc, enabled=True):
    """A future split version can pin these; until then the constant is it."""
    if not enabled:
        return ()
    return tuple(doc.get("val_temporal_variants") or TEMPORAL_VARIANTS)


def held_out_splits(doc, override=None):
    """Same deal: split_v1.json predates the field, so fall back to the
    constant. An explicit override (including an empty one) always wins."""
    if override is not None:
        return tuple(override)
    return tuple(doc.get("held_out_splits") or HELD_OUT_SPLITS)


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
    """Decode sequentially and return BGR frames; embed() converts to RGB."""
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
    Assumes temporal-major flattening; verify_temporal_axis() checks that."""
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
    # a benchmark doc has no "train" split; every clip there carries the full
    # variant set anyway, so fall through to whatever the doc does have
    have_train = any(e["split"] == "train" for e in doc["clips"].values())
    stems = [s for s in sorted(doc["clips"])
             if not have_train or doc["clips"][s]["split"] == "train"][:n_clips]
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
    Catches a bad reshape or a changed preprocessor, but not a time/space swap
    -- a mean is order-invariant."""
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


def rendered_variants(doc):
    """-> {stem: {variant}} actually present under attacks/. One paginated
    LIST. Used to say up front which renders are missing, instead of finding
    out as per-variant FAILED lines an hour into a GPU run."""
    out = defaultdict(set)
    prefix = doc["attacks_prefix"]
    paginator = _ensure_s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            parts = obj["Key"][len(prefix):].split("/")
            if len(parts) == 2 and parts[1].endswith(".mp4"):
                out[parts[0]].add(parts[1][: -len(".mp4")])
    return out


def check_sources(doc, todo, held_out):
    """Report needed variants with no rendered .mp4 behind them."""
    print("  checking that every needed variant is rendered ...", flush=True)
    have = rendered_variants(doc)
    short = Counter()
    clips = 0
    for stem, missing in todo:
        gone = [v for v in missing if v != "clean" and v not in have.get(stem, ())]
        if gone:
            clips += 1
            short.update(gone)
    if not short:
        print("  all rendered.")
        return short
    print(f"  {clips} clips are short a render, {sum(short.values())} "
          f"variants total:")
    for name, c in sorted(short.items()):
        print(f"    {c:>6}  {name}")
    print("  these will print FAILED and be masked out per variant; rerun "
          "attack_suite for them")
    return short


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


def plan(doc, splits, limit=None, workers=CACHE_READ_WORKERS, temporal=True,
         held_out=None):
    """-> [(stem, [variants still needed])], already-cached ones excluded.
    One LIST, then parallel GETs on only the clips that have a record.

    held_out=None resolves from the doc, so a caller that forgets it does not
    silently plan `clean` only for a corpus whose every clip wants all ten."""
    superficial = doc["superficial_variants"]
    temp = temporal_variants(doc, temporal)
    held_out = held_out_splits(doc, held_out)
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
        needed = variants_needed(doc["clips"][stem], superficial, temp,
                                 held_out)
        total += len(needed)
        have = contents.get(stem, {})
        missing = [v for v in needed if v not in have]
        cached += len(needed) - len(missing)
        if missing:
            todo.append((stem, missing))
    return todo, cached, total, partial


def run(splits=SPLITS, limit=None, device="cuda", dry_run=False,
        push_to_s3=True, temporal=True, held_out=None, dataset=None,
        encoder=None, rendered_only=True):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    doc = load_doc(dataset, rendered_only=rendered_only)
    if dataset is not None:
        splits = (dataset,)      # a benchmark corpus is one undivided split

    temp = temporal_variants(doc, temporal)
    held_out = held_out_splits(doc, held_out)
    print(f"\nchecking cache for splits={list(splits)} ...")
    print(f"  full variant set on: {', '.join(held_out) or '(none)'}")
    if temp:
        print(f"  including the expected-sensitivity variants: "
              f"{', '.join(temp)}")
    todo, cached, total, partial = plan(doc, splits, limit, temporal=temporal,
                                        held_out=held_out)

    pending = sum(len(v) for _, v in todo)
    scope = f"first {total} scanned" if partial else f"{total} called for"
    print(f"  {scope}, {cached} already cached")
    print(f"  {len(todo)} clips to touch, {pending} embeddings to compute")
    if not push_to_s3:
        print("  push_to_s3=False: nothing will be uploaded")
    if dry_run:
        check_sources(doc, todo, held_out)
        print("\ndry run; no model loaded")
        return {}
    if not todo:
        print("\nnothing to do.")
        return {}

    # run_all() passes one already-loaded encoder so a multi-corpus run pays
    # the ~5.2 GB weight load once instead of once per dataset
    processor, encoder = encoder or load_encoder(device)

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

    run.last = {"embedded": done, "failed": failed,
                "mismatches": len(mismatches)}
    print(f"\ndone. {done} embedded, {failed} failed, "
          f"{len(mismatches)} frame-count mismatches"
          + ("" if push_to_s3 else " (nothing uploaded)"))
    if failed:
        print("  check the log for FAILED; a clean exit is not a clean run")
    for m in mismatches[:10]:
        print(f"  mismatch {m[0]} {m[1]}: clean={m[2]} variant={m[3]}")
    return results


def expand_datasets(spec):
    """'all' -> every benchmark corpus; None or 'train' -> the training split
    (represented as None, the way load_doc expects it)."""
    if spec is None:
        return (None,)
    names = [spec] if isinstance(spec, str) else list(spec)
    out = []
    for name in names:
        if name == "all":
            out.extend(BENCHMARK_DATASETS)
        elif name == "train":
            out.append(None)
        else:
            if name not in BENCHMARK_DATASETS:
                raise ValueError(f"unknown dataset {name!r}; pick from "
                                 f"{list(BENCHMARK_DATASETS)}, 'train' or 'all'")
            out.append(name)
    return tuple(dict.fromkeys(out))


def run_all(datasets="all", pack=True, device="cuda", **kw):
    """Embed several corpora back to back, then pack each one.

    The point is a single unattended command: the encoder is loaded once and
    reused, and a corpus that blows up does not take the others down with it --
    it is reported in the summary and the run moves on. Everything stays
    resumable per (clip, variant), so rerunning this after a failure only picks
    up what is missing.
    """
    names = expand_datasets(datasets)
    label = ", ".join(n or "train" for n in names)
    print(f"embedding {len(names)} corpora: {label}")

    encoder = None
    if not kw.get("dry_run"):
        encoder = load_encoder(device)

    summary = []
    for i, name in enumerate(names, 1):
        title = name or "videophy2_train"
        print(f"\n{'#' * 70}\n# [{i}/{len(names)}] {title}\n{'#' * 70}")
        run.last = None
        try:
            run(dataset=name, device=device, encoder=encoder, **kw)
            got = run.last or {"embedded": 0, "failed": 0}
            summary.append((title, got["embedded"], got["failed"], None))
        except Exception as e:
            print(f"DATASET FAILED {title}: {type(e).__name__}: {e}")
            summary.append((title, 0, 0, f"{type(e).__name__}: {e}"))
            continue

        if pack and not kw.get("dry_run") and kw.get("push_to_s3", True):
            try:
                consolidate(dataset=name, temporal=kw.get("temporal", True),
                            held_out=kw.get("held_out"),
                            rendered_only=kw.get("rendered_only", True))
            except Exception as e:
                print(f"CONSOLIDATE FAILED {title}: {type(e).__name__}: {e}")

    print(f"\n{'=' * 70}\nsummary")
    print(f"  {'corpus':<28} {'embedded':>9} {'failed':>7}")
    for title, done, failed, err in summary:
        note = f"   ERROR {err}" if err else ""
        print(f"  {title:<28} {done:>9} {failed:>7}{note}")
    if any(f for _, _, f, _ in summary) or any(e for *_, e in summary):
        print("  a clean exit is not a clean run: check the lines above")
    return summary


def consolidate(splits=SPLITS, temporal=True, held_out=None, dataset=None,
                rendered_only=True):
    """Collapse the per-clip npz files into one array per split. Rows are
    (stem, variant); a perturbed row carries its clean clip's label. For the
    temporal rows that label is wrong by construction -- nothing trains on
    them, and the within-clip delta needs no label."""
    doc = load_doc(dataset, rendered_only=rendered_only)
    if dataset is not None:
        splits = (dataset,)
    superficial = doc["superficial_variants"]
    temp = temporal_variants(doc, temporal)
    held_out = held_out_splits(doc, held_out)

    for split in splits:
        stems = sorted(s for s, e in doc["clips"].items() if e["split"] == split)
        X, meta_stem, meta_variant, pc, sa, joint = [], [], [], [], [], []
        missing = 0

        print(f"{split}: reading {len(stems)} records ...", flush=True)
        fetched = read_many(stems)

        for stem in stems:
            entry = doc["clips"][stem]
            have = fetched[stem][0]
            for variant in variants_needed(entry, superficial, temp,
                                           held_out):
                if variant not in have:
                    missing += 1
                    continue
                X.append(have[variant])
                meta_stem.append(stem)
                meta_variant.append(variant)
                # -1 for absent: ImplausiBench has no human physics score at
                # all, and pc was never None on the training path
                pc.append(-1 if entry["pc"] is None else entry["pc"])
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
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS,
                    help="training-split subsets; ignored with --dataset")
    ap.add_argument("--dataset", nargs="+", default=None,
                    choices=list(BENCHMARK_DATASETS) + ["train", "all"],
                    help="embed phase-1 benchmark corpora instead of the "
                         "videophy2_train split: every clip, all 10 variants, "
                         "cached under embeddings/<model>/<dataset>_t32/. "
                         "Accepts several names, or 'all' for every benchmark "
                         "corpus in one unattended run.")
    ap.add_argument("--all-clips", action="store_true",
                    help="with --dataset, keep clips that have no rendered "
                         "variant at all (default: drop them -- a benchmark "
                         "corpus is what attack_suite actually rendered)")
    ap.add_argument("--no-pack", action="store_true",
                    help="with --dataset, skip the consolidate step that "
                         "normally runs after each corpus is embedded")
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
    ap.add_argument("--no-temporal", action="store_true",
                    help="skip shuffle/reverse/freeze on the held-out splits")
    ap.add_argument("--variant-splits", nargs="*", default=None,
                    help="splits that get the full variant set (default "
                         "val cal); pass 'val' alone to reproduce the "
                         "pre-2026-08-21 cache")
    args = ap.parse_args()
    temporal = not args.no_temporal
    held_out = None if args.variant_splits is None else tuple(args.variant_splits)

    datasets = expand_datasets(args.dataset)
    rendered_only = not args.all_clips

    if args.consolidate:
        for name in datasets:
            consolidate(splits=args.splits, temporal=temporal,
                        held_out=held_out, dataset=name,
                        rendered_only=rendered_only)
    elif args.compare_mean:
        compare_to_mean_pooled(limit=args.limit or 20)
    elif args.verify_axis:
        proc, enc = load_encoder(args.device)
        raise SystemExit(0 if verify_temporal_axis(
            proc, enc, load_doc(datasets[0]), args.device) else 1)
    elif len(datasets) == 1:
        run(splits=args.splits, limit=args.limit, device=args.device,
            dry_run=args.dry_run, push_to_s3=not args.no_push,
            temporal=temporal, held_out=held_out, dataset=datasets[0],
            rendered_only=rendered_only)
        if datasets[0] is not None and not args.no_pack and not args.dry_run \
                and not args.no_push:
            consolidate(temporal=temporal, held_out=held_out,
                        dataset=datasets[0], rendered_only=rendered_only)
    else:
        run_all(datasets=args.dataset, pack=not args.no_pack,
                device=args.device, limit=args.limit, dry_run=args.dry_run,
                push_to_s3=not args.no_push, temporal=temporal,
                held_out=held_out, rendered_only=rendered_only)
