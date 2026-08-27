

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np

BUCKET = "nickb-aarj"
DATASET = "train"
SOURCE_PREFIX = "datasets/videophy2_train/"
METADATA_PREFIX = "datasets/videophy2_train/_metadata/"
ATTACKS_PREFIX = "attacks/train/"
SPLIT_KEY = "splits/videophy2_train/split_v1.json"

VIDEO_SUFFIXES = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

# the expected-invariance half of the taxonomy. shuffle/reverse/freeze exist in
# S3 but are not part of the probe's training design.
SUPERFICIAL_VARIANTS = [
    "photometric",
    "caption_echo_rubric_vocab",
    "caption_echo_score_anchor_positive",
    "caption_echo_authoritative_claim",
    "caption_echo_score_anchor_negative",
    "caption_echo_control_irrelevant",
]

# The expected-sensitivity half. Held-out-only and NOT part of the probe's
# training design -- there is no label for a scrambled clip -- but embedded on
# the held-out splits so the probe can be asked whether it notices time at all.
# Availability is only reported here, never a membership gate: a clip missing
# one variant is still a perfectly good clip, and train_probe.build_eval masks
# per variant.
TEMPORAL_VARIANTS = [
    "shuffle",
    "reverse",
    "freeze",
]

# Splits that carry the full variant set. val since 2026-08-19, cal added
# 2026-08-21: val and cal are 10% each, so evaluating perturbations on val
# alone uses half the held-out clips.
HELD_OUT_SPLITS = ("val", "cal")

SPLITS = ("train", "val", "cal")
RATIOS = (0.80, 0.10, 0.10)
PC_LEVELS = (1, 2, 3, 4, 5)
PERTURBATIONS_PER_TRAIN_CLIP = 2
SEED = 20260816

LABEL_COLUMNS = ("pc", "sa", "joint")
ID_COLUMN = "video_url"

try:
    from google.colab import userdata
    os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
except Exception:
    pass  # not Colab, or no secrets set: fall back to the ambient chain
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def in_notebook():
    """pasting the whole file into a cell makes __name__ == '__main__', which
    would fire argparse against the kernel's own argv."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


s3 = None


def _ensure_s3():
    global s3
    if s3 is None:
        s3 = boto3.client("s3")
    return s3


def list_keys(prefix):
    keys = []
    paginator = _ensure_s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def stem_from_url(url):
    """must match attack_suite.dest_key and pushs3.stream_url_dataset_to_s3."""
    return Path(str(url).split("?")[0].split("/")[-1]).stem


def find_metadata_key():
    csvs = [k for k in list_keys(METADATA_PREFIX) if k.lower().endswith(".csv")]
    if not csvs:
        raise RuntimeError(f"no CSV under s3://{BUCKET}/{METADATA_PREFIX}")
    if len(csvs) > 1:
        print(f"  {len(csvs)} CSVs under _metadata/, using {csvs[0]}")
    return csvs[0]


def load_labels(meta_key):
    import pandas as pd

    local = Path("./tmp_split") / Path(meta_key).name
    local.parent.mkdir(parents=True, exist_ok=True)
    _ensure_s3().download_file(BUCKET, meta_key, str(local))
    try:
        df = pd.read_csv(local, on_bad_lines="skip", encoding="utf-8",
                         encoding_errors="replace")
    finally:
        local.unlink(missing_ok=True)

    missing = [c for c in (ID_COLUMN,) + LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{meta_key} is missing {missing}; columns are {list(df.columns)}")

    def as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    labels, dupes, dropped = {}, 0, 0
    for _, row in df.iterrows():
        pc = as_int(row["pc"])
        if pd.isna(row[ID_COLUMN]) or pc not in PC_LEVELS:
            dropped += 1
            continue
        stem = stem_from_url(row[ID_COLUMN])
        if stem in labels:
            dupes += 1
            continue
        labels[stem] = {"pc": pc, "sa": as_int(row["sa"]),
                        "joint": as_int(row["joint"])}

    print(f"  {len(df)} rows -> {len(labels)} labelled stems "
          f"({dupes} duplicate stems ignored, {dropped} rows unusable)")
    if dupes:
        print("  NOTE duplicates dropped keeping first occurrence; if train "
              "carries multiple annotations per clip they need aggregating.")
    return labels


def inventory():
    """stem -> clean source key, and stem -> superficial variants present."""
    sources = {}
    for key in list_keys(SOURCE_PREFIX):
        if key.startswith(METADATA_PREFIX):
            continue
        if Path(key).suffix.lower() in VIDEO_SUFFIXES:
            sources[Path(key).stem] = key

    variants = defaultdict(set)
    wanted = set(SUPERFICIAL_VARIANTS) | set(TEMPORAL_VARIANTS)
    for key in list_keys(ATTACKS_PREFIX):
        parts = key[len(ATTACKS_PREFIX):].split("/")
        if len(parts) != 2 or not parts[1].endswith(".mp4"):
            continue  # skips freeze.mp4.meta.json
        name = parts[1][: -len(".mp4")]
        if name in wanted:
            variants[parts[0]].add(name)

    sup = sum(1 for v in variants.values() if v & set(SUPERFICIAL_VARIANTS))
    tmp = sum(1 for v in variants.values() if v & set(TEMPORAL_VARIANTS))
    print(f"  {len(sources)} clean clips under {SOURCE_PREFIX}")
    print(f"  {sup} stems with at least one superficial variant")
    print(f"  {tmp} stems with at least one temporal variant")
    return sources, variants


def allocate(n, ratios):
    """largest-remainder split of n, summing to exactly n."""
    exact = [n * r for r in ratios]
    base = [int(math.floor(e)) for e in exact]
    order = sorted(range(len(ratios)), key=lambda i: (-(exact[i] - base[i]), i))
    for i in range(n - sum(base)):
        base[order[i % len(order)]] += 1
    return base


def _balanced(n, n_cats, rng):
    reps = int(math.ceil(n / n_cats)) if n else 0
    xs = np.tile(np.arange(n_cats), reps)[:n].copy()
    rng.shuffle(xs)
    return xs


def assign_pairs(n, n_cats, rng):
    """Two balanced passes, then swap-repair so no clip draws the same category
    twice. A per-clip rng.choice(replace=False) gives distinctness but lets the
    per-category totals wander by tens; this keeps them exact."""
    a = _balanced(n, n_cats, rng)
    b = _balanced(n, n_cats, rng)

    for _ in range(50):
        collisions = np.flatnonzero(a == b)
        if collisions.size == 0:
            break
        for i in collisions:
            if a[i] != b[i]:
                continue  # already fixed as someone else's swap partner
            for j in rng.permutation(n):
                if b[j] != a[i] and b[i] != a[j]:
                    b[i], b[j] = b[j], b[i]
                    break

    if n and (a == b).any():
        raise RuntimeError("could not build distinct balanced pairs")
    return a, b


def build(dry_run=False):
    print(f"=== building split for {DATASET} ===")

    meta_key = find_metadata_key()
    labels = load_labels(meta_key)
    sources, variants = inventory()

    complete, incomplete, unlabelled, no_source = [], {}, [], []
    for stem in sorted(labels):
        if stem not in sources:
            no_source.append(stem)
            continue
        missing = [v for v in SUPERFICIAL_VARIANTS
                   if v not in variants.get(stem, set())]
        if missing:
            incomplete[stem] = missing
        else:
            complete.append(stem)

    unlabelled = [s for s in sorted(variants) if s not in labels]

    print(f"\n  usable clips:             {len(complete)}")
    print(f"  labelled, no clean clip:  {len(no_source)}")
    print(f"  labelled, variants short: {len(incomplete)}")
    print(f"  variants, no label:       {len(unlabelled)}")

    if not complete:
        raise RuntimeError("no clip has a label, a clean source and all six "
                           "variants. Check that the stem join is working.")
    if incomplete:
        print("  -> rerun attack_suite for the short clips; they are excluded "
              "here and listed under excluded.missing_variants")

    rng = np.random.default_rng(SEED)

    by_pc = defaultdict(list)
    for stem in complete:
        by_pc[labels[stem]["pc"]].append(stem)

    assigned = {}
    for pc in PC_LEVELS:
        stems = sorted(by_pc.get(pc, []))
        if not stems:
            continue
        shuffled = [stems[i] for i in rng.permutation(len(stems))]
        cut = 0
        for split, k in zip(SPLITS, allocate(len(shuffled), RATIOS)):
            for stem in shuffled[cut:cut + k]:
                assigned[stem] = split
            cut += k

    train_stems = sorted(s for s, sp in assigned.items() if sp == "train")
    a, b = assign_pairs(len(train_stems), len(SUPERFICIAL_VARIANTS), rng)
    perturbations = {
        stem: [SUPERFICIAL_VARIANTS[a[i]], SUPERFICIAL_VARIANTS[b[i]]]
        for i, stem in enumerate(train_stems)
    }

    clips = {}
    for stem in sorted(assigned):
        entry = {
            "split": assigned[stem],
            "source_key": sources[stem],
            "pc": labels[stem]["pc"],
            "sa": labels[stem]["sa"],
            "joint": labels[stem]["joint"],
        }
        if assigned[stem] == "train":
            entry["train_perturbations"] = perturbations[stem]
        elif assigned[stem] in HELD_OUT_SPLITS:
            # recorded, not required: embed_vjepa asks for all three and
            # tolerates a miss, this just makes the shortfall visible up front
            have = variants.get(stem, set())
            missing_t = [v for v in TEMPORAL_VARIANTS if v not in have]
            if missing_t:
                entry["missing_temporal"] = missing_t
        clips[stem] = entry

    split_doc = {
        "version": "v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "seed": SEED,
        "ratios": dict(zip(SPLITS, RATIOS)),
        "stratified_on": "pc",
        "source_prefix": SOURCE_PREFIX,
        "attacks_prefix": ATTACKS_PREFIX,
        "metadata_key": meta_key,
        "superficial_variants": SUPERFICIAL_VARIANTS,
        # key name kept for compatibility with split_v1.json readers
        "val_temporal_variants": TEMPORAL_VARIANTS,
        "held_out_splits": list(HELD_OUT_SPLITS),
        "perturbations_per_train_clip": PERTURBATIONS_PER_TRAIN_CLIP,
        "clips": clips,
        "excluded": {
            "missing_variants": incomplete,
            "missing_source": no_source,
            "missing_label": unlabelled,
        },
    }

    report(split_doc)

    if dry_run:
        print(f"\ndry run; not uploading to s3://{BUCKET}/{SPLIT_KEY}")
        return split_doc

    body = json.dumps(split_doc, indent=2)
    local = Path("./tmp_split") / Path(SPLIT_KEY).name
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(body, encoding="utf-8")
    _ensure_s3().put_object(Bucket=BUCKET, Key=SPLIT_KEY,
                            Body=body.encode("utf-8"),
                            ContentType="application/json")
    print(f"\nuploaded -> s3://{BUCKET}/{SPLIT_KEY}")
    print(f"local copy -> {local}")
    return split_doc


def report(doc):
    clips = doc["clips"]
    per_split = Counter(c["split"] for c in clips.values())
    totals = Counter(c["pc"] for c in clips.values())

    print("\n  PC distribution (count / share within split)")
    print(f"    {'pc':>4} {'all':>13} {'train':>13} {'val':>13} {'cal':>13}")
    for pc in PC_LEVELS:
        row = f"    {pc:>4} {totals[pc]:>6} {totals[pc]/max(len(clips),1):>6.1%}"
        for split in SPLITS:
            n = sum(1 for c in clips.values()
                    if c["split"] == split and c["pc"] == pc)
            row += f" {n:>6} {n/max(per_split[split],1):>6.1%}"
        print(row)
    row = f"    {'all':>4} {len(clips):>6} {1.0:>6.1%}"
    for split in SPLITS:
        row += f" {per_split[split]:>6} {1.0:>6.1%}"
    print(row)

    low = sum(1 for c in clips.values() if c["pc"] <= 2)
    print(f"\n  PC<=2 total {low}: " + ", ".join(
        f"{s} {sum(1 for c in clips.values() if c['split'] == s and c['pc'] <= 2)}"
        for s in SPLITS))

    used = Counter()
    for c in clips.values():
        used.update(c.get("train_perturbations", []))
    if used:
        print(f"\n  train perturbation assignment ({sum(used.values())} slots "
              f"over {len(doc['superficial_variants'])} categories)")
        for name in doc["superficial_variants"]:
            print(f"    {used[name]:>6}  {name}")

    temporal = doc.get("val_temporal_variants") or []
    held_out = tuple(doc.get("held_out_splits") or HELD_OUT_SPLITS)
    n_held = sum(per_split[s] for s in held_out)
    if temporal:
        short = sum(1 for c in clips.values() if c.get("missing_temporal"))
        print(f"\n  temporal variants ({len(temporal)}) on "
              + "+".join(held_out) + ": " + ", ".join(temporal))
        print(f"    {n_held - short}/{n_held} held-out clips have all three; "
              f"{short} are short (masked out per variant, not excluded)")

    n = (per_split["train"] * (1 + doc["perturbations_per_train_clip"])
         + n_held * (1 + len(doc["superficial_variants"]) + len(temporal))
         + sum(per_split[s] for s in SPLITS
               if s != "train" and s not in held_out))
    print(f"\n  embeddings to compute: {n}")


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(description="build the videophy2_train split")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the split without uploading it")
    build(dry_run=ap.parse_args().dry_run)
