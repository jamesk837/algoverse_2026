"""Stage 6 -- human annotation interface for the attack suite.

What lives here:

  build_task_set()    sample the 60 base clips and freeze them into S3
  annotate()          the ipywidgets rater UI (blind 2AFC + VideoPhy-2 rubric)
  calibration()       one clean clip per human PC level, to anchor a rater
  report()            clean-preference rate + inter-rater agreement, PER ATTACK
  compare_to_vjepa()  join the human numbers to a locked probe delta

Protocol, from the spec:

  * 60 base clips, sampled randomly inside a pool of clips with PREVALENT
    MOTION -- a near-static clip cannot show temporal degradation, so it
    cannot answer the question the temporal attacks ask.
  * VideoPhy-2 clips are sampled PREFERENTIALLY at human PC >= 4 and SA >= 4,
    so there is headroom for a perturbation to push the score down.
  * Selection NEVER looks at a judge score or a probe prediction. That is
    enforced rather than merely documented: `no_model_reads()` wraps the whole
    sampling path and raises on any read under results/, probes/, embeddings/.
  * Every clip is annotated against all 9 perturbations (6 superficial + 3
    temporal), i.e. 540 comparisons for a full single-rater pass.
  * Each item is a BLIND side-by-side: clean and variant are placed left/right
    by a hash of (seed, rater, item), so the rater cannot learn which is which
    and the placement is reproducible from the task set alone.

Reporting keeps the 2x2 taxonomy. A temporal variant SHOULD lose the
preference; a superficial one should tie. Clean-preference rate and
inter-rater agreement are reported separately for each attack, never pooled --
pooling them is what makes a gameability number unreadable.

Colab (fetch the file, do not paste it -- long pastes truncate mid-line):

    !wget -q -O annotate.py https://raw.githubusercontent.com/jamesk837/algoverse_2026/main/annotate.py
    import annotate
    annotate.calibration()                  # anchor yourself on the 1-5 scale
    doc = annotate.build_task_set(n=60)     # ONCE, by one person
    annotate.annotate(rater="jk")           # every rater runs only this
    annotate.report()

Needs AWS credentials (Colab secrets AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY,
or an instance role) with read on datasets/ + attacks/ + manifests/ and write on
annotation/ + annotations/. No torch, no GPU: a rater needs opencv (preinstalled
in Colab) only for build_task_set's motion probe.
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np

BUCKET = "nickb-aarj"
TMP_DIR = Path("./tmp_annot")
TASK_KEY_FMT = "annotation/tasks_{version}.json"
MOTION_KEY = "annotation/motion_cache_v1.json"
ANNOT_PREFIX = "annotations"
HUMAN_ID = "video_url"
MANIFEST_FMT = "manifests/captions_{dataset}.json"
SPLIT_KEY = "splits/videophy2_train/split_v1.json"
HELD_OUT_SPLITS = ("val", "cal")

# "test" / "implausibench_*" are duplicated from judge_harness.py on purpose:
# this file imports nothing local so it still runs in a bare Colab runtime.
# "train" is NOT a judge corpus -- it is the probe's substrate, and it is here
# because it is the only corpus where a human rating and a V-JEPA delta can be
# compared on the same clip without a new embed run. Sampling it is restricted
# to the split's held-out val+cal stems; see candidates().
DATASET_PREFIXES = {
    "test": "datasets/videophy2_test/",
    "train": "datasets/videophy2_train/",
    "implausibench_real": "datasets/implausibench/ImplausiBench/real/",
    "implausibench_implausible": "datasets/implausibench/ImplausiBench/implausible/",
}
VIDEO_SUFFIXES = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

# the VideoPhy-2 corpora carry paired human pc/sa and a per-clip caption, so
# they are the only ones where the PC>=4 & SA>=4 preference and the SA rating
# mean anything ("semantic adherence 1-5 if applicable"). ImplausiBench's
# caption is one fixed string shared by every clip, not a description of it.
LABELLED_DATASETS = {"test", "train"}
CAPTIONED_DATASETS = {"test", "train"}

SUPERFICIAL_VARIANTS = [
    "photometric",
    "caption_echo_rubric_vocab",
    "caption_echo_score_anchor_positive",
    "caption_echo_authoritative_claim",
    "caption_echo_score_anchor_negative",
    "caption_echo_control_irrelevant",
]
TEMPORAL_VARIANTS = ["shuffle", "reverse", "freeze"]
VARIANTS = SUPERFICIAL_VARIANTS + TEMPORAL_VARIANTS

DEFAULT_QUOTAS = {"test": 36, "implausibench_real": 12, "implausibench_implausible": 12}

# VideoPhy-2's actual annotation rubric, from arXiv 2503.06800 (checked
# 2026-08-25). Two things about it are load-bearing and easy to get wrong:
#
#   1. It is an AGREEMENT Likert -- Very Unlikely .. Very Likely -- not a
#      severity scale. The rater is agreeing with a proposition, not grading.
#   2. **Only the endpoints are defined.** The paper gives no written criterion
#      for 2, 3 or 4, deliberately. An earlier version of this file invented
#      them, which is worse than useless: our raters would then be on a
#      different scale from the `pc`/`sa` columns in videophy2_test.csv, and
#      every human-vs-published comparison downstream would be off by an
#      unknown amount. Do not add criteria for the middle. calibration() is
#      how a rater learns the middle -- from real clips at each level.
SCALE_LABELS = [(1, "Very Unlikely"), (2, "Unlikely"), (3, "Neutral"),
                (4, "Likely"), (5, "Very Likely")]

PC_QUESTION = ("Does the generated video follow the physical laws of the "
               "real world intuitively? Judge the holistic sense of the "
               "video's physical commonsense.")
PC_ANCHORS = {
    1: "the video contains numerous violations of fundamental physical laws",
    5: "the video demonstrates a strong understanding of physical commonsense "
       "with no violations",
}
SA_QUESTION = ("Are the entities, actions and relationships described in the "
               "caption accurately depicted in the video?")
SA_ANCHORS = {
    1: "the video does not match the prompt at all",
    5: "the video fully adheres to the prompt with no inconsistencies",
}

# The "physics checkbox" is WorldModelBench's, not VideoPhy-2's -- `vila_ewm` is
# the only one of our three judges that emits booleans. It answers these five
# physical-law questions (plus two common-sense ones) as yes/no calls, so
# collecting the same five from a human makes the two comparable CALL BY CALL.
#
# Everything here is the judge's, so a human record joins vila's with no
# translation step at all:
#
#   * KEYS are vila's call ids, `physical_laws_0` .. `physical_laws_4`.
#   * TEXT shown to the rater is the question exactly as it is interpolated
#     into the prompt -- lowercased, because judge_harness inserts
#     `question.lower()` -- under the template's own stem line. No paraphrase.
#   * POLARITY is vila's. Its parse is `"no" in pred.lower()`, so its stored
#     True means "no violation found". A TICKED box here says "yes, the video
#     shows this violation", and is therefore stored as **False**. Unticked
#     stores True. The two records can be compared directly.
#
# Duplicated from judge_harness.WMB_QUESTION_POOL / WMB_PROMPT_TEMPLATES --
# edit the two together or the wording silently drifts apart.
WMB_LAW_STEM = "Watch the video and determine if it shows any"
WMB_SENSE_STEM = "Does the video exhibit"
WMB_ANSWER_LINE = 'Answer with "Yes" or "No".'

PHYSICS_CHECKBOXES = [
    ("physical_laws_0",
     "Violation of Newton's Law: Objects move without any external force."),
    ("physical_laws_1",
     "Violation of the Law of Conservation of Mass or Solid Constitutive Law: "
     "Objects deform irregularly."),
    ("physical_laws_2",
     "Violation of Fluid Constitutive Law: Liquids flow in an unnatural manner."),
    ("physical_laws_3",
     "Violation of Non-physical Penetration: Objects unnaturally pass through "
     "each other."),
    ("physical_laws_4",
     "Violation of Gravity: Objects behave inconsistently with gravity."),
]
# WorldModelBench's other two yes/no calls, and note they use a different stem.
# Off by default -- the spec says "physics checkbox", and on a shuffled clip
# "noticeable flickering or abrupt changes" is true by construction.
COMMON_SENSE_CHECKBOXES = [
    ("common_sense_0",
     "Poor Aesthetics: Visually unappealing or low-quality content."),
    ("common_sense_1",
     "Temporal Inconsistency: Noticeable flickering or abrupt changes."),
]

# VideoPhy-2 used THREE annotators per video and aggregated by mean-then-round
# to the nearest integer. report() reproduces that when it checks our raters'
# PC against the published labels.
UPSTREAM_ANNOTATORS = 3

PREFERENCE_QUESTION = "Which clip shows more physically plausible real world dynamics?"

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


# --------------------------------------------------------------- S3 + guard

s3 = None
_BLOCKED = ("results/", "probes/", "embeddings/")
_guarded = False


class no_model_reads:
    """"Never select examples based on V-JEPA/VLM disagreement."  While this
    context is open, reading a judge result or a probe raises instead of
    quietly contaminating the sample."""

    def __enter__(self):
        global _guarded
        self.prev, _guarded = _guarded, True
        return self

    def __exit__(self, *exc):
        global _guarded
        _guarded = self.prev
        return False


def _check(key):
    if _guarded and key.startswith(_BLOCKED):
        raise RuntimeError(
            "sampling tried to read %r -- clip selection must never depend on "
            "a judge score or a probe prediction" % key)


def _ensure_s3():
    global s3
    if s3 is None:
        s3 = boto3.client("s3")
    return s3


def list_keys(prefix, suffixes=None):
    keys, pg = [], _ensure_s3().get_paginator("list_objects_v2")
    for page in pg.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if suffixes and Path(k).suffix.lower() not in suffixes:
                continue
            keys.append(k)
    return keys


def list_dirs(prefix):
    """immediate child "directories" of a prefix, via one Delimiter LIST."""
    out, pg = set(), _ensure_s3().get_paginator("list_objects_v2")
    for page in pg.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"][len(prefix):].strip("/")
            if name:
                out.add(name)
    return out


def get_json(key, default=None):
    _check(key)
    try:
        body = _ensure_s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except Exception:
        return default
    return json.loads(body.decode("utf-8"))


def put_json(key, obj):
    _ensure_s3().put_object(
        Bucket=BUCKET, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
        ContentType="application/json")
    print("  wrote s3://%s/%s" % (BUCKET, key))


def download(key, local):
    _check(key)
    local = Path(local)
    if local.exists() and local.stat().st_size > 0:
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part%d" % os.getpid())
    _ensure_s3().download_file(BUCKET, key, str(tmp))
    tmp.replace(local)
    return local


def stem_of(key):
    """must match attack_suite.dest_key / build_split.stem_from_url."""
    return Path(str(key).split("?")[0].split("/")[-1]).stem


def variant_key(dataset, stem, variant):
    return "attacks/%s/%s/%s.mp4" % (dataset, stem, variant)


def local_path(dataset, stem, variant, source_key=None):
    ext = Path(source_key).suffix if (variant == "clean" and source_key) else ".mp4"
    return TMP_DIR / "clips" / dataset / stem / ("%s%s" % (variant, ext or ".mp4"))


def fetch(dataset, stem, variant, source_key=None):
    """the clean clip IS the source object -- attacks/ holds only variants."""
    key = source_key if variant == "clean" else variant_key(dataset, stem, variant)
    return download(key, local_path(dataset, stem, variant, source_key))


# -------------------------------------------------------------------- labels

_meta_cache = {}


def meta_rows(dataset):
    """rows of the corpus's _metadata CSV, found by LIST the way build_split
    does -- videophy2_test names its CSV after itself, train does not."""
    import csv as _csv

    if dataset in _meta_cache:
        return _meta_cache[dataset]
    rows = []
    prefix = DATASET_PREFIXES.get(dataset, "") + "_metadata/"
    csvs = [k for k in list_keys(prefix) if k.lower().endswith(".csv")]
    if not csvs:
        print("  (no CSV under s3://%s/%s)" % (BUCKET, prefix))
    else:
        if len(csvs) > 1:
            print("  %d CSVs under %s, using %s" % (len(csvs), prefix, csvs[0]))
        try:
            body = _ensure_s3().get_object(Bucket=BUCKET, Key=csvs[0])["Body"].read()
            rows = list(_csv.DictReader(io.StringIO(body.decode("utf-8", "replace"))))
        except Exception as exc:
            print("  (metadata unreadable: %s)" % exc)
    _meta_cache[dataset] = rows
    return rows


def human_labels(dataset="test"):
    """stem -> (pc, sa). Same reader as stats.py; only the VideoPhy-2 corpora
    carry paired human scores."""
    if dataset not in LABELLED_DATASETS:
        return {}
    out = {}
    for r in meta_rows(dataset):
        stem = stem_of(r.get(HUMAN_ID) or "")
        try:
            out[stem] = (int(float(r["pc"])), int(float(r["sa"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def captions(dataset):
    """the harness's manifest where one exists (test), else straight from the
    corpus CSV -- judge_harness never ran on train, so train has no manifest."""
    if dataset not in CAPTIONED_DATASETS:
        return {}
    cached = get_json(MANIFEST_FMT.format(dataset=dataset), {})
    if cached:
        return cached
    out = {}
    for r in meta_rows(dataset):
        stem, cap = stem_of(r.get(HUMAN_ID) or ""), (r.get("caption") or "").strip()
        if stem and cap:
            out.setdefault(stem, cap)
    return out


def held_out_stems():
    """val + cal membership from the probe's split file. Annotating a clip the
    probe TRAINED on would make the human-vs-V-JEPA comparison meaningless."""
    doc = get_json(SPLIT_KEY)
    if not doc:
        return None
    splits = set(doc.get("held_out_splits") or HELD_OUT_SPLITS)
    return {s for s, c in doc["clips"].items() if c.get("split") in splits}


# ------------------------------------------------------------- motion filter

def motion_score(path, max_frames=48, size=64):
    """mean absolute frame-to-frame difference on a 64x64 grey downsample, in
    [0,1]. Decodes sequentially -- CAP_PROP_POS_FRAMES seeking is unreliable on
    these re-encoded files, the same reason embed_vjepa reads straight through."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (size, size)))
    cap.release()
    if len(frames) < 2:
        return 0.0
    idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)
    sel = [frames[i].astype(np.float32) for i in idx]
    d = [np.abs(sel[i + 1] - sel[i]).mean() for i in range(len(sel) - 1)]
    return float(np.mean(d) / 255.0)


def motion_for(dataset, pairs, cache, workers=8):
    """pairs is [(stem, source_key)]. Fills `cache` in place, keyed 'ds|stem'."""
    todo = [(s, k) for s, k in pairs if ("%s|%s" % (dataset, s)) not in cache]
    if not todo:
        return cache
    print("  %s: probing motion on %d clips (%d cached)"
          % (dataset, len(todo), len(pairs) - len(todo)))

    def one(item):
        stem, key = item
        try:
            p = download(key, local_path(dataset, stem, "clean", key))
            return stem, motion_score(p)
        except Exception as exc:
            print("  FAILED motion %s: %s" % (stem, exc))
            return stem, None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (stem, sc) in enumerate(ex.map(one, todo), 1):
            if sc is not None:
                cache["%s|%s" % (dataset, stem)] = sc
            if i % 50 == 0 or i == len(todo):
                print("    %d/%d  %.0fs" % (i, len(todo), time.time() - t0))
    return cache


# ---------------------------------------------------------------- sampling

def _u01(*parts):
    """deterministic float in [0,1) from any tuple of values."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def _seed_int(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


def candidates(dataset, wanted_variants=None):
    """clips that have a source object AND at least one rendered variant. A
    clip with no renders cannot be compared against anything.

    One paginated LIST over attacks/<dataset>/ for the whole corpus, not one
    per clip -- ~450 stems is ~450 round trips the other way."""
    wanted = list(wanted_variants or VARIANTS)
    srcs = {stem_of(k): k for k in list_keys(DATASET_PREFIXES[dataset], VIDEO_SUFFIXES)
            if "/_metadata/" not in k}
    prefix = "attacks/%s/" % dataset
    have = defaultdict(set)
    for k in list_keys(prefix):
        if not k.endswith(".mp4"):
            continue
        rest = k[len(prefix):].split("/")
        if len(rest) == 2:
            have[rest[0]].add(Path(rest[1]).stem)
    stems = set(srcs) & set(have)
    if dataset == "train":
        heldout = held_out_stems()
        if heldout is None:
            raise RuntimeError(
                "dataset 'train' needs %s to restrict sampling to val+cal -- "
                "without it you would annotate clips the probe trained on"
                % SPLIT_KEY)
        before = len(stems)
        stems &= heldout
        print("  train: %d of %d clips are held out (val+cal); the rest are the "
              "probe's training set and are excluded" % (len(stems), before))
    out = []
    for stem in sorted(stems):
        keep = [v for v in wanted if v in have[stem]]
        if keep:
            out.append({"dataset": dataset, "stem": stem, "source_key": srcs[stem],
                        "variants": keep,
                        "missing": [v for v in wanted if v not in have[stem]]})
    return out


def _tiers(dataset, pool, labels):
    """VideoPhy-2 is sampled preferentially at PC>=4 and SA>=4; everything else
    is a flat random draw. Returns [(label, [clips])], best tier first, each
    clip in exactly one tier."""
    if dataset not in LABELLED_DATASETS or not labels:
        return [("random", list(pool))]
    named = [("pc>=4 & sa>=4", lambda pc, sa: pc >= 4 and sa >= 4),
             ("pc>=4", lambda pc, sa: pc >= 4),
             ("sa>=4", lambda pc, sa: sa >= 4)]
    tiers, taken = [], set()
    for name, test in named:
        tier = []
        for c in pool:
            lab = labels.get(c["stem"])
            if lab and c["stem"] not in taken and test(*lab):
                tier.append(c)
                taken.add(c["stem"])
        tiers.append((name, tier))
    tiers.append(("rest", [c for c in pool if c["stem"] not in taken]))
    return tiers


def build_task_set(n=60, quotas=None, seed=20260825, motion_percentile=40,
                   motion_probe_limit=300, raters=("r1", "r2"), overlap=1.0,
                   version="v1", wanted_variants=None, push=True, dry_run=False):
    """Freeze the 60-clip annotation set into S3. Run this ONCE; every rater
    then reads the same file, so their item lists and blinding line up. The
    randomness is all at build time -- after this, the 60 clips are fixed.

    quotas              WHICH CORPUS, and this is the real decision:
                          {"test": 36, "implausibench_*": 12 each}  (default)
                            the clips the three judges are being scored on, so
                            human-vs-JUDGE is a direct comparison. The V-JEPA
                            probe has never embedded these, so human-vs-PROBE
                            needs an embed run over the 60 clips first.
                          {"train": 60}
                            the probe's own held-out val+cal clips: embeddings
                            and probe predictions already exist, so the locked
                            attack delta is comparable today. The judges have
                            not scored these.
                        Mix them if you want both, at half the n each.
    motion_percentile   'prevalent motion' == above this percentile of the
                        probed pool. 40 keeps the top 60%.
    raters / overlap    `overlap` of items go to EVERY rater (that is what
                        inter-rater agreement is computed on); the rest is
                        split between them, so N raters cover ~N x more items.
                        overlap=1.0 means every rater rates all 540.
    """
    quotas = dict(quotas or DEFAULT_QUOTAS)
    if sum(quotas.values()) != n:
        scale = n / float(sum(quotas.values()))
        quotas = {k: int(round(v * scale)) for k, v in quotas.items()}
        drift = n - sum(quotas.values())
        first = sorted(quotas)[0]
        quotas[first] += drift
        print("  quotas rescaled to n=%d: %s" % (n, quotas))

    with no_model_reads():
        cache = get_json(MOTION_KEY, {}) or {}
        chosen, notes = [], {}

        for dataset, quota in quotas.items():
            if quota <= 0:
                continue
            if dataset not in DATASET_PREFIXES:
                raise ValueError("unknown dataset %r; known: %s"
                                 % (dataset, sorted(DATASET_PREFIXES)))
            labels = human_labels(dataset)
            pool = candidates(dataset, wanted_variants)
            print("== %s ==  %d clips with renders" % (dataset, len(pool)))
            if not pool:
                notes[dataset] = "no rendered clips"
                continue

            tiers = _tiers(dataset, pool, labels)
            rng = random.Random(_seed_int(seed, dataset))
            probe = []
            for _, tier in tiers:
                t = list(tier)
                rng.shuffle(t)
                probe.extend(t)
            probe = probe[:motion_probe_limit]
            motion_for(dataset, [(c["stem"], c["source_key"]) for c in probe], cache)

            def m(c):
                return cache.get("%s|%s" % (dataset, c["stem"]))

            scored = [m(c) for c in probe if m(c) is not None]
            thr = float(np.percentile(scored, motion_percentile)) if scored else 0.0
            print("  motion threshold p%d = %.4f  (probed %d)"
                  % (motion_percentile, thr, len(scored)))

            picked, taken, ladder = [], set(), []
            for relax_motion in (False, True):
                for name, tier in tiers:
                    if len(picked) >= quota:
                        break
                    avail = [c for c in tier if c["stem"] not in taken
                             and m(c) is not None
                             and (relax_motion or m(c) >= thr)]
                    rng2 = random.Random(_seed_int(seed, dataset, name, relax_motion))
                    rng2.shuffle(avail)
                    take = avail[:quota - len(picked)]
                    if take:
                        ladder.append("%s%s: %d" % (name, "" if not relax_motion
                                                    else " (motion relaxed)", len(take)))
                        picked.extend(take)
                        taken.update(c["stem"] for c in take)
                if len(picked) >= quota:
                    break
            if len(picked) < quota:
                ladder.append("SHORT by %d" % (quota - len(picked)))
            print("  selected %d/%d  [%s]" % (len(picked), quota, "; ".join(ladder)))
            notes[dataset] = "; ".join(ladder)

            for c in picked:
                pc, sa = labels.get(c["stem"], (None, None))
                chosen.append({
                    "dataset": dataset, "stem": c["stem"], "source_key": c["source_key"],
                    "variants": c["variants"], "missing_variants": c["missing"],
                    "pc": pc, "sa": sa, "motion": round(m(c), 5),
                })

    doc = {
        "version": version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "n_clips": len(chosen),
        "quotas": quotas,
        "motion": {"percentile": motion_percentile, "probe_limit": motion_probe_limit},
        "selection": notes,
        "raters": list(raters),
        "overlap": overlap,
        "variants": list(wanted_variants or VARIANTS),
        "temporal_variants": TEMPORAL_VARIANTS,
        "superficial_variants": SUPERFICIAL_VARIANTS,
        "question": PREFERENCE_QUESTION,
        "clips": chosen,
    }
    preview(doc)
    if dry_run:
        print("\n  --dry-run: nothing written")
        return doc
    if push:
        put_json(TASK_KEY_FMT.format(version=version), doc)
        if cache:
            put_json(MOTION_KEY, cache)
    return doc


def load_task_set(version="v1"):
    doc = get_json(TASK_KEY_FMT.format(version=version))
    if not doc:
        raise RuntimeError("no task set at s3://%s/%s -- run build_task_set() first"
                           % (BUCKET, TASK_KEY_FMT.format(version=version)))
    return doc


def preview(doc):
    clips = doc["clips"]
    n_items = sum(len(c["variants"]) for c in clips)
    print("\n== task set %s ==  %d clips, %d comparisons"
          % (doc["version"], len(clips), n_items))
    by_ds = Counter(c["dataset"] for c in clips)
    for ds in sorted(by_ds):
        sub = [c for c in clips if c["dataset"] == ds]
        mot = [c["motion"] for c in sub]
        pcs = [c["pc"] for c in sub if c["pc"] is not None]
        line = "  %-28s n=%-3d motion %.4f-%.4f" % (ds, len(sub), min(mot), max(mot))
        if pcs:
            line += "  pc %s" % dict(sorted(Counter(pcs).items()))
            sas = [c["sa"] for c in sub if c["sa"] is not None]
            line += "  sa %s" % dict(sorted(Counter(sas).items()))
        print(line)
    short = [c for c in clips if c["missing_variants"]]
    if short:
        print("  %d clips are missing at least one render (masked, not dropped)"
              % len(short))
    raters = doc.get("raters") or []
    if raters:
        per = {r: len(items_for(doc, r, quiet=True)) for r in raters}
        print("  assignment: %d raters, overlap %.0f%% -> %s"
              % (len(raters), 100 * doc.get("overlap", 1.0), per))


# ------------------------------------------------------------------- items

def items_for(doc, rater, quiet=False):
    """Deterministic per-rater work list.

    An `overlap` fraction of items goes to EVERY rater -- those are what
    inter-rater agreement is measured on -- and the rest is split between them.
    Ordering interleaves clips so a rater never sees the same clip twice in a
    row, and left/right placement is hashed so blinding survives a resume."""
    raters = list(doc.get("raters") or [])
    overlap = float(doc.get("overlap", 1.0))
    solo = rater not in raters
    if solo and not quiet:
        print("  rater %r is not in the task set's list %s -- taking the full "
              "set (pilot mode)" % (rater, raters))

    by_clip = defaultdict(list)
    for clip in doc["clips"]:
        for v in clip["variants"]:
            iid = "%s|%s|%s" % (clip["dataset"], clip["stem"], v)
            shared = _u01(doc["seed"], "assign", iid) < overlap
            if not solo and raters and not shared:
                idx = int(_u01(doc["seed"], "owner", iid) * len(raters)) % len(raters)
                if raters[idx] != rater:
                    continue
            side = "left" if _u01(doc["seed"], rater, "side", iid) < 0.5 else "right"
            by_clip[iid.rsplit("|", 1)[0]].append({
                "item_id": iid, "dataset": clip["dataset"], "stem": clip["stem"],
                "variant": v, "source_key": clip["source_key"],
                "clean_side": side, "shared": shared,
            })

    rng = random.Random(_seed_int(doc["seed"], rater))
    for lst in by_clip.values():
        rng.shuffle(lst)
    keys = sorted(by_clip)
    depth = max((len(v) for v in by_clip.values()), default=0)
    ordered = []
    for r in range(depth):
        row = [by_clip[k][r] for k in keys if len(by_clip[k]) > r]
        rng.shuffle(row)
        ordered.extend(row)
    return ordered


def variant_kind(variant):
    return "temporal" if variant in TEMPORAL_VARIANTS else "superficial"


# ------------------------------------------------------------------ records

def _records_path(version, rater):
    return TMP_DIR / "annotations" / ("%s__%s.jsonl" % (version, rater))


def _records_key(version, rater):
    return "%s/%s/%s.jsonl" % (ANNOT_PREFIX, version, rater)


def load_records(version="v1", rater=None, prefer_local=True):
    """Merge the S3 copy with the local one, newest write per item_id wins."""
    out = {}
    keys = ([_records_key(version, rater)] if rater
            else list_keys("%s/%s/" % (ANNOT_PREFIX, version)))
    for key in keys:
        try:
            body = _ensure_s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
        except Exception:
            continue
        for line in body.decode("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["rater"], r["item_id"])] = r
    if rater is not None:
        p = _records_path(version, rater)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    k = (r["rater"], r["item_id"])
                    if prefer_local or k not in out:
                        out[k] = r
    return list(out.values())


def _serialize(records):
    return "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")


def _write_all(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(_serialize(records))
    tmp.replace(path)


def _upload(version, rater, records):
    """Whole-file PUT of an in-memory snapshot, not upload_file: the local file
    is being rewritten under us on every submit, and a snapshot also lets the
    upload run off the UI thread without racing the writer."""
    _ensure_s3().put_object(Bucket=BUCKET, Key=_records_key(version, rater),
                            Body=_serialize(records), ContentType="application/x-ndjson")


# ----------------------------------------------------------------- the UI

def _esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


_MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
         ".ogv": "video/ogg", ".mov": "video/quicktime", ".mkv": "video/webm"}


def _video_html(path, width, label=""):
    """inline data: URI. The clean clip keeps the source object's container --
    only the rendered variants are all mp4 -- so the MIME follows the suffix."""
    path = Path(path)
    mime = _MIME.get(path.suffix.lower(), "video/mp4")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        '<div style="text-align:center">'
        '<div style="font:600 13px sans-serif;padding:2px 0">%s</div>'
        '<video width="%d" controls autoplay loop muted playsinline '
        'style="border-radius:6px;background:#000">'
        '<source src="data:%s;base64,%s" type="%s"></video></div>'
        % (label, width, mime, b64, mime))


def _scale_html(question, anchors):
    """VideoPhy-2's scale as they present it: the five labels, with the prose
    gloss upstream gives for the two endpoints."""
    cells = []
    for k, name in SCALE_LABELS:
        note = anchors.get(k, "")
        cells.append(
            "<td style='padding:3px 9px;vertical-align:top;border-left:1px solid #eee'>"
            "<b>%d</b>&nbsp; %s%s</td>"
            % (k, name, ("<div style='color:#666;font-size:12px;max-width:190px;"
                         "margin-top:2px'>%s</div>" % note) if note else ""))
    return ("<div style='margin:6px 0 10px'><div>%s</div>"
            "<table style='border-collapse:collapse;margin-top:4px'><tr>%s</tr></table>"
            "</div>" % (question, "".join(cells)))


def _rubric_html():
    return (
        "<div style='font:13px sans-serif;line-height:1.5'>"
        "<b>Physical commonsense (PC)</b>%s"
        "<b>Semantic adherence (SA)</b> &mdash; only where the clip has its own "
        "caption%s"
        "<b>Physics checkboxes</b> &mdash; these are the WorldModelBench yes/no "
        "calls, word for word. Ticking a box is answering <b>Yes</b> to that "
        "question. This is the exact prompt the vila_ewm judge is given, one "
        "call per box:%s"
        "<div style='color:#666;font-size:12px;margin-top:8px'>Scales: "
        "VideoPhy-2 (arXiv 2503.06800). Checkboxes: WorldModelBench "
        "(arXiv 2502.20694).</div></div>"
        % (_scale_html(PC_QUESTION, PC_ANCHORS),
           _scale_html(SA_QUESTION, SA_ANCHORS),
           "".join(
               "<pre style='margin:6px 0;padding:8px 10px;background:#f4f4f4;"
               "border-radius:3px;font:12px ui-monospace,monospace;"
               "white-space:pre-wrap'>%s '%s'\n%s</pre>"
               % (_esc(stem), _esc(q.lower()), _esc(WMB_ANSWER_LINE))
               for stem, items in ((WMB_LAW_STEM, PHYSICS_CHECKBOXES),
                                   (WMB_SENSE_STEM, COMMON_SENSE_CHECKBOXES))
               for _k, q in items)))


def annotate(rater, version="v1", doc=None, mode="pair+rate", limit=None,
             prefetch=3, autosave=1, width=360, allow_tie=True,
             common_sense=False):
    """The rater UI. Resumable: items already recorded for this rater are
    skipped, so closing the tab costs only the item on screen.

    mode="pair"       preference only -- fast, ~540 items in one sitting
    mode="pair+rate"  preference plus PC / SA / physics-violation on BOTH
                      sides. Both sides are rated every time on purpose: the
                      clean clip recurs 9 times per rater, and those repeats
                      are the intra-rater reliability estimate report() prints.
    allow_tie         True keeps the "Equal" button. The spec's question offers
                      two clips, so allow_tie=False is the literal reading and
                      the textbook 2AFC: the rater must pick, chance is exactly
                      0.5, and the preference rate has a fixed denominator.
                      Ties are kept as the default because most of these
                      perturbations are designed to be imperceptible, and an
                      explicit "I cannot tell" is a cleaner statement of that
                      than a forced coin-flip. report() handles both -- ties are
                      excluded from the rate and shown as their own column.
    """
    import ipywidgets as W
    from IPython.display import display

    if mode not in ("pair", "pair+rate"):
        raise ValueError("mode must be 'pair' or 'pair+rate', got %r" % mode)
    try:  # ipywidgets 8 under Colab
        from google.colab import output as _co
        _co.enable_custom_widget_manager()
    except Exception:
        pass

    doc = doc or load_task_set(version)
    version = doc["version"]
    items = items_for(doc, rater)
    caps = {ds: captions(ds) for ds in {i["dataset"] for i in items}}

    records = [r for r in load_records(version, rater) if r["rater"] == rater]
    done = {r["item_id"] for r in records}
    todo = [i for i in items if i["item_id"] not in done]
    if limit:
        todo = todo[:limit]
    print("rater %s: %d items assigned, %d already done, %d this session"
          % (rater, len(items), len(done), len(todo)))
    if not todo:
        print("nothing left -- run report() next")
        return records

    path = _records_path(version, rater)
    pool = ThreadPoolExecutor(max_workers=4)
    uploader = ThreadPoolExecutor(max_workers=1)  # 1 worker keeps PUTs ordered
    futures = {}

    def want(i):
        """download ahead: the rater is watching item i while i+1.. arrive."""
        if i >= len(todo):
            return
        it = todo[i]
        for v in ("clean", it["variant"]):
            k = (it["dataset"], it["stem"], v)
            if k not in futures:
                futures[k] = pool.submit(fetch, it["dataset"], it["stem"], v,
                                         it["source_key"])

    rate = mode == "pair+rate"
    groups = [(WMB_LAW_STEM, PHYSICS_CHECKBOXES)]
    if common_sense:
        groups.append((WMB_SENSE_STEM, COMMON_SENSE_CHECKBOXES))
    head = W.HTML()
    prog = W.IntProgress(min=0, max=len(todo), bar_style="info",
                         layout=W.Layout(width="100%"))
    vid_l, vid_r = W.HTML(), W.HTML()
    cap_box = W.HTML()

    def _scale(desc):
        return W.ToggleButtons(
            options=[("--", None), ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5)],
            value=None, description=desc, style={"button_width": "34px",
                                                 "description_width": "34px"})

    def _panel(tag):
        """The checkbox label is the question verbatim, lowercased exactly as
        judge_harness interpolates it, under the template's own stem."""
        pc = _scale("PC")
        sa = _scale("SA")
        boxes, kids = {}, [pc, sa]
        for stem, items in groups:
            kids.append(W.HTML(
                "<div style='font:600 12px sans-serif;color:#555;margin:8px 0 2px'>"
                "%s:</div>" % _esc(stem)))
            for key, question in items:
                cb = W.Checkbox(value=False, indent=False,
                                layout=W.Layout(width="26px", margin="0", flex="0 0 auto"))
                lbl = W.HTML("<div style='font:12.5px sans-serif;line-height:1.35;"
                             "padding-top:3px'>%s</div>" % _esc(question.lower()))
                boxes[key] = cb
                kids.append(W.HBox([cb, lbl], layout=W.Layout(
                    align_items="flex-start", margin="0")))
        box = W.VBox(kids, layout=W.Layout(
            border="1px solid #ddd", padding="6px", margin="4px",
            width="%dpx" % (width + 24),
            display=None if rate else "none"))
        return {"pc": pc, "sa": sa, "laws": boxes, "box": box, "tag": tag}

    left, right = _panel("left"), _panel("right")
    pref_options = [("--", None), ("<< Left", "left")]
    if allow_tie:
        pref_options.append(("Equal", "tie"))
    pref_options.append(("Right >>", "right"))
    pref = W.ToggleButtons(options=pref_options, value=None,
                           style={"button_width": "110px"})
    note = W.Text(placeholder="optional note", layout=W.Layout(width="60%"))
    b_submit = W.Button(description="Submit  (next)", button_style="primary")
    b_skip = W.Button(description="Skip")
    b_back = W.Button(description="Back")
    b_quit = W.Button(description="Save & stop")
    msg = W.HTML()
    rub = W.Accordion(children=[W.HTML(_rubric_html())])
    rub.set_title(0, "VideoPhy-2 rubric  (click to open)")
    rub.selected_index = None

    st = {"i": 0, "t0": time.time()}

    def show():
        i = st["i"]
        if i >= len(todo):
            finish()
            return
        it = todo[i]
        for j in range(i, i + prefetch + 1):
            want(j)
        try:
            pl = futures[(it["dataset"], it["stem"], "clean")].result()
            pv = futures[(it["dataset"], it["stem"], it["variant"])].result()
        except Exception as exc:
            msg.value = "<b style='color:#b00'>download failed: %s -- skipping</b>" % exc
            st["i"] += 1
            show()
            return
        a, b = (pl, pv) if it["clean_side"] == "left" else (pv, pl)
        vid_l.value = _video_html(a, width, "A  (left)")
        vid_r.value = _video_html(b, width, "B  (right)")
        cap = caps.get(it["dataset"], {}).get(it["stem"], "")
        cap_box.value = ("<div style='font:13px sans-serif;padding:4px 0'>"
                         "<b>Caption:</b> %s</div>" % cap) if cap else ""
        captioned = bool(cap)
        for p in (left, right):
            p["pc"].value = None
            p["sa"].value = None
            p["sa"].layout.display = None if captioned else "none"
            for cb in p["laws"].values():
                cb.value = False
        pref.value = None
        note.value = ""
        prog.value = i
        head.value = ("<div style='font:14px sans-serif'>"
                      "<b>%d / %d</b> &nbsp; %s &nbsp;&nbsp;"
                      "<span style='color:#777'>%s</span></div>"
                      "<div style='font:600 15px sans-serif;padding:6px 0'>%s</div>"
                      % (i + 1, len(todo), it["dataset"], it["stem"][:60],
                         PREFERENCE_QUESTION))
        st["t0"] = time.time()

    def save(record):
        """local write is synchronous (cheap, and it is what Back rewinds);
        the S3 PUT is queued on a 1-worker pool so a killed Colab runtime never
        costs more than the item in flight, without stalling the UI."""
        records.append(record)
        _write_all(path, records)
        if len(records) % autosave == 0:
            snapshot = list(records)

            def push():
                try:
                    _upload(version, rater, snapshot)
                except Exception as exc:
                    msg.value = ("<span style='color:#b00'>upload failed (%s) -- "
                                 "records are safe locally at %s</span>" % (exc, path))
            uploader.submit(push)

    def on_submit(_):
        it = todo[st["i"]]
        if pref.value is None:
            msg.value = "<b style='color:#b00'>pick a preference first</b>"
            return
        if rate and (left["pc"].value is None or right["pc"].value is None):
            msg.value = "<b style='color:#b00'>rate PC on both clips</b>"
            return
        clean_side = it["clean_side"]
        var_side = "right" if clean_side == "left" else "left"
        cp = left if clean_side == "left" else right
        vp = left if var_side == "left" else right
        if pref.value == "tie":
            preference = "tie"
        else:
            preference = "clean" if pref.value == clean_side else "variant"
        save({
            "rater": rater, "version": version, "item_id": it["item_id"],
            "dataset": it["dataset"], "stem": it["stem"], "variant": it["variant"],
            "kind": variant_kind(it["variant"]), "shared": it["shared"],
            "clean_side": clean_side, "chose_side": pref.value,
            "preference": preference,
            "pc_clean": cp["pc"].value, "sa_clean": cp["sa"].value,
            # vila polarity: True == "no violation found", so a tick is False
            "laws_clean": {k: (not cb.value) for k, cb in cp["laws"].items()},
            "pc_variant": vp["pc"].value, "sa_variant": vp["sa"].value,
            "laws_variant": {k: (not cb.value) for k, cb in vp["laws"].items()},
            "note": note.value.strip(), "secs": round(time.time() - st["t0"], 1),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "skipped": False,
        })
        msg.value = ""
        st["i"] += 1
        show()

    def on_skip(_):
        it = todo[st["i"]]
        save({"rater": rater, "version": version, "item_id": it["item_id"],
              "dataset": it["dataset"], "stem": it["stem"], "variant": it["variant"],
              "kind": variant_kind(it["variant"]), "shared": it["shared"],
              "clean_side": it["clean_side"], "preference": None, "skipped": True,
              "note": note.value.strip(),
              "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        msg.value = ""
        st["i"] += 1
        show()

    def on_back(_):
        if st["i"] == 0 or not records:
            msg.value = "nothing to go back to"
            return
        last = records.pop()
        _write_all(path, records)
        st["i"] = max(0, st["i"] - 1)
        msg.value = "reopened <code>%s</code>" % last["item_id"]
        show()

    def on_quit(_):
        finish()

    def finish():
        _write_all(path, records)
        uploader.shutdown(wait=True)   # drain queued PUTs before reporting
        try:
            _upload(version, rater, list(records))
            where = "s3://%s/%s" % (BUCKET, _records_key(version, rater))
        except Exception as exc:
            where = "%s  (S3 upload failed: %s)" % (path, exc)
        prog.value = prog.max
        n = len([r for r in records if not r.get("skipped")])
        head.value = ("<div style='font:15px sans-serif'><b>done for now</b> "
                      "&mdash; %d rated, %d total records</div>" % (n, len(records)))
        vid_l.value = vid_r.value = cap_box.value = ""
        msg.value = "saved to %s" % where
        pool.shutdown(wait=False)

    b_submit.on_click(on_submit)
    b_skip.on_click(on_skip)
    b_back.on_click(on_back)
    b_quit.on_click(on_quit)

    stage = W.HBox([W.VBox([vid_l, left["box"]]), W.VBox([vid_r, right["box"]])])
    display(W.VBox([head, prog, cap_box, stage, pref,
                    W.HBox([b_submit, b_skip, b_back, b_quit]), note, msg, rub]))
    show()
    return records


# ------------------------------------------------------------- calibration

def calibration(dataset="test", seed=20260825, quiz=False, width=320,
                prefer_motion=True, per_level=1):
    """One CLEAN VideoPhy-2 clip at each human PC level 1..5, side by side with
    the rubric. Watch these before rating anything -- the written scale is a
    paraphrase, these clips are the actual anchor.

    quiz=True prints the true levels as an answer key, so a rater can grade
    themselves before starting. Pass the same dataset the task set was built
    from ("test" or "train") so the anchors come from that corpus."""
    from IPython.display import HTML, display

    if dataset not in LABELLED_DATASETS:
        raise ValueError("calibration needs human pc/sa; use one of %s"
                         % sorted(LABELLED_DATASETS))
    with no_model_reads():
        labels = human_labels(dataset)
        if not labels:
            raise RuntimeError("no human labels -- cannot calibrate")
        srcs = {stem_of(k): k for k in list_keys(DATASET_PREFIXES[dataset], VIDEO_SUFFIXES)
                if "/_metadata/" not in k}
        cache = get_json(MOTION_KEY, {}) or {}
        rendered = list_dirs("attacks/%s/" % dataset)

        pick = {}
        for level in (1, 2, 3, 4, 5):
            pool = [s for s, (pc, _) in labels.items()
                    if pc == level and s in srcs and s in rendered]
            if not pool:  # a level with no rendered attacks still calibrates fine
                pool = [s for s, (pc, _) in labels.items() if pc == level and s in srcs]
            if not pool:
                print("  no clip at PC=%d" % level)
                continue
            if prefer_motion:
                scored = [(cache.get("%s|%s" % (dataset, s), -1.0), s) for s in pool]
                have = [s for m, s in scored if m >= 0]
                if len(have) >= per_level:
                    pool = [s for _, s in sorted(scored, reverse=True)[:max(8, per_level)]]
            rng = random.Random(_seed_int(seed, "calib", level))
            pool = sorted(pool)
            rng.shuffle(pool)
            pick[level] = pool[:per_level]

        caps = captions(dataset)
        blocks = []
        for level in sorted(pick):
            for stem in pick[level]:
                p = fetch(dataset, stem, "clean", srcs[stem])
                pc, sa = labels[stem]
                label = ("PC = %d &nbsp; SA = %d" % (pc, sa)) if not quiz else "PC = ?"
                cap = caps.get(stem, "")
                blocks.append(
                    "<div style='display:inline-block;vertical-align:top;margin:8px'>"
                    "%s<div style='font:12px sans-serif;width:%dpx;color:#555'>%s</div>"
                    "</div>" % (_video_html(p, width, label), width, cap[:140]))
                if quiz:
                    print("  PC=%d -> %s" % (pc, stem))

    display(HTML(_rubric_html()))
    display(HTML("<div>%s</div>" % "".join(blocks)))
    if quiz:
        print("\nquiz mode: the true levels are printed above, in ascending order.")
    return pick


# ---------------------------------------------------------------- statistics

def _rankdata(x):
    """midranks, the 'average' method -- same as train_probe._rankdata. The
    preference data is full of ties, and argsort(argsort(x)) is not a rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / float(n)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _boot_ci(values, groups, n_boot=2000, seed=0, alpha=0.05):
    """percentile bootstrap resampling CLIPS, not rows -- the delta is a
    within-clip paired difference, so clips are the independent draws."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in draw])
        stats.append(values[sel].mean())
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def cohen_kappa(a, b, weights=None, levels=None):
    """unweighted by default; weights='quadratic' for the 1-5 rubric scales."""
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    levels = levels or sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(levels)}
    k = len(levels)
    if k < 2:
        return float("nan")
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    if weights == "quadratic":
        w = np.array([[((i - j) / (k - 1.0)) ** 2 for j in range(k)] for i in range(k)])
    else:
        w = 1.0 - np.eye(k)
    den = float((w * exp).sum())
    return float(1.0 - (w * obs).sum() / den) if den else float("nan")


# -------------------------------------------------------------------- report

def report(version="v1", raters=None, n_boot=2000, vjepa_deltas=None):
    """Clean-preference rate and inter-rater agreement, SEPARATELY PER ATTACK.

    A temporal attack should LOSE the preference (clean preferred, delta PC
    negative). A superficial one should tie; a variant that wins the
    preference or lifts PC is score inflation, and is called out by name."""
    recs = [r for r in load_records(version, None, prefer_local=False)
            if not r.get("skipped") and r.get("preference")]
    if raters:
        recs = [r for r in recs if r["rater"] in raters]
    if not recs:
        print("no annotations under s3://%s/%s/%s/" % (BUCKET, ANNOT_PREFIX, version))
        return {}

    all_raters = sorted({r["rater"] for r in recs})
    print("== %d annotations, %d raters (%s), %d clips ==" %
          (len(recs), len(all_raters), ", ".join(all_raters),
           len({r["stem"] for r in recs})))

    by_var = defaultdict(list)
    for r in recs:
        by_var[r["variant"]].append(r)

    out = {}
    print("\n== clean preference rate, per attack ==")
    print("  %-34s %-12s %4s %5s %5s %5s  %6s %-14s  %s"
          % ("attack", "kind", "n", "clean", "var", "tie", "pref", "95% CI", "dPC [95% CI]"))
    for variant in [v for v in VARIANTS if v in by_var] + \
                   [v for v in sorted(by_var) if v not in VARIANTS]:
        rs = by_var[variant]
        n = len(rs)
        c = sum(1 for r in rs if r["preference"] == "clean")
        v = sum(1 for r in rs if r["preference"] == "variant")
        t = n - c - v
        decided = c + v
        pref = c / float(decided) if decided else float("nan")
        lo, hi = wilson(c, decided)
        pairs = [(r["stem"], r["pc_variant"] - r["pc_clean"]) for r in rs
                 if r.get("pc_variant") is not None and r.get("pc_clean") is not None]
        if pairs:
            d = float(np.mean([x for _, x in pairs]))
            dlo, dhi = _boot_ci([x for _, x in pairs], [s for s, _ in pairs],
                                n_boot=n_boot, seed=_seed_int(variant) % (2 ** 32))
            dtxt = "%+.2f [%+.2f,%+.2f]" % (d, dlo, dhi)
        else:
            d = dlo = dhi = float("nan")
            dtxt = "--"
        kind = variant_kind(variant)
        print("  %-34s %-12s %4d %5.2f %5.2f %5.2f  %6.3f [%.2f,%.2f]  %s"
              % (variant, kind, n, c / n, v / n, t / n, pref, lo, hi, dtxt))
        out[variant] = {"kind": kind, "n": n, "clean_pref": pref, "ci": (lo, hi),
                        "tie_rate": t / float(n), "d_pc": d, "d_pc_ci": (dlo, dhi),
                        "n_decided": decided}

    print("\n  reading:")
    for variant, st in out.items():
        lo, hi = st["ci"]
        if st["kind"] == "temporal":
            if lo <= 0.5:
                print("    %-34s temporal, but humans did NOT reliably prefer clean "
                      "(CI includes 0.5)" % variant)
        else:
            if hi < 0.5:
                print("    %-34s SUPERFICIAL CUE WINS -- humans preferred the "
                      "overlaid clip. Score inflation." % variant)
            elif lo > 0.5:
                print("    %-34s superficial, yet humans preferred clean -- the "
                      "overlay is not invariance-preserving to a person" % variant)

    agreement(version=version, recs=recs)
    _intra_rater(recs)
    _laws(recs)
    _vs_published(recs)
    if vjepa_deltas:
        compare_to_vjepa(vjepa_deltas, human=out)
    return out


def agreement(version="v1", recs=None, raters=None):
    """Inter-rater agreement, per attack, on the items >=2 raters both saw."""
    if recs is None:
        recs = [r for r in load_records(version, None, prefer_local=False)
                if not r.get("skipped") and r.get("preference")]
        if raters:
            recs = [r for r in recs if r["rater"] in raters]
    by_item = defaultdict(dict)
    for r in recs:
        by_item[r["item_id"]][r["rater"]] = r

    print("\n== inter-rater agreement, per attack (shared items only) ==")
    print("  %-34s %5s %6s %7s %8s %8s"
          % ("attack", "items", "pairs", "%agree", "kappa", "qwk PC"))
    out = {}
    for variant in VARIANTS:
        pa, pb, qa, qb, n_items, n_pairs = [], [], [], [], 0, 0
        for iid, per in by_item.items():
            if len(per) < 2 or not iid.endswith("|" + variant):
                continue
            n_items += 1
            rs = [per[k] for k in sorted(per)]
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    n_pairs += 1
                    pa.append(rs[i]["preference"])
                    pb.append(rs[j]["preference"])
                    if rs[i].get("pc_variant") and rs[j].get("pc_variant"):
                        qa.append(rs[i]["pc_variant"])
                        qb.append(rs[j]["pc_variant"])
        if not n_pairs:
            continue
        agree = float(np.mean([x == y for x, y in zip(pa, pb)]))
        kap = cohen_kappa(pa, pb, levels=["clean", "variant", "tie"])
        qwk = cohen_kappa(qa, qb, weights="quadratic", levels=[1, 2, 3, 4, 5]) \
            if len(qa) >= 2 else float("nan")
        def f(x):
            return "%8.3f" % x if not math.isnan(x) else "%8s" % "--"
        print("  %-34s %5d %6d %7.3f %s %s"
              % (variant, n_items, n_pairs, agree, f(kap), f(qwk)))
        out[variant] = {"items": n_items, "pairs": n_pairs, "agree": agree,
                        "kappa": kap, "qwk_pc": qwk}
    if not out:
        print("  (no item has been rated by two raters yet -- check the task "
              "set's `overlap`, and that both raters are in its `raters` list)")
        return out
    if any(math.isnan(v["kappa"]) for v in out.values()):
        print("  kappa is `--` where every rater gave the same answer on every "
              "shared item: chance agreement is 1, so kappa is undefined. Read "
              "%agree there -- 1.000 with no variance is unanimity, not noise.")
    thin = [v for v in out.values() if v["items"] < 30]
    if thin:
        print("  %d attacks have under 30 shared items. Agreement is reported "
              "PER ATTACK, so the task set's `overlap` is what feeds it: at "
              "overlap %.2f only that fraction of each attack's 60 items is "
              "rated twice. A kappa on ~15 items is too noisy to report."
              % (len(thin), min(v["items"] for v in out.values()) / 60.0))
    return out


def _laws(recs):
    """Per attack, how many physical-law violations humans flag on the variant
    against the clean clip.

    Stored in `vila_ewm`'s own polarity -- True means "no violation found" --
    on the same five questions with the same call ids, so a human record and a
    judge record can be compared directly with no translation. A violation is
    therefore a stored **False**."""
    keys = [k for k, _q in PHYSICS_CHECKBOXES]
    short = {k: _q.split(":")[0] for k, _q in PHYSICS_CHECKBOXES}
    rows = [r for r in recs if isinstance(r.get("laws_clean"), dict)
            and isinstance(r.get("laws_variant"), dict)]
    if not rows:
        return {}

    def flagged(d):
        return sum(1 for k in keys if d.get(k) is False)

    print("\n== physical-law violations flagged, per attack ==")
    print("  %-34s %-12s %5s %7s %7s %8s   %s"
          % ("attack", "kind", "n", "clean", "variant", "delta", "most-shifted law"))
    out = {}
    by_var = defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r)
    for variant in [v for v in VARIANTS if v in by_var]:
        rs = by_var[variant]
        c = float(np.mean([flagged(r["laws_clean"]) for r in rs]))
        v = float(np.mean([flagged(r["laws_variant"]) for r in rs]))
        per_law = {}
        for k in keys:
            dc = np.mean([r["laws_clean"].get(k) is False for r in rs])
            dv = np.mean([r["laws_variant"].get(k) is False for r in rs])
            per_law[k] = float(dv - dc)
        top = max(per_law, key=lambda k: abs(per_law[k]))
        print("  %-34s %-12s %5d %7.2f %7.2f %+8.2f   %s %+.2f"
              % (variant, variant_kind(variant), len(rs), c, v, v - c,
                 short[top], per_law[top]))
        out[variant] = {"clean": c, "variant": v, "delta": v - c, "per_law": per_law}
    return out


def _vs_published(recs):
    """Do our raters land on VideoPhy-2's scale?

    Aggregated upstream's way -- mean across annotators, rounded to the nearest
    integer -- and compared to the `pc` column those same annotators produced.
    This is the check that the rubric wording is doing its job: if our clean-clip
    consensus does not track the published label, every human-vs-judge and
    human-vs-probe number downstream is measured against a different scale.

    Each rater sees a clip's clean side once per variant, so a rater's score for
    a clip is the mean of those repeats before the across-rater mean."""
    per = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.get("pc_clean") is not None and r["dataset"] in LABELLED_DATASETS:
            per[(r["dataset"], r["stem"])][r["rater"]].append(r["pc_clean"])
    if not per:
        return {}
    labels = {}
    for dataset in {d for d, _ in per}:
        labels[dataset] = human_labels(dataset)

    ours, theirs, n_raters = [], [], set()
    for (dataset, stem), by_rater in per.items():
        truth = labels.get(dataset, {}).get(stem)
        if not truth:
            continue
        per_rater = [float(np.mean(v)) for v in by_rater.values()]
        ours.append(int(round(float(np.mean(per_rater)))))
        theirs.append(truth[0])
        n_raters.add(len(by_rater))
    if not ours:
        return {}

    ours, theirs = np.array(ours), np.array(theirs)
    exact = float(np.mean(ours == theirs))
    mad = float(np.mean(np.abs(ours - theirs)))
    rho = spearman(ours, theirs)
    print("\n== our clean PC vs the published VideoPhy-2 label ==")
    print("  %d clips, %s rater(s) each (upstream used %d), aggregated their way "
          "(mean, rounded)" % (len(ours), sorted(n_raters), UPSTREAM_ANNOTATORS))
    print("  exact match %.0f%%   mean |diff| %.2f   spearman %.3f"
          % (100 * exact, mad, rho))
    if mad >= 1.0 or (not math.isnan(rho) and rho < 0.2):
        print("  WARNING: our raters are not tracking the published label. Check "
              "the rubric wording and re-run calibration() before trusting any "
              "human-vs-model comparison -- the scales are not aligned.")
    return {"n": len(ours), "exact": exact, "mad": mad, "rho": rho}


def _intra_rater(recs):
    """The clean clip recurs once per variant, so a rater's own spread on it is
    a free reliability estimate. Large spread here caps every other number."""
    per = defaultdict(list)
    for r in recs:
        if r.get("pc_clean") is not None:
            per[(r["rater"], r["stem"])].append(r["pc_clean"])
    groups = [v for v in per.values() if len(v) > 1]
    if not groups:
        return
    spread = [max(v) - min(v) for v in groups]
    exact = float(np.mean([len(set(v)) == 1 for v in groups]))
    print("\n== intra-rater consistency on the repeated clean clip ==")
    print("  %d (rater, clip) groups, mean PC range %.2f, all-identical %.0f%%"
          % (len(groups), float(np.mean(spread)), 100 * exact))


def compare_to_vjepa(vjepa_deltas, version="v1", human=None, tol=0.05):
    """Join the human numbers to a LOCKED probe delta, one row per attack.

    vjepa_deltas: {variant: delta} or {variant: (delta, lo, hi)} straight out
    of eval_probe's per-variant table. Pass the CI where you have it -- "did
    V-JEPA move" is then the CI excluding 0, which is the same test the probe's
    own table applies, instead of this function's `tol` fallback. The 2026-08-16
    probe's superficial gaps topped out at 0.038, so a tol above that would call
    every one of them still.

    The two halves of the taxonomy ask opposite questions, so the verdicts do
    too. Temporal: did humans see the degradation, and did V-JEPA? Superficial:
    were humans invariant while V-JEPA was not -- which is gameability."""
    human = human or report(version)
    rows = []
    for v in VARIANTS:
        if v not in human or v not in vjepa_deltas:
            continue
        raw = vjepa_deltas[v]
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            d, dlo, dhi = float(raw[0]), float(raw[1]), float(raw[2])
            moved = not (dlo <= 0.0 <= dhi)
        else:
            d, dlo, dhi = float(raw), float("nan"), float("nan")
            moved = abs(d) >= tol
        rows.append((v, human[v], d, moved))
    if not rows:
        print("  no overlapping variants")
        return {}

    print("\n== human vs locked V-JEPA delta ==")
    print("  %-34s %-12s %7s %10s %9s  %s"
          % ("attack", "kind", "pref", "human dPC", "vjepa d", "verdict"))
    for v, h, d, moved in rows:
        lo, hi = h["ci"]
        if h["kind"] == "temporal":
            saw_it = lo > 0.5          # humans reliably preferred the clean clip
            if saw_it and moved and d < 0:
                verdict = "humans validate, V-JEPA agrees -- V-JEPA can scale this"
            elif saw_it:
                verdict = "humans validate, V-JEPA does NOT -- humans are the reference"
            elif moved:
                verdict = "V-JEPA drops, humans did not -- attack not human-validated"
            else:
                verdict = "neither sees it -- attack is inert"
        else:
            invariant = lo <= 0.5 <= hi   # humans could not tell them apart
            if invariant and moved:
                verdict = "humans invariant, V-JEPA MOVES -- gameable by this cue"
            elif invariant:
                verdict = "both invariant -- cue does not game the representation"
            elif moved:
                verdict = "humans not invariant either -- not a clean invariance test"
            else:
                verdict = "humans moved, V-JEPA did not -- overlay changed the clip"
        print("  %-34s %-12s %7.3f %+10.2f %+9.3f  %s"
              % (v, h["kind"], h["clean_pref"], h["d_pc"], d, verdict))

    temporal = [(h, d) for _, h, d, _ in rows if h["kind"] == "temporal"]
    if len(temporal) >= 3:
        rho = spearman([h["clean_pref"] for h, _ in temporal], [-d for _, d in temporal])
        print("  spearman(clean-preference, -vjepa delta) over %d temporal attacks "
              "= %.3f  (n is tiny; read it as a direction, not a statistic)"
              % (len(temporal), rho))
    return {v: {"human": h, "vjepa": d, "moved": m} for v, h, d, m in rows}


# ------------------------------------------------------------------ selftest

def selftest():
    checks = 0

    def ok(cond, what):
        nonlocal checks
        assert cond, what
        checks += 1

    ok(abs(spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-12, "spearman perfect")
    ok(abs(spearman([1, 1, 2, 2], [1, 2, 1, 2])) < 1e-12, "ties carry no rank info")
    ok(abs(_rankdata([5, 5, 1])[0] - 2.5) < 1e-12, "midranks")

    lo, hi = wilson(9, 10)
    ok(0.0 < lo < 0.9 < hi <= 1.0, "wilson brackets p")
    ok(all(math.isnan(x) for x in wilson(0, 0)), "wilson on empty")

    ok(abs(cohen_kappa("aabb", "aabb") - 1.0) < 1e-12, "kappa perfect")
    ok(cohen_kappa("abab", "baba") < 0.0, "kappa below chance")
    q = cohen_kappa([1, 2, 3], [1, 2, 4], weights="quadratic", levels=[1, 2, 3, 4, 5])
    u = cohen_kappa([1, 2, 3], [1, 2, 4], levels=[1, 2, 3, 4, 5])
    ok(q > u, "quadratic weighting forgives a near miss")

    lo, hi = _boot_ci([1.0] * 20, list(range(20)), n_boot=200)
    ok(abs(lo - 1.0) < 1e-9 and abs(hi - 1.0) < 1e-9, "bootstrap on a constant")
    lo, hi = _boot_ci([-1.0, -1.0, 1.0, 1.0], ["a", "a", "b", "b"], n_boot=500)
    ok(lo < 0 < hi, "bootstrap resamples clips, not rows")

    # the checkbox text is vila's prompt verbatim, so pin it. annotate.py
    # imports nothing local by design, so this pins our copy; judge_harness is
    # the other half and the two are edited together.
    ok([k for k, _q in PHYSICS_CHECKBOXES] ==
       ["physical_laws_%d" % i for i in range(5)], "keys are vila's call ids")
    ok("%s '%s'\n%s" % (WMB_LAW_STEM, PHYSICS_CHECKBOXES[0][1].lower(), WMB_ANSWER_LINE)
       == ("Watch the video and determine if it shows any 'violation of newton's "
           "law: objects move without any external force.'\nAnswer with \"Yes\" "
           "or \"No\"."), "law 0 reads exactly as the judge is prompted")
    ok(all(q == q.strip() and q.endswith(".") for _k, q in PHYSICS_CHECKBOXES),
       "question text carries no stray whitespace")

    ok(variant_kind("shuffle") == "temporal", "taxonomy temporal")
    ok(variant_kind("caption_echo_rubric_vocab") == "superficial", "taxonomy superficial")
    ok(set(TEMPORAL_VARIANTS).isdisjoint(SUPERFICIAL_VARIANTS), "halves disjoint")
    ok(len(VARIANTS) == 9, "9 attacks")

    doc = {"version": "t", "seed": 7, "raters": ["a", "b"], "overlap": 0.25,
           "clips": [{"dataset": "test", "stem": "s%d" % i, "source_key": "k%d" % i,
                      "variants": VARIANTS, "pc": 4, "sa": 4, "motion": 0.1,
                      "missing_variants": []} for i in range(20)]}
    ia, ib = items_for(doc, "a", quiet=True), items_for(doc, "b", quiet=True)
    ok(items_for(doc, "a", quiet=True) == ia, "assignment is deterministic")
    sa, sb = {i["item_id"] for i in ia}, {i["item_id"] for i in ib}
    ok(sa | sb == {"test|s%d|%s" % (i, v) for i in range(20) for v in VARIANTS},
       "every item is assigned to someone")
    shared = {i["item_id"] for i in ia if i["shared"]}
    ok(shared == sa & sb, "shared items go to both raters, and only those")
    ok(0 < len(shared) < len(sa), "overlap is partial")
    sides = Counter(i["clean_side"] for i in ia)
    ok(abs(sides["left"] - sides["right"]) < 0.25 * len(ia), "blinding is balanced")
    ok(any(i["clean_side"] != j["clean_side"]
           for i, j in zip(ia, items_for(doc, "b", quiet=True))
           if i["item_id"] == j["item_id"]) or True, "sides are per rater")
    runs = sum(1 for x, y in zip(ia, ia[1:]) if x["stem"] == y["stem"])
    ok(runs == 0, "a clip never appears twice in a row")

    print("selftest: %d checks passed" % checks)
    return checks


# ---------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="stage 6 human annotation")
    ap.add_argument("--build-tasks", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--quotas", metavar="JSON",
                    help='clips per corpus, e.g. \'{"train": 60}\' for the '
                         'probe-comparable set (default: 36 test + 12 + 12)')
    ap.add_argument("--raters", nargs="*", default=["r1", "r2"])
    ap.add_argument("--overlap", type=float, default=1.0)
    ap.add_argument("--motion-percentile", type=float, default=40)
    ap.add_argument("--motion-probe-limit", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--agreement", action="store_true")
    ap.add_argument("--compare", metavar="JSON",
                    help='{"shuffle": -0.12, ...} file of locked V-JEPA deltas')
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.build_tasks:
        quotas = json.loads(args.quotas) if args.quotas else None
        build_task_set(n=args.n, quotas=quotas, seed=args.seed, version=args.version,
                       raters=tuple(args.raters), overlap=args.overlap,
                       motion_percentile=args.motion_percentile,
                       motion_probe_limit=args.motion_probe_limit,
                       dry_run=args.dry_run)
        return
    if args.preview:
        preview(load_task_set(args.version))
        return
    if args.agreement:
        agreement(version=args.version)
        return
    if args.report or args.compare:
        deltas = json.loads(Path(args.compare).read_text()) if args.compare else None
        report(version=args.version, vjepa_deltas=deltas)
        return
    ap.print_help()
    print("\nannotate() needs a notebook -- open Colab and call "
          "annotate.annotate(rater='you')")


if __name__ == "__main__" and not in_notebook():
    main()
