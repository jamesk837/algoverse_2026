"""End-of-project completeness check for the S3 database (nickb-aarj).

Read-only. Loads no models, imports nothing local, runs in a bare Colab
runtime or on any box with credentials.

    python check_complete.py                 # full check, every section
    python check_complete.py --quick         # counts + a sample, no deep call scan
    python check_complete.py --section pass1  # one section only
    In Colab:  import check_complete as C; C.main()

WHAT "COMPLETE" MEANS HERE -- the contract this asserts, from the project spec:

  Phase 1 judge runs, one full matrix per PROMPT CONDITION
    conditions   results/pass1, results/paraphrase/p0, results/paraphrase/p1
    judges       vila_ewm (8 calls), videophy2_auto (2), phyjudge_9b (16)
    datasets     test=450, implausibench_real=150, implausibench_implausible=150
    variants     10 = clean + 9 attacks
    a clip is COMPLETE when all 10 variants are present, every call in each is
    present, and no parsed score is null

  Ablation 11 -- identity codec control
    render   attacks/<ds>/<stem>/identity.mp4 for every curated clip
    score    results/pass1/<judge>/<ds>/<stem>.json carries a full `identity`
             variant for all three judges  (pass 1 only)

  Ablation 1 -- photometric on real
    a targeted re-check that `photometric` is present and parsed for every
    implausibench_real clip in pass1 (it is inside the main matrix, but the
    ablation reads it on its own so it is verified on its own)

  Pass 2 -- score + rationale, phyjudge_9b ONLY
    variants   clean, shuffle, caption_echo_control_irrelevant,
               caption_echo_score_anchor_positive
    counts     test=80, implausibench_implausible=25, implausibench_real=15
    each scored variant should also carry rationale prose

  Phase 2 artifacts -- existence / shape only
    split, both embedding packs, the 4-arch probe ablation family, the locked
    reference probe + its dV export, and the Step-10 predictor outputs

  NOT CHECKED -- flagged, not asserted
    the "3 attack-aware temporal upper-bound" probes: nothing in this repo
    (no script, no S3 prefix, no commit) defines that artifact, so this file
    cannot verify it. Confirm with the project lead whether it was descoped or
    lives outside the bucket.

Constants below are duplicated from judge_harness.py (VARIANTS, CALLS,
DATASET names, result layout) -- the same trade audit_runs.py / check_results.py
/ monitor.py make. Edit them together with judge_harness.py.
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

# Colab hands credentials through userdata; everywhere else boto3's ambient
# chain does. Every read is optional -- the import succeeding does not mean
# userdata is usable (needs a live kernel), so any failure falls through.
try:
    from google.colab import userdata
except ImportError:
    userdata = None

if userdata is not None:
    for _v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        try:
            os.environ[_v] = userdata.get(_v)
        except Exception:
            pass
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

BUCKET = "nickb-aarj"
READ_WORKERS = 32
s3 = boto3.client("s3", config=Config(
    max_pool_connections=READ_WORKERS + 8,
    retries={"max_attempts": 5, "mode": "standard"}))

# ---- the phase-1 matrix ---------------------------------------------------

# from judge_harness.ATTACK_FILES, with clean prepended
ATTACKS = [
    "shuffle", "reverse", "freeze", "photometric",
    "caption_echo_rubric_vocab", "caption_echo_score_anchor_positive",
    "caption_echo_authoritative_claim", "caption_echo_score_anchor_negative",
    "caption_echo_control_irrelevant",
]
VARIANTS = ["clean"] + ATTACKS          # the 10-wide experiment
TEMPORAL = {"shuffle", "reverse", "freeze"}
IDENTITY = "identity"                   # ablation 11, rendered + scored separately

# from each judge's call_ids()
CALLS = {
    "vila_ewm": (["instruction"]
                 + ["physical_laws_%d" % i for i in range(5)]
                 + ["common_sense_%d" % i for i in range(2)]),
    "videophy2_auto": ["SA", "PC"],
    "phyjudge_9b": (["SA", "PTV", "persistence"]
                    + ["gravity", "inertia", "momentum", "impenetrability",
                       "collision", "material", "buoyancy", "displacement",
                       "flow_dynamics", "boundary_interaction",
                       "fluid_continuity", "reflection", "shadow"]),
}
JUDGES = list(CALLS)

DATASETS = ["test", "implausibench_real", "implausibench_implausible"]
CLIP_TARGETS = {"test": 450, "implausibench_real": 150,
                "implausibench_implausible": 150}

# prompt conditions: native + two paraphrases, each a full matrix
PROMPT_RUNS = {
    "pass1": "results/pass1",
    "p0": "results/paraphrase/p0",
    "p1": "results/paraphrase/p1",
}

ATTACKS_PREFIX = "attacks"

# a parsed 0 is a real score only for vila's instruction call (0-3). VP2 SA/PC
# and phyjudge are 1-5, so a 0 there is upstream's unparseable default. vila's
# yes/no calls parse to bool and False == 0, so bools are excluded by type.
ZERO_OK = {("vila_ewm", "instruction")}

# ---- pass 2 -------------------------------------------------------------

PASS2_PREFIX = "results/pass2"
PASS2_JUDGE = "phyjudge_9b"
PASS2_VARIANTS = ["clean", "shuffle", "caption_echo_control_irrelevant",
                  "caption_echo_score_anchor_positive"]
PASS2_TARGETS = {"test": 80, "implausibench_implausible": 25,
                 "implausibench_real": 15}
RATIONALE_MIN_CHARS = 60       # a reply at or under this carries no prose

# ---- phase 2 artifacts ------------------------------------------------

HUB_MODEL = "vjepa2_1_vit_large_384"
PRED_MODEL = "vjepa2_1_vit_giant_384"
SPLIT_KEY = "splits/videophy2_train/split_v1.json"
PACKS_T32 = ["embeddings/%s/packs_t32/%s.npz" % (HUB_MODEL, s)
             for s in ("train", "val", "cal")]
PACKS_MEAN = ["embeddings/%s/packs/%s.npz" % (HUB_MODEL, s)
              for s in ("train", "val", "cal")]
PROBE_PREFIX = "probes"
ABLATION_ARCHS = ["attn", "mean_linear", "proj_mean", "diff_conv"]
REFERENCE_PREFIX = "reference"
PREDICTOR_REPORT = "predictor/%s/report.json" % PRED_MODEL
PREDICTOR_ERRORS = "predictor/%s/errors/" % PRED_MODEL


# ======================================================================
# S3 helpers
# ======================================================================

def _list(prefix, delimiter=None):
    keys, prefixes = [], []
    p = s3.get_paginator("list_objects_v2")
    kw = {"Bucket": BUCKET, "Prefix": prefix}
    if delimiter:
        kw["Delimiter"] = delimiter
    for page in p.paginate(**kw):
        keys += [o["Key"] for o in page.get("Contents", [])]
        prefixes += [c["Prefix"] for c in page.get("CommonPrefixes", [])]
    return keys, prefixes


def _get(key):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except Exception:
        return None


def _get_many(keys):
    if not keys:
        return []
    with ThreadPoolExecutor(min(READ_WORKERS, len(keys))) as ex:
        return list(ex.map(_get, keys))


def _exists(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


# ======================================================================
# curated corpus -- which stems each dataset is supposed to carry
# ======================================================================

def curated_stems(dataset):
    """stem -> set(rendered variant names) under attacks/<dataset>/.

    Returns (full, partial, identity_stems):
      full     stems with every one of the 9 attacks rendered (the corpus)
      partial  stem -> missing attack names, for stems with some but not all
      identity stems that also carry an identity.mp4 render
    """
    prefix = "%s/%s/" % (ATTACKS_PREFIX, dataset)
    keys, _ = _list(prefix)
    have = defaultdict(set)
    for k in keys:
        rest = k[len(prefix):]
        stem, _, name = rest.rpartition("/")
        if stem and name.endswith(".mp4"):
            have[stem].add(name[:-4])
    full, partial, ident = set(), {}, set()
    for stem, got in have.items():
        got.discard("clean")
        if IDENTITY in got:
            ident.add(stem)
        miss = [a for a in ATTACKS if a not in got]
        if not miss:
            full.add(stem)
        elif len(miss) < len(ATTACKS):
            partial[stem] = miss
    return full, partial, ident


# ======================================================================
# phase 1 -- one prompt condition's full matrix
# ======================================================================

def _scan_record(rec, judge, variants):
    """-> dict of per-record gap counts, one class per missing call."""
    call_ids = CALLS[judge]
    runs = rec.get("runs", {})
    out = dict(missing_variant=0, missing_call=0, unparsed=0, zero=0, empty=0,
               complete=True)
    for v in variants:
        calls = runs.get(v, {}).get("calls", {})
        if v not in runs:
            out["missing_variant"] += len(call_ids)
            out["complete"] = False
            continue
        for c in call_ids:
            if c not in calls:
                out["missing_call"] += 1
                out["complete"] = False
                continue
            o = calls[c]
            parsed, raw = o.get("parsed"), o.get("raw")
            if parsed is None:
                out["unparsed"] += 1
                out["complete"] = False
            elif (isinstance(parsed, (int, float)) and not isinstance(parsed, bool)
                  and parsed == 0 and (judge, c) not in ZERO_OK):
                out["zero"] += 1
            if not (raw or "").strip():
                out["empty"] += 1
    return out


def check_phase1(runs=None, judges=None, datasets=None, quick=False, top=8):
    runs = runs or list(PROMPT_RUNS)
    judges = judges or JUDGES
    datasets = datasets or DATASETS
    problems = []

    print("=" * 78)
    print("PHASE 1 -- judge matrix (native + paraphrase prompt conditions)")
    print("=" * 78)

    print("\ncurated corpus (attacks/<ds>/ with all 9 attacks rendered):")
    corpus = {}
    for ds in datasets:
        full, partial, ident = curated_stems(ds)
        corpus[ds] = (full, partial, ident)
        tgt = CLIP_TARGETS[ds]
        flag = "" if len(full) == tgt else "  <-- expected %d" % tgt
        print("  %-26s %4d full  %3d partial  %4d with identity%s"
              % (ds, len(full), len(partial), len(ident), flag))
        if partial and top:
            for stem, miss in list(partial.items())[:top]:
                print("      partial: %s  missing %s" % (stem[:52], ",".join(miss)))
        if len(full) != tgt:
            problems.append("attacks/%s: %d fully-rendered clips, expected %d"
                            % (ds, len(full), tgt))

    for cond in runs:
        prefix = PROMPT_RUNS[cond]
        print("\n--- %s  (%s) " % (cond, prefix) + "-" * 40)
        for judge in judges:
            for ds in datasets:
                full, _, _ = corpus[ds]
                keys, _ = _list("%s/%s/%s/" % (prefix, judge, ds))
                keys = [k for k in keys if k.endswith(".json")]
                got_stems = {k.rsplit("/", 1)[-1][:-5] for k in keys}
                missing_clips = sorted(full - got_stems)
                extra_clips = sorted(got_stems - full)

                agg = Counter()
                complete = 0
                scan_keys = keys if not quick else keys[:25]
                for rec in _get_many(scan_keys):
                    if not rec:
                        agg["unreadable"] += 1
                        continue
                    r = _scan_record(rec, judge, VARIANTS)
                    complete += r.pop("complete")
                    for k, val in r.items():
                        agg[k] += val

                scope = "sample of %d" % len(scan_keys) if quick else "%d recs" % len(keys)
                clean = (not missing_clips and not extra_clips
                         and not agg["unparsed"] and not agg["missing_call"]
                         and not agg["unreadable"]
                         and (quick or complete == len(keys)))
                print("  [%s] %-15s %-26s %4d/%d clips  complete %d  (%s)"
                      % ("OK" if clean else "GAP", judge, ds,
                         len(got_stems), len(full), complete, scope))
                if missing_clips:
                    problems.append("%s/%s/%s: %d curated clips never scored"
                                    % (prefix, judge, ds, len(missing_clips)))
                    for stem in missing_clips[:top]:
                        print("        missing clip: %s" % stem[:56])
                    if len(missing_clips) > top:
                        print("        ... +%d more" % (len(missing_clips) - top))
                if extra_clips:
                    print("        note: %d scored clips are NOT in the curated "
                          "set (e.g. %s)" % (len(extra_clips), extra_clips[0][:48]))
                for k in ("missing_variant", "missing_call", "unparsed",
                          "zero", "empty", "unreadable"):
                    if not agg[k]:
                        continue
                    hard = k in ("missing_call", "unparsed", "unreadable")
                    if hard:
                        problems.append("%s/%s/%s: %d %s"
                                        % (prefix, judge, ds, agg[k], k))
                    print("        %s: %d%s"
                          % (k, agg[k], "" if hard else "  (informational)"))
    return problems


# ======================================================================
# ablation 11 -- identity codec control
# ======================================================================

def check_identity(judges=None, datasets=None, quick=False, top=8):
    judges = judges or JUDGES
    datasets = datasets or DATASETS
    problems = []
    print("\n" + "=" * 78)
    print("ABLATION 11 -- identity codec control (render + pass-1 score)")
    print("=" * 78)

    for ds in datasets:
        full, _, ident = curated_stems(ds)
        missing_render = sorted(full - ident)
        print("\n  %s:  %d/%d curated clips have attacks/%s/<stem>/identity.mp4"
              % (ds, len(ident), len(full), ds))
        if missing_render:
            problems.append("attacks/%s: %d curated clips missing the identity "
                            "render" % (ds, len(missing_render)))
            for stem in missing_render[:top]:
                print("      no identity render: %s" % stem[:56])

        for judge in judges:
            keys, _ = _list("%s/%s/%s/" % (PROMPT_RUNS["pass1"], judge, ds))
            keys = [k for k in keys if k.endswith(".json")]
            recs = _get_many(keys if not quick else keys[:25])
            scored = missing = badcalls = 0
            for rec in recs:
                if not rec:
                    continue
                calls = rec.get("runs", {}).get(IDENTITY, {}).get("calls", {})
                if not calls:
                    missing += 1
                    continue
                scored += 1
                badcalls += sum(1 for c in CALLS[judge]
                                if c not in calls
                                or calls.get(c, {}).get("parsed") is None)
            base = len(recs) if quick else len(full)
            ok = (missing == 0 and badcalls == 0 and not quick
                  and scored == len(full))
            print("    [%s] %-15s identity scored on %d/%d clips, %d missing, "
                  "%d bad calls%s"
                  % ("OK" if ok else "GAP", judge, scored, base, missing,
                     badcalls, "  (sample)" if quick else ""))
            if not quick and scored < len(full):
                problems.append("pass1/%s/%s: identity variant on %d/%d clips"
                                % (judge, ds, scored, len(full)))
            if badcalls:
                problems.append("pass1/%s/%s: identity has %d missing/unparsed "
                                "calls" % (judge, ds, badcalls))
    return problems


# ======================================================================
# ablation 1 -- photometric on real (targeted re-check)
# ======================================================================

def check_photometric_real(quick=False):
    problems = []
    print("\n" + "=" * 78)
    print("ABLATION 1 -- photometric on implausibench_real (targeted)")
    print("=" * 78)
    ds = "implausibench_real"
    full, _, _ = curated_stems(ds)
    for judge in JUDGES:
        keys, _ = _list("%s/%s/%s/" % (PROMPT_RUNS["pass1"], judge, ds))
        keys = [k for k in keys if k.endswith(".json")]
        recs = _get_many(keys if not quick else keys[:25])
        ok = bad = absent = 0
        for rec in recs:
            if not rec:
                continue
            calls = rec.get("runs", {}).get("photometric", {}).get("calls", {})
            if not calls:
                absent += 1
            elif any(c not in calls or calls.get(c, {}).get("parsed") is None
                     for c in CALLS[judge]):
                bad += 1
            else:
                ok += 1
        base = len(recs) if quick else len(full)
        good = (absent == 0 and bad == 0 and not quick and ok == len(full))
        print("  [%s] %-15s photometric fully parsed on %d/%d clips  "
              "(%d absent, %d incomplete)%s"
              % ("OK" if good else "GAP", judge, ok, base, absent, bad,
                 "  (sample)" if quick else ""))
        if not quick and (absent or bad):
            problems.append("pass1/%s/%s: photometric absent on %d, incomplete "
                            "on %d" % (judge, ds, absent, bad))
    return problems


# ======================================================================
# pass 2 -- score + rationale, phyjudge only
# ======================================================================

def check_pass2(quick=False, top=8):
    problems = []
    print("\n" + "=" * 78)
    print("PASS 2 -- score + rationale (phyjudge_9b only)")
    print("=" * 78)
    judge = PASS2_JUDGE

    for other in ("vila_ewm", "videophy2_auto"):
        k, _ = _list("%s/%s/" % (PASS2_PREFIX, other))
        if k:
            print("  note: %d stray %s records under %s/ (early attempt, not "
                  "part of the run -- ignored)" % (len(k), other, PASS2_PREFIX))

    for ds, target in PASS2_TARGETS.items():
        keys, _ = _list("%s/%s/%s/" % (PASS2_PREFIX, judge, ds))
        keys = [k for k in keys if k.endswith(".json")]
        recs = [r for r in _get_many(keys) if r]
        n = len(recs)

        var_seen = Counter()
        rawlens, missing_call, unparsed, full_clips = [], 0, 0, 0
        for rec in recs:
            runs = rec.get("runs", {})
            here = 0
            for v in PASS2_VARIANTS:
                calls = runs.get(v, {}).get("calls", {})
                if not calls:
                    continue
                var_seen[v] += 1
                here += 1
                for c in CALLS[judge]:
                    o = calls.get(c)
                    if o is None:
                        missing_call += 1
                        continue
                    if o.get("parsed") is None:
                        unparsed += 1
                    rawlens.append(len(o.get("raw") or ""))
            if here == len(PASS2_VARIANTS):
                full_clips += 1

        ok = (n == target and full_clips == target
              and not missing_call and not unparsed)
        print("\n  [%s] %-26s %d/%d clips written, %d with all %d variants"
              % ("OK" if ok else "GAP", ds, n, target, full_clips,
                 len(PASS2_VARIANTS)))
        print("        per-variant coverage: "
              + ", ".join("%s:%d" % (v.split("caption_echo_")[-1], var_seen[v])
                          for v in PASS2_VARIANTS))
        if missing_call or unparsed:
            print("        missing calls %d, unparsed %d" % (missing_call, unparsed))
            problems.append("%s/%s/%s: %d missing calls, %d unparsed"
                            % (PASS2_PREFIX, judge, ds, missing_call, unparsed))
        if rawlens:
            med = statistics.median(rawlens)
            prose = sum(1 for L in rawlens if L > RATIONALE_MIN_CHARS) / len(rawlens)
            print("        rationale: median %.0f chars, %.0f%% over %d"
                  % (med, prose * 100, RATIONALE_MIN_CHARS))
            if prose < 0.5:
                problems.append("%s/%s/%s: most pass-2 replies carry no "
                                "rationale prose" % (PASS2_PREFIX, judge, ds))
        if n != target:
            problems.append("%s/%s/%s: %d clips, expected %d"
                            % (PASS2_PREFIX, judge, ds, n, target))
    return problems


# ======================================================================
# phase 2 artifacts -- existence / shape
# ======================================================================

def check_phase2_artifacts():
    problems = []
    print("\n" + "=" * 78)
    print("PHASE 2 artifacts (existence / shape only)")
    print("=" * 78)

    def row(label, key, hard=True):
        ok = _exists(key)
        print("  [%s] %-34s s3://%s/%s"
              % ("OK" if ok else "MISS", label, BUCKET, key))
        if not ok and hard:
            problems.append("missing: %s" % key)
        return ok

    row("split", SPLIT_KEY)
    for k in PACKS_T32:
        row("pack (t32, current)", k)
    for k in PACKS_MEAN:
        row("pack (mean, baseline)", k, hard=False)

    print("\n  probes/ :")
    keys, _ = _list("%s/" % PROBE_PREFIX)
    names = sorted({k.rsplit("/", 1)[-1] for k in keys if k.endswith(".pt")})
    for nm in names:
        print("      %s" % nm)
    if not names:
        problems.append("probes/: no .pt checkpoints at all")

    fam = defaultdict(set)
    for nm in names:
        stem = nm[:-3]
        for arch in ABLATION_ARCHS:
            if stem.endswith("_" + arch):
                fam[stem[:-(len(arch) + 1)]].add(arch)
    complete_fam = [b for b, a in fam.items() if set(a) == set(ABLATION_ARCHS)]
    if complete_fam:
        print("  [OK]   4-arch ablation family present: "
              + ", ".join("%s_{%s}" % (b, ",".join(ABLATION_ARCHS))
                          for b in complete_fam))
    else:
        print("  [MISS] no complete 4-arch ablation family {%s}; partial: %s"
              % (",".join(ABLATION_ARCHS),
                 {b: sorted(a) for b, a in fam.items()} or "none"))
        problems.append("probes/: no complete 4-arch ablation family "
                        "(train_probe.py --ablation)")

    locked = [nm for nm in names if "lock" in nm.lower()]
    if locked:
        for nm in locked:
            has_json = _exists("%s/%s.json" % (PROBE_PREFIX, nm[:-3]))
            print("  [%s]   locked probe %s (+ .json manifest: %s)"
                  % ("OK" if has_json else "MISS", nm, has_json))
            if not has_json:
                problems.append("probes/%s.json manifest missing" % nm[:-3])
    else:
        print("  [MISS] no locked reference probe (lock_probe.py)")
        problems.append("probes/: no locked reference probe")

    print("\n  reference/ (locked probe -> dV, for Step 12):")
    _, refdirs = _list("%s/" % REFERENCE_PREFIX, delimiter="/")
    if not refdirs:
        print("  [MISS] reference/ is empty")
        problems.append("reference/: no dV export (score_corpus.py)")
    for d in refdirs:
        nm = d.rstrip("/").rsplit("/", 1)[-1]
        for fn in ("dv.json", "vjepa_deltas.json"):
            ok = _exists("%s%s" % (d, fn))
            print("      [%s] %s/%s" % ("OK" if ok else "MISS", nm, fn))
            if not ok:
                problems.append("reference/%s/%s missing" % (nm, fn))

    print("\n  Step 10 predictor (optional -- gated by decide_predictor_probe):")
    rep_ok = _exists(PREDICTOR_REPORT)
    err_keys, _ = _list(PREDICTOR_ERRORS)
    err_n = len([k for k in err_keys if k.endswith(".npz")])
    print("      [%s] %s" % ("OK" if rep_ok else "MISS", PREDICTOR_REPORT))
    print("      [%s] %s*.npz  (%d files)"
          % ("OK" if err_n else "MISS", PREDICTOR_ERRORS, err_n))
    if not rep_ok or not err_n:
        print("      note: Step 10 is optional; treat as informational unless "
              "the predictor run was intended.")
    return problems


# ======================================================================
# the thing we cannot verify
# ======================================================================

def note_unverifiable():
    print("\n" + "=" * 78)
    print("NOT CHECKED -- needs a human answer")
    print("=" * 78)
    print("""
  "3 attack-aware temporal upper-bound" probes
      No script, S3 prefix, git commit, or CLAUDE.md entry in this repo
      defines this artifact, so this checker has nothing to look for. The
      closest thing that exists is train_probe.py --ablation (the 4-arch
      order-sensitivity ablation), which the project lead framed as the
      REPLACEMENT for training on shuffle/reverse/freeze -- see
      no-temporal-in-probe-training. If an attack-aware upper bound was a
      separate deliverable, it either was descoped or lives outside this
      bucket. Confirm with the project lead.
""")


# ======================================================================

SECTIONS = {
    "pass1": lambda a: check_phase1(runs=a.runs, judges=a.judges,
                                    datasets=a.datasets, quick=a.quick,
                                    top=a.top),
    "identity": lambda a: check_identity(judges=a.judges, datasets=a.datasets,
                                         quick=a.quick, top=a.top),
    "photometric": lambda a: check_photometric_real(quick=a.quick),
    "pass2": lambda a: check_pass2(quick=a.quick, top=a.top),
    "phase2": lambda a: check_phase2_artifacts(),
}


def main(args=None):
    if args is None:
        args = _parse([])
    problems = []
    todo = args.section or list(SECTIONS)
    for name in todo:
        problems += SECTIONS[name](args)
    note_unverifiable()

    print("\n" + "#" * 78)
    if problems:
        print("# %d PROBLEM(S) -- data is NOT complete" % len(problems))
        for p in problems:
            print("#   - %s" % p)
    else:
        scope = "checked sections" if args.section else "all sections"
        qual = " (quick mode -- counts and samples only)" if args.quick else ""
        print("# no problems detected across %s%s" % (scope, qual))
    print("#" * 78)
    return problems


def _parse(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", nargs="+", choices=list(SECTIONS),
                    help="run only these sections (default: all)")
    ap.add_argument("--runs", nargs="+", choices=list(PROMPT_RUNS),
                    help="phase-1 prompt conditions to check (default: all 3)")
    ap.add_argument("--judges", nargs="+", choices=JUDGES)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS)
    ap.add_argument("--quick", action="store_true",
                    help="skip the deep per-call GET scan; counts + 25-rec sample")
    ap.add_argument("--top", type=int, default=8,
                    help="example stems printed per gap")
    return ap.parse_args(argv)


def in_notebook():
    try:
        __IPYTHON__          # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__" and not in_notebook():
    sys.exit(1 if main(_parse(sys.argv[1:])) else 0)
