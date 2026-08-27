"""Audit every judging run in S3 -- pass 1, pass 2 and the paraphrase runs.

Read-only. Loads no models and imports nothing local, so it runs in a bare
Colab runtime or on any box with credentials.

    python audit_runs.py                       # everything found under results/
    python audit_runs.py --prefix results/pass2
    python audit_runs.py --top 8               # more example clips per gap
    In Colab:  audit()  /  audit(prefix="results/pass2")

WHAT IT ANSWERS THAT A KEY COUNT CANNOT

Checkpointing is per (clip, variant, call), so a record object appears at a
clip's FIRST variant and a half-finished clip looks identical to a finished one
from a LIST. "Partial" is therefore the normal state of a live run and useless
on its own -- the question is always WHICH (variant, call) pairs are missing and
WHY. Every gap is classified as exactly one of:

  no_render       the variant was never rendered under attacks/<dataset>/<stem>/
                  so the harness printed `missing video` and skipped it. NOT a
                  judging failure -- the fix is attack_suite.py, and these are
                  excluded from the completeness percentage.
  variant_absent  the variant IS rendered but the record has no entry for it:
                  the run stopped before reaching it, or the clip was never
                  requested (a subset run). Expected mid-run, expected forever
                  on a deliberate subset.
  calls_missing   the variant is in the record but individual call ids are not:
                  judge.run() raised and printed `FAILED variant/call`. This is
                  the only class that is always a real defect.

Judges, datasets and variants are DISCOVERED by listing, never assumed. Pass 2
is phyjudge_9b ONLY, over a named variant subset of a clip subset: vila cannot
produce a rationale at all (four prompt rewrites and min_new_tokens were all
tested and failed) and videophy2 was not run. A hardcoded judge x dataset
matrix therefore reports cells as missing data when nothing was ever meant to
be there -- a judge with no keys under a prefix is silently skipped here
instead. For a subset run the expected variant set is inferred from the records
themselves and printed, so what is being checked is visible rather than
asserted.

Duplicated from judge_harness.py (the same trade check_results.py and
monitor.py make -- self-contained beats importable here): CALLS, VARIANTS,
DATASETS, RESULT_ROOT and the results/<pass>/<judge>/<dataset>/<stem>.json
layout. Edit them together with judge_harness.py or this reports phantom gaps.
"""

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3

# Colab hands credentials through userdata; everywhere else boto3's ambient
# chain does. Every read is optional -- the import succeeding does not mean
# userdata is usable (it needs a live kernel), so a failure falls through.
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
RESULT_ROOT = "results/"
s3 = boto3.client("s3")

# from judge_harness.ATTACK_FILES, with clean prepended
VARIANTS = [
    "clean", "shuffle", "reverse", "freeze", "photometric",
    "caption_echo_rubric_vocab", "caption_echo_score_anchor_positive",
    "caption_echo_authoritative_claim", "caption_echo_score_anchor_negative",
    "caption_echo_control_irrelevant",
]
TEMPORAL = {"shuffle", "reverse", "freeze"}

# from each judge's call_ids(). vila is 1 instruction score + 5 physical-law +
# 2 common-sense yes/no; phyjudge is 3 general dims + 13 named laws.
CALLS = {
    "vila_ewm": (["instruction"]
                 + [f"physical_laws_{i}" for i in range(5)]
                 + [f"common_sense_{i}" for i in range(2)]),
    "videophy2_auto": ["SA", "PC"],
    "phyjudge_9b": (["SA", "PTV", "persistence"]
                    + ["gravity", "inertia", "momentum", "impenetrability",
                       "collision", "material", "buoyancy", "displacement",
                       "flow_dynamics", "boundary_interaction",
                       "fluid_continuity", "reflection", "shadow"]),
}

DATASETS = ["test", "implausibench_real", "implausibench_implausible"]

# A parsed 0 is a real score only for vila's instruction call, which is 0-3.
# VideoPhy-2 SA/PC and phyjudge are 1-5, so a 0 there is upstream's
# unparseable-default leaking through. vila's yes/no calls parse to bools and
# `False == 0` in Python, so bools are excluded by type, not by call id.
ZERO_OK = {("vila_ewm", "instruction")}

# (prefix, judge, dataset) whose missing/absent clips are known and expected.
# Gaps there are still counted and printed; they just do not raise a PROBLEM.
IGNORE_MISSING = {
    ("results/paraphrase/p0", "phyjudge_9b", "implausibench_implausible"),
}

RATIONALE_MIN_CHARS = 60   # a pass-2 reply at or under this carries no prose


# ------------------------------------------------------------------ S3 helpers

def _list(prefix, delimiter=None):
    keys, prefixes = [], []
    paginator = s3.get_paginator("list_objects_v2")
    kw = {"Bucket": BUCKET, "Prefix": prefix}
    if delimiter:
        kw["Delimiter"] = delimiter
    for page in paginator.paginate(**kw):
        keys += [o["Key"] for o in page.get("Contents", [])]
        prefixes += [c["Prefix"] for c in page.get("CommonPrefixes", [])]
    return keys, prefixes


def _get(key):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except Exception:
        return None


def discover_runs(root=RESULT_ROOT):
    """-> [prefix] for every results/<pass>/ that holds records, with the
    paraphrase runs expanded one level deeper (results/paraphrase/p0, p1, ...).

    Listing beats a hardcoded list because a run nobody has started yet should
    be silently absent, not reported as a hole.
    """
    _, tops = _list(root, delimiter="/")
    out = []
    for top in sorted(tops):
        _, subs = _list(top, delimiter="/")
        # a pass prefix holds <judge>/ dirs; paraphrase holds p<k>/ dirs which
        # in turn hold the judges. One extra LIST tells the two apart.
        if any(s.rstrip("/").rsplit("/", 1)[-1] in CALLS for s in subs):
            out.append(top.rstrip("/"))
        else:
            out += [s.rstrip("/") for s in sorted(subs)]
    return out


def rendered_variants(dataset):
    """stem -> {variant} actually rendered under attacks/<dataset>/.

    This is the denominator that makes `no_render` separable from a judging
    failure. One paginated LIST, ~9 keys per clip.
    """
    prefix = f"attacks/{dataset}/"
    keys, _ = _list(prefix)
    have = defaultdict(set)
    for key in keys:
        rest = key[len(prefix):]
        stem, _, name = rest.rpartition("/")
        if stem and name.endswith(".mp4"):
            have[stem].add(name[: -len(".mp4")])
    for stem in have:
        have[stem].add("clean")      # clean IS the source object, always there
    return dict(have)


# ------------------------------------------------------------------- the audit

def _expected_variants(records, subset_threshold=0.5):
    """Which variants this run was MEANT to cover.

    A full run covers all ten. A subset run (pass 2) covers whatever the caller
    named, and there is no record of that anywhere in S3 -- so infer it: a
    variant present on more than half the records was requested, one present on
    a handful is a run that stopped early. Returned with the evidence so the
    inference is visible rather than silent.
    """
    seen = Counter()
    for rec in records:
        for v in rec.get("runs", {}):
            seen[v] += 1
    if not records or not seen:
        return [], seen, False
    top = max(seen.values())
    inferred = [v for v in VARIANTS if seen.get(v, 0) >= subset_threshold * top]
    inferred += sorted(v for v in seen if v not in VARIANTS)
    is_subset = len(inferred) < len(VARIANTS)
    return inferred, seen, is_subset


def audit_cell(prefix, judge, dataset, rendered, top=3):
    """One (prefix, judge, dataset). Prints its block, returns problem strings."""
    call_ids = CALLS[judge]
    keys, _ = _list(f"{prefix}/{judge}/{dataset}/")
    keys = [k for k in keys if k.endswith(".json")]
    if not keys:
        return []

    with ThreadPoolExecutor(32) as ex:
        records = [r for r in ex.map(_get, keys) if r]

    variants, seen, is_subset = _expected_variants(records)
    need = len(variants) * len(call_ids)

    # gap classification, aggregated three ways: by cause, by variant, by call
    by_cause = Counter()
    cause_variant = defaultdict(Counter)
    cause_call = defaultdict(Counter)
    examples = defaultdict(list)
    complete = attempted = 0
    unparsed = zeros = empty = 0
    rawlens = []

    for rec in records:
        stem = rec.get("clip", rec.get("source_key", "?"))
        runs = rec.get("runs", {})
        have_render = rendered.get(stem)
        gaps = 0
        for v in variants:
            done = runs.get(v, {}).get("calls", {})
            # a variant with no render was skipped by design, not failed
            if have_render is not None and v not in have_render:
                cause = "no_render"
            elif v not in runs:
                cause = "variant_absent"
            else:
                cause = "calls_missing"
            for c in call_ids:
                if c in done:
                    attempted += 1
                    out = done[c]
                    parsed, raw = out.get("parsed"), out.get("raw")
                    if parsed is None:
                        unparsed += 1
                    elif (isinstance(parsed, (int, float))
                          and not isinstance(parsed, bool)
                          and parsed == 0 and (judge, c) not in ZERO_OK):
                        zeros += 1
                    if not (raw or "").strip():
                        empty += 1
                    rawlens.append(len(raw or ""))
                    continue
                gaps += 1
                by_cause[cause] += 1
                cause_variant[cause][v] += 1
                cause_call[cause][c] += 1
                if len(examples[cause]) < top:
                    examples[cause].append((stem, v, c))
        if not gaps:
            complete += 1

    # completeness is measured against what could have been judged: a variant
    # nobody rendered is not a hole this run could have filled
    reachable = len(records) * need - by_cause["no_render"]
    pct = attempted / reachable if reachable else 0.0

    eligible = len(rendered) if rendered else None
    absent_clips = (eligible - len(records)) if eligible is not None else 0

    print(f"\n{judge:<15} {dataset:<28} {len(variants)}v x {len(call_ids)}c "
          f"= {need} calls/clip" + ("   [SUBSET RUN]" if is_subset else ""))
    if is_subset:
        counts = ", ".join(f"{v}:{seen[v]}" for v in variants)
        stray = {v: n for v, n in seen.items() if v not in variants}
        print(f"   inferred variant set -> {counts}")
        if stray:
            print(f"   below the subset threshold (partial early stop?) -> "
                  + ", ".join(f"{v}:{n}" for v, n in sorted(stray.items())))
    line = f"   clips {len(records)}"
    if eligible is not None and not is_subset:
        line += f"/{eligible} with renders"
    print(f"{line}   complete {complete}   partial {len(records) - complete}")
    print(f"   calls {attempted}/{reachable} reachable = {pct:.1%}   "
          f"unparsed {unparsed}   empty raw {empty}   illegal zeros {zeros}")

    for cause in ("calls_missing", "variant_absent", "no_render"):
        n = by_cause[cause]
        if not n:
            continue
        vs = ", ".join(f"{v}({c})" for v, c in cause_variant[cause].most_common(4))
        print(f"   {cause:<15} {n:>6} calls   variants: {vs}")
        if cause == "calls_missing":
            cs = ", ".join(f"{c}({n2})"
                           for c, n2 in cause_call[cause].most_common(4))
            print(f"   {'':<15} {'':>6}         calls:    {cs}")
        for stem, v, c in examples[cause]:
            print(f"   {'':<15} e.g. {stem[:52]}  {v}/{c}")

    if rawlens and prefix.endswith("pass2"):
        med = statistics.median(rawlens)
        prose = sum(1 for L in rawlens if L > RATIONALE_MIN_CHARS) / len(rawlens)
        print(f"   rationale: median {med:.0f} chars, {prose:.1%} over "
              f"{RATIONALE_MIN_CHARS}")
        if prose < 0.5:
            print(f"   WARNING most pass-2 replies carry no rationale text")

    ignored = (prefix, judge, dataset) in IGNORE_MISSING
    problems = []
    if by_cause["calls_missing"]:
        problems.append(f"{prefix}/{judge}/{dataset}: "
                        f"{by_cause['calls_missing']} calls FAILED mid-variant")
    if unparsed:
        problems.append(f"{prefix}/{judge}/{dataset}: {unparsed} unparsed calls")
    if zeros:
        problems.append(f"{prefix}/{judge}/{dataset}: {zeros} illegal zero scores")
    if empty:
        problems.append(f"{prefix}/{judge}/{dataset}: {empty} calls with empty raw")
    if not ignored and not is_subset:
        if absent_clips > 0:
            problems.append(f"{prefix}/{judge}/{dataset}: {absent_clips} "
                            f"eligible clips never scored")
        if by_cause["variant_absent"]:
            problems.append(f"{prefix}/{judge}/{dataset}: "
                            f"{by_cause['variant_absent']} calls on variants "
                            f"the record never reached (run incomplete)")
    if ignored:
        print("   NOTE gaps here are on the IGNORE_MISSING list, "
              "reported but not raised")
    return problems


def audit(prefix=None, judges=None, datasets=None, top=3):
    prefixes = [prefix] if prefix else discover_runs()
    judges = judges or list(CALLS)
    datasets = datasets or DATASETS

    print("rendered-variant index (one LIST per dataset) ...")
    rendered = {ds: rendered_variants(ds) for ds in datasets}
    for ds in datasets:
        full = sum(1 for v in rendered[ds].values() if len(v) == len(VARIANTS))
        print(f"  {ds:<28} {len(rendered[ds])} clips with renders, "
              f"{full} with all {len(VARIANTS)}")

    problems = []
    for pfx in prefixes:
        print("\n" + "=" * 78)
        print(pfx)
        print("=" * 78)
        for judge in judges:
            for ds in datasets:
                problems += audit_cell(pfx, judge, ds, rendered[ds], top=top)

    print("\n" + "=" * 78)
    if problems:
        print(f"{len(problems)} PROBLEM(S)")
        for p in problems:
            print("  -", p)
    else:
        print("no problems detected")
    return problems


def in_notebook():
    try:
        __IPYTHON__          # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(description="audit results/ in S3")
    ap.add_argument("--prefix", help="one run, e.g. results/pass2")
    ap.add_argument("--judges", nargs="+", choices=list(CALLS))
    ap.add_argument("--datasets", nargs="+", choices=DATASETS)
    ap.add_argument("--top", type=int, default=3,
                    help="example clips printed per gap class")
    a = ap.parse_args()
    audit(prefix=a.prefix, judges=a.judges, datasets=a.datasets, top=a.top)
