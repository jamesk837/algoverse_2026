"""Repair what audit_runs.py found: re-render, re-parse, and re-judge.

    python fix_runs.py                  # plan only. Reads S3, writes nothing.
    python fix_runs.py --reparse        # dry-run the phyjudge re-parse
    python fix_runs.py --reparse --write
    python fix_runs.py --render         # re-render the missing attack variants
    In Colab:  import fix_runs; fix_runs.plan()

The gap list is DERIVED from S3 on every run, never hardcoded, so this cannot
go stale against a run that has moved on since the audit.

THE THREE GAP CLASSES NEED THREE DIFFERENT FIXES, AND ONLY ONE IS A RERUN

`unparsed` -- RE-PARSE, never re-judge. phyjudge decodes greedily
    (`do_sample=False`), so the same prompt on the same clip emits
    byte-identical tokens and parses to None again: a rerun cannot fix it and
    the GPU time buys a copy of what is already stored. Every observed case is
    the same failure -- the model renamed the JSON key after the video's
    subject (`{"flow_darts": 2}`, `{"flow_drops": 3}` for `flow_dynamics`) and
    upstream's `infer.parse_score` looks the key up by name. The raw text is
    stored precisely so a score can be recovered without re-running a model,
    which is what reparse() does.

`no_render` -- attack_suite FIRST, then the judges. The variant was never
    rendered, so every judge skipped it and printed `missing video`. Rendering
    after a rerun leaves it just as unscored, so the order is not optional.

`variant_absent` -- re-judge, and this is the only true rerun. Checkpointing is
    per (clip, variant, call), so re-running the whole corpus regenerates ONLY
    the missing calls: a complete clip is skipped before its video is even
    downloaded. There is no need to name the clip, and no way to -- run_judges
    takes no stem filter.

WHY THE RERUNS ARE PRINTED RATHER THAN RUN

The three judges' upstream pins are mutually exclusive (torch 2.3 + tf 4.46 for
vila, tf 4.28.1 for videophy2, and something newer than either for phyjudge),
so one process cannot hold two of them, and a second model load in a live
interpreter is what produces phyjudge's silent meta-device corruption. Commands
are therefore emitted per venv, one judge per process, for you to run on the
box that has that stack. Each carries the flags that reproduce the run being
repaired -- `paraphrase=k` reworded prompts, `pass2=True` rewritten format
instructions -- because the flag is what selects both the prompt and the
destination prefix, and getting it wrong writes a pass-2 record into pass 1.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

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

# duplicated from judge_harness.py, as in audit_runs.py / check_results.py --
# this has to run in a bare runtime with no torch. Edit them together.
VARIANTS = [
    "clean", "shuffle", "reverse", "freeze", "photometric",
    "caption_echo_rubric_vocab", "caption_echo_score_anchor_positive",
    "caption_echo_authoritative_claim", "caption_echo_score_anchor_negative",
    "caption_echo_control_irrelevant",
]
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
VENVS = {"phyjudge_9b": "phyjudge", "vila_ewm": "vila",
         "videophy2_auto": "videophy2"}

PASS2_PREFIX = "results/pass2"
PASS2_JUDGES = ["phyjudge_9b"]

# phyjudge's scale. A repaired score outside it is not a rename, it is
# something else, and reparse() refuses rather than inventing a number.
PHYJUDGE_MIN, PHYJUDGE_MAX = 1, 5
_JSON_OBJ = re.compile(r"\{.*?\}", re.S)


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


def _put(key, obj):
    s3.put_object(Bucket=BUCKET, Key=key,
                  Body=json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"),
                  ContentType="application/json")


def discover_runs(root="results/"):
    _, tops = _list(root, delimiter="/")
    out = []
    for top in sorted(tops):
        _, subs = _list(top, delimiter="/")
        if any(s.rstrip("/").rsplit("/", 1)[-1] in CALLS for s in subs):
            out.append(top.rstrip("/"))
        else:
            out += [s.rstrip("/") for s in sorted(subs)]
    return out


def judges_for(prefix):
    return PASS2_JUDGES if prefix.startswith(PASS2_PREFIX) else list(CALLS)


def rerun_kwargs(prefix, dataset, variants):
    """The run_judges call that reproduces `prefix` -- prompts AND destination.

    The flag is not bookkeeping: paraphrase=k rewords every prompt, pass2=True
    rewrites each judge's format instruction, and each routes to its own S3
    prefix. Calling with the wrong one writes the wrong text into the wrong
    place, which is unrecoverable without knowing it happened.
    """
    kw = {"dataset": dataset, "num_clips": None}
    m = re.fullmatch(r"results/paraphrase/p(\d+)", prefix)
    if m:
        kw["paraphrase"] = int(m.group(1))
    elif prefix.startswith(PASS2_PREFIX):
        kw["pass2"] = True
        kw["require_pass1"] = "phyjudge_9b"   # how the run was defined
        kw["variants"] = sorted(variants)
    elif prefix != "results/pass1":
        raise ValueError(f"unknown run prefix {prefix}")
    return kw


def rendered_variants(dataset):
    prefix = f"attacks/{dataset}/"
    keys, _ = _list(prefix)
    have = defaultdict(set)
    for key in keys:
        stem, _, name = key[len(prefix):].rpartition("/")
        if stem and name.endswith(".mp4"):
            have[stem].add(name[: -len(".mp4")])
    for stem in have:
        have[stem].add("clean")
    return dict(have)


# ------------------------------------------------------------------- scanning

def scan(prefixes=None, datasets=None):
    """-> (missing_renders, to_rerun, unparsed).

    missing_renders  {dataset: {stem: [variant]}}
    to_rerun         {(prefix, judge, dataset): {"clips": n, "calls": n,
                                                 "variants": Counter}}
    unparsed         [(key, judge, prefix, dataset, stem, variant, call, raw)]
    """
    prefixes = prefixes or discover_runs()
    datasets = datasets or DATASETS

    rendered = {ds: rendered_variants(ds) for ds in datasets}
    missing_renders = {}
    for ds in datasets:
        gaps = {stem: sorted(set(VARIANTS) - got)
                for stem, got in rendered[ds].items()
                if set(VARIANTS) - got}
        if gaps:
            missing_renders[ds] = gaps

    print(f"scanning {len(prefixes)} prefix(es): {', '.join(prefixes)}")
    n_records = n_calls = 0

    to_rerun, unparsed = {}, []
    for prefix in prefixes:
        for judge in judges_for(prefix):
            call_ids = CALLS[judge]
            for ds in datasets:
                keys, _ = _list(f"{prefix}/{judge}/{ds}/")
                keys = [k for k in keys if k.endswith(".json")]
                if not keys:
                    continue
                with ThreadPoolExecutor(READ_WORKERS) as ex:
                    recs = list(ex.map(lambda k: (k, _get(k)), keys))
                recs = [(k, r) for k, r in recs if r]

                # a subset run covers only what it was asked for; infer it the
                # same way audit_runs does so a deliberate subset is not a gap
                seen = Counter(v for _, r in recs for v in r.get("runs", {}))
                top = max(seen.values()) if seen else 0
                want = [v for v in VARIANTS if seen.get(v, 0) >= 0.5 * top]

                n_records += len(recs)
                clips, calls, per_variant = 0, 0, Counter()
                for key, rec in recs:
                    stem = rec.get("clip", "?")
                    runs = rec.get("runs", {})
                    got = rendered[ds].get(stem)
                    gap = 0
                    for v in want:
                        if got is not None and v not in got:
                            continue           # no_render: attack_suite's job
                        done = runs.get(v, {}).get("calls", {})
                        for c in call_ids:
                            if c in done:
                                n_calls += 1
                                if done[c].get("parsed") is None:
                                    unparsed.append(
                                        (key, judge, prefix, ds, stem, v, c,
                                         done[c].get("raw")))
                            else:
                                gap += 1
                                per_variant[v] += 1
                    if gap:
                        clips += 1
                        calls += gap
                if calls:
                    to_rerun[(prefix, judge, ds)] = {
                        "clips": clips, "calls": calls,
                        "variants": per_variant, "want": want}

    print(f"  read {n_records} records, {n_calls} stored calls -> "
          f"{len(unparsed)} unparsed, "
          f"{sum(v['calls'] for v in to_rerun.values())} calls to re-judge, "
          f"{sum(len(g) for g in missing_renders.values())} clips missing a render")
    if n_records == 0:
        print("  READ NOTHING -- wrong bucket, no credentials, or the result "
              "prefixes are not where this expects them")
    return missing_renders, to_rerun, unparsed


# -------------------------------------------------------------------- reparse

def repair_score(raw, call_id):
    """-> (score, note) or (None, why). Conservative by construction.

    Accepts exactly one shape: a JSON object carrying a single numeric value,
    under a key that is not the one asked for. That is the observed failure --
    the model renames the key after the video's subject -- and the value is
    unambiguous because each call asks about exactly one criterion. Anything
    else (no object, several numbers, a value off phyjudge's 1-5 scale) is
    refused: an unparsed call is visible in `unparsed`, a wrongly repaired one
    is invisible forever.
    """
    if not raw:
        return None, "empty raw"
    m = _JSON_OBJ.search(raw)
    if not m:
        return None, "no JSON object in the reply"
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        return None, f"JSON did not parse ({e.__class__.__name__})"
    if not isinstance(obj, dict):
        return None, "JSON is not an object"

    nums = {k: v for k, v in obj.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if len(nums) != 1:
        return None, f"{len(nums)} numeric values, need exactly 1"
    key, val = next(iter(nums.items()))
    if key == call_id:
        return None, "the key is already correct -- a different failure"
    if float(val) != int(val):
        return None, f"non-integer score {val!r}"
    val = int(val)
    if not PHYJUDGE_MIN <= val <= PHYJUDGE_MAX:
        return None, f"score {val} outside {PHYJUDGE_MIN}-{PHYJUDGE_MAX}"
    return val, f"key {key!r} -> {call_id!r}"


def reparse(unparsed=None, write=False, prefixes=None, datasets=None):
    """Recover a score from stored raw text. Dry-run unless write=True.

    Writes back `parsed` and appends to a `repaired` list on the record naming
    the (variant, call) and the key the model actually used, so the change is
    auditable and reversible; `unparsed` is refreshed the way the harness does.
    """
    if unparsed is None:
        _, _, unparsed = scan(prefixes, datasets)
    if not unparsed:
        print("no unparsed calls found")
        return []

    fixed, refused = [], []
    for key, judge, prefix, ds, stem, variant, call, raw in unparsed:
        if judge != "phyjudge_9b":
            refused.append((key, variant, call, "not phyjudge; no rule for it"))
            continue
        score, note = repair_score(raw, call)
        (fixed if score is not None else refused).append(
            (key, variant, call, score if score is not None else note, note))

    print(f"\n{len(fixed)} repairable, {len(refused)} refused "
          f"({'WRITING' if write else 'dry run -- pass --write to apply'})")
    for key, variant, call, score, note in fixed:
        print(f"  {key.rsplit('/', 1)[-1][:44]:<46} {variant}/{call} "
              f"-> {score}   ({note})")
    for row in refused:
        print(f"  REFUSED {row[0].rsplit('/', 1)[-1][:44]:<38} "
              f"{row[1]}/{row[2]}: {row[3]}")

    if not write:
        return fixed

    by_key = defaultdict(list)
    for key, variant, call, score, note in fixed:
        by_key[key].append((variant, call, score, note))
    for key, rows in by_key.items():
        rec = _get(key)
        if rec is None:
            print(f"  FAILED to re-read {key}")
            continue
        repaired = rec.setdefault("repaired", [])
        for variant, call, score, note in rows:
            entry = rec["runs"][variant]["calls"][call]
            if entry.get("parsed") is not None:
                continue              # someone else fixed it since the scan
            entry["parsed"] = score
            repaired.append({"variant": variant, "call": call,
                             "score": score, "note": note})
        rec["unparsed"] = [
            f"{v}/{c}" for v, run in rec["runs"].items()
            for c, out in run["calls"].items() if out.get("parsed") is None]
        _put(key, rec)
        print(f"  wrote {key}")
    return fixed


# --------------------------------------------------------------------- render

def render(missing_renders=None, datasets=None, num_workers=4):
    """Re-run attack_suite for the datasets that are missing a variant.

    run_suite takes no stem filter, and does not need one: it is idempotent by
    head_object, so a whole-corpus pass renders only the absent keys. Needs
    ffmpeg, cv2 and the DejaVu font -- i.e. Colab or a Linux box, not Windows.
    """
    if missing_renders is None:
        missing_renders, _, _ = scan(datasets=datasets or DATASETS)
    if not missing_renders:
        print("every clip has all 10 variants; nothing to render")
        return
    from attack_suite import run_suite
    for ds, gaps in missing_renders.items():
        n = sum(len(v) for v in gaps.values())
        print(f"\n=== {ds}: {n} missing renders across {len(gaps)} clips ===")
        for stem, variants in sorted(gaps.items()):
            print(f"  {stem[:60]}  {', '.join(variants)}")
        run_suite(dataset=ds, limit_clips=None, num_workers=num_workers)
    print("\nCheck stdout above for FAILED -- a render that failed once "
          "usually fails again for the same reason (a very short clip, a "
          "codec the decoder rejects), and a zero exit code does not mean it "
          "rendered. Re-run the audit to confirm before re-judging.")


# ----------------------------------------------------------------------- plan

def plan(prefixes=None, datasets=None):
    missing_renders, to_rerun, unparsed = scan(prefixes, datasets)

    print("=" * 78)
    print("1. MISSING RENDERS -- attack_suite, and BEFORE any re-judge")
    print("=" * 78)
    if not missing_renders:
        print("  none")
    for ds, gaps in missing_renders.items():
        for stem, variants in sorted(gaps.items()):
            print(f"  {ds:<28} {stem[:44]:<46} {', '.join(variants)}")
    if missing_renders:
        print("\n  python fix_runs.py --render        # then re-judge below")

    print("\n" + "=" * 78)
    print("2. UNPARSED -- re-parse the stored raw, do NOT re-judge")
    print("=" * 78)
    if not unparsed:
        print("  none")
    else:
        per_call = Counter(row[6] for row in unparsed)
        print(f"  {len(unparsed)} calls: "
              + ", ".join(f"{c}({n})" for c, n in per_call.most_common()))
        print("  phyjudge decodes greedily, so a rerun reproduces the same "
              "tokens and the same None.")
        print("\n  python fix_runs.py --reparse            # dry run")
        print("  python fix_runs.py --reparse --write")

    print("\n" + "=" * 78)
    print("3. RE-JUDGE -- one judge per process, on the box with that stack")
    print("=" * 78)
    if not to_rerun:
        print("  none")
    for (prefix, judge, ds), info in sorted(to_rerun.items()):
        vs = ", ".join(f"{v}({n})" for v, n in info["variants"].most_common(4))
        print(f"\n  {prefix}/{judge}/{ds}")
        print(f"    {info['calls']} calls over {info['clips']} clip(s): {vs}")
        kw = rerun_kwargs(prefix, ds, info["want"])
        kw["models"] = [judge]
        args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        venv = VENVS[judge]
        print(f"    ~/venvs/{venv}/bin/python -u -c \"from judge_harness "
              f"import run_judges; run_judges({args})\"")
        if prefix.startswith(PASS2_PREFIX) and prefix != PASS2_PREFIX:
            override = prefix
            print(f"    # NOTE {prefix} is not a prefix run_judges writes on "
                  f"its own -- set it first:")
            print(f"    #   import judge_harness as jh; "
                  f"jh.PASS2_RESULT_PREFIX = {override!r}")

    print("\n" + "=" * 78)
    print("The reruns regenerate ONLY the missing calls -- checkpointing is per")
    print("(clip, variant, call), so a complete clip is skipped before its")
    print("video is downloaded. num_clips=None means the whole corpus, which")
    print("is what makes naming the clip unnecessary.")
    return missing_renders, to_rerun, unparsed


def in_notebook():
    try:
        __IPYTHON__          # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(description="repair what audit_runs found")
    ap.add_argument("--reparse", action="store_true",
                    help="recover scores from stored raw text")
    ap.add_argument("--write", action="store_true",
                    help="apply the re-parse (default is a dry run)")
    ap.add_argument("--render", action="store_true",
                    help="re-run attack_suite for the missing variants")
    ap.add_argument("--prefix", help="one run, e.g. results/pass1")
    ap.add_argument("--datasets", nargs="+", choices=DATASETS)
    a = ap.parse_args()
    pfx = [a.prefix] if a.prefix else None
    if a.reparse:
        reparse(write=a.write, prefixes=pfx, datasets=a.datasets)
    elif a.render:
        render(datasets=a.datasets)
    else:
        plan(prefixes=pfx, datasets=a.datasets)
