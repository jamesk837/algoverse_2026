"""Inspect Pass-1 judge results already in S3. Read-only.

Paste the whole file into a Colab cell and call check(), or run it as a CLI on
any box with credentials. It loads no models and imports no torch, so it is
cheap to run anywhere -- the point is to look at what the harness wrote without
standing up a judge stack.

    check()                          # every judge, every clip, on `test`
    check(clip="some_clip_stem")     # one clip, full variant x call table
    check(models=["phyjudge_9b"])    # one judge
    coverage()                       # how far along each judge is
    raw("clip_stem", "phyjudge_9b", "shuffle", "gravity")   # one raw output

Deliberately self-contained -- no import from judge_harness -- because the
workflow is pasting one file into one Colab cell. The cost is that VARIANTS and
RESULT_PREFIX are duplicated from there and must be edited in both together.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3

# Colab hands credentials through userdata; everywhere else boto3's ambient
# chain does. Same tolerant pattern as judge_harness.py.
try:
    from google.colab import userdata
except ImportError:
    userdata = None

if userdata is not None:
    os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('AWS_ACCESS_KEY_ID')
    os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')

BUCKET = "nickb-aarj"
RESULT_PREFIX = "results/pass1"
s3 = boto3.client('s3')

# duplicated from judge_harness.py -- display order only; anything the records
# carry that is not listed here still shows, appended at the end
VARIANTS = [
    "clean", "shuffle", "reverse", "freeze", "photometric",
    "caption_echo_rubric_vocab", "caption_echo_score_anchor_positive",
    "caption_echo_authoritative_claim", "caption_echo_score_anchor_negative",
    "caption_echo_control_irrelevant",
]
TEMPORAL = {"shuffle", "reverse", "freeze"}
JUDGE_NAMES = ["vila_ewm", "videophy2_auto", "phyjudge_9b"]
DATASETS = ["test", "implausibench_real", "implausibench_implausible"]
CALLS_EXPECTED = {"vila_ewm": 8, "videophy2_auto": 2, "phyjudge_9b": 16}


def _list(prefix):
    keys, paginator = [], s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                keys.append(obj["Key"])
    return keys


def _get(key):
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return json.loads(body)


def load(dataset="test", models=None, clip=None, workers=32):
    """Fetch result records. Parallel because a full split is hundreds of GETs."""
    models = models or JUDGE_NAMES
    keys = []
    for model in models:
        found = _list(f"{RESULT_PREFIX}/{model}/{dataset}/")
        if clip:
            found = [k for k in found if k.rsplit("/", 1)[-1] == f"{clip}.json"]
        keys.extend(found)
    if not keys:
        where = f" for clip={clip!r}" if clip else ""
        print(f"no results under {RESULT_PREFIX}/*/{dataset}/{where}")
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(keys))) as ex:
        return list(ex.map(_get, keys))


def _variant_order(record):
    seen = list(record.get("runs", {}))
    known = [v for v in VARIANTS if v in seen]
    return known + [v for v in seen if v not in VARIANTS]


def frame(records):
    import pandas as pd

    rows = []
    for rec in records:
        for variant, run in rec.get("runs", {}).items():
            for cid, out in run.get("calls", {}).items():
                rows.append({
                    "clip": rec.get("clip"), "model": rec.get("model"),
                    "variant": variant, "call": cid,
                    "parsed": out.get("parsed"),
                    "raw": (out.get("raw") or "").replace("\n", " ")[:160],
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        order = VARIANTS + sorted(set(df["variant"]) - set(VARIANTS))
        df["variant"] = pd.Categorical(df["variant"], categories=order, ordered=True)
        df = df.sort_values(["clip", "model", "variant", "call"])
    return df.reset_index(drop=True)


def check(dataset="test", models=None, clip=None, show_raw=False):
    """Per (clip, judge): a variants-down, calls-across table of parsed scores."""
    import pandas as pd

    records = load(dataset=dataset, models=models, clip=clip)
    if not records:
        return None
    df = frame(records)
    if df.empty:
        print("records exist but contain no calls")
        return df

    try:
        from IPython.display import display
    except ImportError:
        display = print

    for (c, model), group in df.groupby(["clip", "model"], observed=True):
        expected = CALLS_EXPECTED.get(model)
        n_var = group["variant"].nunique()
        n_unparsed = int(group["parsed"].isna().sum())
        head = f"=== {c} | {model} | {n_var}/10 variants, {len(group)} calls"
        if expected:
            head += f" (expect {expected * 10})"
        if n_unparsed:
            head += f" | {n_unparsed} UNPARSED"
        print("\n" + head + " ===")

        table = group.pivot_table(index="variant", columns="call", values="parsed",
                                  observed=True, dropna=False)
        display(table.dropna(how="all"))

        # The taxonomy is the whole experiment: temporal variants should score
        # BELOW clean, superficial ones should sit on top of it. This is a
        # per-clip eyeball, not the study -- n=1 says nothing on its own.
        num = group.dropna(subset=["parsed"])
        if not num.empty and "clean" in set(num["variant"]):
            base = num[num["variant"] == "clean"]["parsed"].mean()
            print(f"  mean parsed by variant (clean = {base:.3f}):")
            for v in _variant_order(records[0]):
                sub = num[num["variant"] == v]
                if sub.empty:
                    continue
                m = sub["parsed"].mean()
                kind = "temporal   " if v in TEMPORAL else "superficial"
                if v == "clean":
                    kind = "baseline   "
                print(f"    {v:38s} {kind} {m:6.3f}  d={m - base:+.3f}")

        if show_raw:
            for _, r in group.iterrows():
                print(f"    [{r['variant']}/{r['call']}] {r['raw']}")

    total_unparsed = int(df["parsed"].isna().sum())
    print(f"\n{len(df)} calls, {total_unparsed} unparsed")
    if total_unparsed:
        print("unparsed calls keep their raw text -- inspect with show_raw=True")
    return df


def coverage(dataset=None, models=None):
    """How many clips each judge has finished, and how many are only partial.

    Covers all three phase-1 corpora by default, matching what run_shard.sh
    runs; pass a single dataset name to narrow it.
    """
    if dataset is None:
        for ds in DATASETS:
            print(f"\n== {ds} ==")
            coverage(dataset=ds, models=models)
        return
    models = models or JUDGE_NAMES
    for model in models:
        keys = _list(f"{RESULT_PREFIX}/{model}/{dataset}/")
        if not keys:
            print(f"{model:16s} nothing written yet")
            continue
        expected = CALLS_EXPECTED.get(model)
        with ThreadPoolExecutor(max_workers=32) as ex:
            recs = list(ex.map(_get, keys))
        complete = partial = 0
        unparsed = 0
        for rec in recs:
            runs = rec.get("runs", {})
            n_calls = sum(len(r.get("calls", {})) for r in runs.values())
            unparsed += sum(1 for r in runs.values()
                            for o in r.get("calls", {}).values()
                            if o.get("parsed") is None)
            if expected and len(runs) == 10 and n_calls == expected * 10:
                complete += 1
            else:
                partial += 1
        note = f", {partial} partial" if partial else ""
        print(f"{model:16s} {len(recs):5d} clips written, {complete} complete{note}, "
              f"{unparsed} unparsed calls")


def raw(clip, model, variant, call, dataset="test"):
    """The full untruncated model output for one call."""
    rec = _get(f"{RESULT_PREFIX}/{model}/{dataset}/{clip}.json")
    out = rec["runs"][variant]["calls"][call]
    print(f"--- {clip} | {model} | {variant} | {call} -> parsed={out.get('parsed')!r}")
    print(out.get("raw"))
    return out


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


if __name__ == "__main__" and not in_notebook():
    import argparse

    ap = argparse.ArgumentParser(description="inspect Pass-1 results in S3")
    ap.add_argument("--dataset", default=None,
                    help="one of %s; default: all three for --coverage" % DATASETS)
    ap.add_argument("--clip", default=None, help="one clip stem")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--raw", action="store_true", help="print raw outputs too")
    ap.add_argument("--coverage", action="store_true", help="progress only")
    a = ap.parse_args()

    if a.coverage:
        coverage(dataset=a.dataset, models=a.models)
    else:
        check(dataset=a.dataset or "test", models=a.models, clip=a.clip, show_raw=a.raw)
