"""Aggregate sanity stats over whatever Pass-1 has graded so far. Read-only.

    !wget -qO stats.py https://raw.githubusercontent.com/jamesk837/algoverse_2026/main/stats.py
    import stats; stats.report()

Three questions, in the order worth asking:

  1. How much is graded, and is any of it unparseable?
  2. Does the judge track the human label at all? -- mean score on the CLEAN
     clip grouped by the human PC level 1-5, for videophy2_test, which is the
     only corpus with paired human scores. A judge that works should climb
     across those five rows. This is the "is it roughly working" check.
  3. Does the 2x2 taxonomy behave? -- per-variant mean and its delta against
     clean. Temporal (shuffle/reverse/freeze) should push DOWN; superficial
     (caption_echo_*, photometric) should not move, and a rise is the score
     inflation the project exists to measure.

Scores are pooled per (clip, variant) as the mean of that call's parsed values,
with booleans counted as 1/0. That is a crude summary and deliberately so: the
judges use different scales and different numbers of calls, so this is a
directional check, never a result. vila in particular decodes at temperature
0.7, so every one of its numbers is a single stochastic draw.

Nothing here is the study. It answers "is the pipeline producing plausible
numbers", not "what did we find".
"""

import csv
import io
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3

try:
    from google.colab import userdata
except ImportError:
    userdata = None

if userdata is not None:
    os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

BUCKET = "nickb-aarj"
RESULT_PREFIX = "results/pass1"
JUDGES = ["phyjudge_9b", "vila_ewm", "videophy2_auto"]
DATASETS = ["test", "implausibench_real", "implausibench_implausible"]

# only videophy2_test carries paired human pc/sa
HUMAN_CSV = "datasets/videophy2_test/_metadata/videophy2_test.csv"
HUMAN_ID = "video_url"

# duplicated from judge_harness.py -- display order; edit together
VARIANTS = [
    "clean", "shuffle", "reverse", "freeze", "photometric",
    "caption_echo_rubric_vocab", "caption_echo_score_anchor_positive",
    "caption_echo_authoritative_claim", "caption_echo_score_anchor_negative",
    "caption_echo_control_irrelevant",
]
TEMPORAL = {"shuffle", "reverse", "freeze"}

s3 = boto3.client("s3")


def _keys(prefix):
    out, pg = [], s3.get_paginator("list_objects_v2")
    for page in pg.paginate(Bucket=BUCKET, Prefix=prefix):
        out += [o["Key"] for o in page.get("Contents", [])
                if o["Key"].endswith(".json")]
    return out


def _get(key):
    return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())


def _numeric(value):
    """Parsed values are floats or bools; bools mean 'no violation found'."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def human_labels():
    """stem -> pc, from the metadata CSV. {} if it is not readable."""
    try:
        body = s3.get_object(Bucket=BUCKET, Key=HUMAN_CSV)["Body"].read()
    except Exception as exc:
        print("  (no human labels: %s)" % exc)
        return {}
    rows = list(csv.DictReader(io.StringIO(body.decode("utf-8", "replace"))))
    if not rows or HUMAN_ID not in rows[0]:
        print("  (no human labels: %s has no %r column)" % (HUMAN_CSV, HUMAN_ID))
        return {}
    out = {}
    for r in rows:
        stem = os.path.splitext(os.path.basename(r[HUMAN_ID] or ""))[0]
        try:
            out[stem] = int(float(r["pc"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load(judge, dataset, workers=32):
    keys = _keys("%s/%s/%s/" % (RESULT_PREFIX, judge, dataset))
    if not keys:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(keys))) as ex:
        return list(ex.map(_get, keys))


def _per_variant(records):
    """(stem, variant) -> pooled score, plus an unparsed count."""
    scores, unparsed, calls = {}, 0, 0
    for rec in records:
        stem = rec.get("clip")
        for variant, run in rec.get("runs", {}).items():
            vals = []
            for out in run.get("calls", {}).values():
                calls += 1
                v = _numeric(out.get("parsed"))
                if v is None:
                    unparsed += 1
                else:
                    vals.append(v)
            m = _mean(vals)
            if m is not None:
                scores[(stem, variant)] = m
    return scores, unparsed, calls


def report(judges=None, datasets=None):
    judges = judges or JUDGES
    datasets = datasets or DATASETS
    human = human_labels()

    for dataset in datasets:
        printed_header = False
        for judge in judges:
            records = load(judge, dataset)
            if not records:
                continue
            if not printed_header:
                print("\n" + "=" * 72)
                print("== %s ==" % dataset)
                printed_header = True

            scores, unparsed, calls = _per_variant(records)
            clean = {s: v for (s, var), v in scores.items() if var == "clean"}
            print("\n%s  --  %d clips, %d calls, %d unparsed"
                  % (judge, len(records), calls, unparsed))
            if not clean:
                print("  no clean scores yet")
                continue
            print("  mean clean score: %.3f  (n=%d)"
                  % (_mean(clean.values()), len(clean)))

            # 2. does it track the human label at all
            if human:
                by_level = defaultdict(list)
                for stem, v in clean.items():
                    if stem in human:
                        by_level[human[stem]].append(v)
                if by_level:
                    print("  clean score by human PC level:")
                    means = []
                    for lvl in sorted(by_level):
                        m = _mean(by_level[lvl])
                        means.append(m)
                        print("    pc=%d  n=%-5d mean %.3f"
                              % (lvl, len(by_level[lvl]), m))
                    if len(means) > 1:
                        rising = all(b >= a for a, b in zip(means, means[1:]))
                        print("    span %.3f, monotonic: %s"
                              % (max(means) - min(means), "yes" if rising else "no"))
                    print("    (a working judge climbs across these rows;"
                          " a flat column means it is not tracking the label)")

            # 3. does the taxonomy behave
            base = _mean(clean.values())
            seen = {var for (_, var) in scores}
            order = [v for v in VARIANTS if v in seen]
            order += sorted(seen - set(VARIANTS))
            print("  by variant (delta vs clean):")
            for var in order:
                vals = [v for (s, vr), v in scores.items() if vr == var]
                m = _mean(vals)
                if m is None:
                    continue
                if var == "clean":
                    kind, flag = "baseline   ", ""
                elif var in TEMPORAL:
                    kind = "temporal   "
                    flag = "" if m - base < 0 else "   <- expected to DROP"
                else:
                    kind = "superficial"
                    flag = "   <- INFLATION" if m - base > 0.05 else ""
                print("    %-38s %s n=%-5d %6.3f  d=%+.3f%s"
                      % (var, kind, len(vals), m, m - base, flag))

    print("\n" + "=" * 72)
    print("Pooled score = mean of a call's parsed values, bools as 1/0. The")
    print("judges use different scales, so compare WITHIN a judge, never")
    print("across. Directional sanity check only -- not a result.")


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


if __name__ == "__main__" and not in_notebook():
    import argparse

    ap = argparse.ArgumentParser(description="Pass-1 sanity stats")
    ap.add_argument("--judges", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args()
    report(judges=a.judges, datasets=a.datasets)
