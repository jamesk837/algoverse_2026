"""Aggregate sanity stats over whatever Pass-1 has graded so far. Read-only.

    !wget -qO stats.py https://raw.githubusercontent.com/jamesk837/algoverse_2026/main/stats.py
    import stats; stats.report()

Three questions, in the order worth asking:

  1. How much is graded, is any of it unparseable, and what scale is each call
     actually on? The observed min/max is printed because only videophy2's
     scale is knowable from this repo -- phyjudge's parsing is upstream's
     `infer.parse_score`, which the harness does not reimplement.
  2. Does the judge track the human labels? Mean CLEAN score by human PC level
     for the physics group, and by human SA level for the SA group. Only
     videophy2_test carries paired human scores.
  3. Does the 2x2 taxonomy behave? Per-variant delta against clean, computed on
     the physics group. Temporal (shuffle/reverse/freeze) should push DOWN;
     superficial (caption_echo_*, photometric) should not move, and a rise is
     the score inflation the project exists to measure.

**Calls are grouped by construct, never pooled across constructs.** Only
videophy2_auto emits a PC at all. phyjudge emits SA, PTV, persistence and 13
named laws; vila emits one float instruction score plus seven booleans where
True means "no violation found". Averaging SA into a physics number and calling
the result comparable to a human PC is exactly the mistake this grouping
exists to prevent.

Compare WITHIN a judge, never across: the scales differ and are partly unknown.
vila decodes at temperature 0.7, so every one of its numbers is a single
stochastic draw. Directional check only -- this is not the study.
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

# duplicated from judge_harness.py -- edit together
VARIANTS = [
    "clean", "shuffle", "reverse", "freeze", "photometric",
    "caption_echo_rubric_vocab", "caption_echo_score_anchor_positive",
    "caption_echo_authoritative_claim", "caption_echo_score_anchor_negative",
    "caption_echo_control_irrelevant",
]
TEMPORAL = {"shuffle", "reverse", "freeze"}
PHYJUDGE_LAWS = [
    "gravity", "inertia", "momentum", "impenetrability", "collision", "material",
    "buoyancy", "displacement", "flow_dynamics", "boundary_interaction",
    "fluid_continuity", "reflection", "shadow",
]

# Which calls belong to which construct, per judge. A group is a (name,
# predicate) pair; PHYSICS_GROUP names the one the human PC is compared
# against, SA_GROUP the one the human SA is compared against.
GROUPS = {
    "phyjudge_9b": [
        ("SA", lambda c: c == "SA"),
        ("PTV", lambda c: c == "PTV"),
        ("persistence", lambda c: c == "persistence"),
        ("laws (13)", lambda c: c in PHYJUDGE_LAWS),
    ],
    "vila_ewm": [
        ("instruction", lambda c: c == "instruction"),
        ("physical_laws (bool)", lambda c: c.startswith("physical_laws_")),
        ("common_sense (bool)", lambda c: c.startswith("common_sense_")),
    ],
    "videophy2_auto": [
        ("SA", lambda c: c == "SA"),
        ("PC", lambda c: c == "PC"),
    ],
}
PHYSICS_GROUP = {
    "phyjudge_9b": "laws (13)",
    "vila_ewm": "physical_laws (bool)",
    "videophy2_auto": "PC",
}
SA_GROUP = {"phyjudge_9b": "SA", "videophy2_auto": "SA"}

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
    """Floats stay floats; bools become 1/0 (True = no violation found)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def human_labels():
    """stem -> (pc, sa) from the metadata CSV; {} if unreadable."""
    try:
        body = s3.get_object(Bucket=BUCKET, Key=HUMAN_CSV)["Body"].read()
    except Exception as exc:
        print("  (no human labels: %s)" % exc)
        return {}
    rows = list(csv.DictReader(io.StringIO(body.decode("utf-8", "replace"))))
    if not rows or HUMAN_ID not in rows[0]:
        print("  (no human labels: %s lacks a %r column)" % (HUMAN_CSV, HUMAN_ID))
        return {}
    out = {}
    for r in rows:
        stem = os.path.splitext(os.path.basename(r.get(HUMAN_ID) or ""))[0]
        try:
            out[stem] = (int(float(r["pc"])), int(float(r["sa"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load(judge, dataset, workers=32):
    keys = _keys("%s/%s/%s/" % (RESULT_PREFIX, judge, dataset))
    if not keys:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(keys))) as ex:
        return list(ex.map(_get, keys))


def group_scores(records, groups):
    """{group: {(stem, variant): mean}}, plus unparsed and total call counts."""
    out = {name: {} for name, _ in groups}
    unparsed = calls = 0
    for rec in records:
        stem = rec.get("clip")
        for variant, run in rec.get("runs", {}).items():
            buckets = defaultdict(list)
            for cid, res in run.get("calls", {}).items():
                calls += 1
                v = _numeric(res.get("parsed"))
                if v is None:
                    unparsed += 1
                    continue
                for name, match in groups:
                    if match(cid):
                        buckets[name].append(v)
                        break
            for name, vals in buckets.items():
                m = _mean(vals)
                if m is not None:
                    out[name][(stem, variant)] = m
    return out, unparsed, calls


def _by_level(clean, human, index, label):
    levels = defaultdict(list)
    for stem, v in clean.items():
        if stem in human:
            levels[human[stem][index]].append(v)
    if not levels:
        return
    print("    mean clean score by human %s:" % label)
    means = []
    for lvl in sorted(levels):
        m = _mean(levels[lvl])
        means.append(m)
        print("      %s=%d  n=%-5d %.3f" % (label, lvl, len(levels[lvl]), m))
    if len(means) > 1:
        rising = all(b >= a for a, b in zip(means, means[1:]))
        print("      span %.3f, monotonic: %s"
              % (max(means) - min(means), "yes" if rising else "no"))


def report(judges=None, datasets=None):
    judges = judges or JUDGES
    datasets = datasets or DATASETS
    human = human_labels()

    for dataset in datasets:
        header = False
        for judge in judges:
            records = load(judge, dataset)
            if not records:
                continue
            if not header:
                print("\n" + "=" * 74)
                print("== %s ==" % dataset)
                header = True

            groups = GROUPS.get(judge, [("all calls", lambda c: True)])
            scores, unparsed, calls = group_scores(records, groups)
            print("\n%s  --  %d clips, %d calls, %d unparsed"
                  % (judge, len(records), calls, unparsed))

            for name, _ in groups:
                vals = scores[name]
                clean = {s: v for (s, var), v in vals.items() if var == "clean"}
                if not clean:
                    print("  %-22s no clean scores yet" % name)
                    continue
                flat = list(clean.values())
                print("  %-22s clean n=%-5d mean %.3f  range %.2f-%.2f"
                      % (name, len(flat), _mean(flat), min(flat), max(flat)))
                if human and PHYSICS_GROUP.get(judge) == name:
                    _by_level(clean, human, 0, "pc")
                if human and SA_GROUP.get(judge) == name:
                    _by_level(clean, human, 1, "sa")

            # taxonomy, on the physics group only
            pname = PHYSICS_GROUP.get(judge)
            vals = scores.get(pname, {})
            base = _mean([v for (s, var), v in vals.items() if var == "clean"])
            if base is None:
                continue
            print("  variant deltas on %s (temporal should DROP, "
                  "superficial should not move):" % pname)
            seen = {var for (_, var) in vals}
            order = [v for v in VARIANTS if v in seen] + sorted(seen - set(VARIANTS))
            for var in order:
                xs = [v for (s, vr), v in vals.items() if vr == var]
                m = _mean(xs)
                if m is None:
                    continue
                if var == "clean":
                    kind, flag = "baseline   ", ""
                elif var in TEMPORAL:
                    kind = "temporal   "
                    flag = "" if m - base < 0 else "  <- did NOT drop"
                else:
                    kind = "superficial"
                    flag = "  <- INFLATION" if m - base > 0.05 else ""
                print("    %-38s %s n=%-5d %6.3f  d=%+.3f%s"
                      % (var, kind, len(xs), m, m - base, flag))

    print("\n" + "=" * 74)
    print("Calls are grouped by construct and never pooled across them: only")
    print("videophy2 emits a PC. Bools count as 1/0. Compare WITHIN a judge,")
    print("never across -- scales differ and phyjudge's is set by upstream's")
    print("parse_score. Directional sanity check, not a result.")


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
