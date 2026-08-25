"""Live progress of a sharded Pass-1 run, broken down by shard. Read-only.

    !wget -qO monitor.py https://raw.githubusercontent.com/jamesk837/algoverse_2026/main/monitor.py
    import monitor; monitor.progress(shards=4)

Answers "is every machine still working", which coverage() cannot: it
recomputes each shard's stripe exactly the way the harness does -- sorted
source listing, filtered to clips with a rendered attack directory, then
[i::n] -- so a row maps to one specific machine.

LIST calls only, no GETs, so it is cheap to re-run while a fleet is going.
The cost is that a result file appears as soon as a clip's FIRST variant is
scored, so `written` slightly overstates completion. Use check_results for a
strict answer; this is for liveness.
"""

import datetime
import os

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
JUDGES = ["phyjudge_9b", "vila_ewm", "videophy2_auto"]

# duplicated from judge_harness.DATASET_PREFIXES; edit both together
DATASET_PREFIXES = {
    "test": "datasets/videophy2_test/",
    "implausibench_real": "datasets/implausibench/ImplausiBench/real/",
    "implausibench_implausible":
        "datasets/implausibench/ImplausiBench/implausible/",
}
VIDEO_SUFFIXES = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

# a shard is called stalled if its newest write is older than this
ACTIVE_MINUTES = 15

s3 = boto3.client("s3")


def _objects(prefix):
    out, pg = [], s3.get_paginator("list_objects_v2")
    for page in pg.paginate(Bucket=BUCKET, Prefix=prefix):
        out += page.get("Contents", [])
    return out


def _source_stems(dataset):
    keys = [o["Key"] for o in _objects(DATASET_PREFIXES[dataset])
            if os.path.splitext(o["Key"])[1].lower() in VIDEO_SUFFIXES]
    keys.sort()          # the harness sorts too; the stripe depends on it
    return [os.path.splitext(os.path.basename(k))[0] for k in keys]


def _attacked(dataset):
    """Clip stems with a rendered variant directory, via one Delimiter LIST."""
    prefix = "attacks/%s/" % dataset
    out, pg = set(), s3.get_paginator("list_objects_v2")
    for page in pg.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            stem = cp["Prefix"][len(prefix):].strip("/")
            if stem:
                out.add(stem)
    return out


def _written(judge, dataset):
    out = {}
    for o in _objects("results/pass1/%s/%s/" % (judge, dataset)):
        if o["Key"].endswith(".json"):
            out[os.path.basename(o["Key"])[:-5]] = o["LastModified"]
    return out


def _age(minutes):
    if minutes < 90:
        return "%.0fm" % minutes
    if minutes < 48 * 60:
        return "%.1fh" % (minutes / 60)
    return "%.1fd" % (minutes / 1440)


def progress(shards=4, datasets=None, judges=None):
    """Per (dataset, judge, shard): clips written, last write, liveness."""
    datasets = datasets or list(DATASET_PREFIXES)
    judges = judges or JUDGES
    now = datetime.datetime.now(datetime.timezone.utc)

    for dataset in datasets:
        have = _attacked(dataset)
        stems = [s for s in _source_stems(dataset) if s in have]
        print("")
        if not stems:
            print("== %s ==  nothing rendered under attacks/, skipped" % dataset)
            continue
        print("== %s ==  %d clips with attacks" % (dataset, len(stems)))
        print("  %-16s %5s %11s %7s  %s"
              % ("judge", "shard", "written", "last", "status"))
        for judge in judges:
            seen = _written(judge, dataset)
            for i in range(shards):
                mine = set(stems[i::shards])
                done = mine & seen.keys()
                count = "%d/%d" % (len(done), len(mine))
                if not done:
                    age, status = "-", "not started"
                else:
                    mins = (now - max(seen[s] for s in done)).total_seconds() / 60
                    age = _age(mins)
                    if len(done) == len(mine):
                        status = "COMPLETE"
                    elif mins < ACTIVE_MINUTES:
                        status = "ACTIVE"
                    else:
                        status = "STALLED?"
                print("  %-16s %5d %11s %7s  %s"
                      % (judge, i, count, age, status))


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


if __name__ == "__main__" and not in_notebook():
    import argparse

    ap = argparse.ArgumentParser(description="sharded Pass-1 progress")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--judges", nargs="*", default=None)
    a = ap.parse_args()
    progress(shards=a.shards, datasets=a.datasets, judges=a.judges)
