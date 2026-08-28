"""Main-benchmark analysis over the completed judge runs (research doc sec. 6).

Reads analysis/data/{pass1,paraphrase/p0,paraphrase/p1}/<judge>/<dataset>/*.json
and videophy2_test.csv, writes report.txt + deltas.csv.

Per judge x dataset x construct group (grouping copied from stats.py -- calls
are never pooled across constructs, comparisons stay within a judge):
  1. paired attack effects: per-clip delta (variant - clean) of the group
     mean; signed mean, mean |d|, paired-bootstrap 95% CI (2000 resamples)
  2. gameability gap: temporal signed delta (should be < 0) vs superficial
     signed delta (should be ~ 0; CI entirely > 0 = inflation)
  3. judge prompt robustness: clean-score agreement across pass1/p0/p1
     (Spearman + MAD) and the spread of each variant's signed delta across
     the three prompts
  4. human tracking (videophy2 test only): mean clean group score by human
     PC / SA level + Spearman
"""
import csv, glob, json, os, sys
from collections import defaultdict

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RUNS = {"pass1": "pass1", "p0": "paraphrase/p0", "p1": "paraphrase/p1"}
JUDGES = ["phyjudge_9b", "vila_ewm", "videophy2_auto"]
DATASETS = ["test", "implausibench_real", "implausibench_implausible"]
VARIANTS = ["clean", "shuffle", "reverse", "freeze", "photometric",
            "caption_echo_rubric_vocab", "caption_echo_score_anchor_positive",
            "caption_echo_authoritative_claim", "caption_echo_score_anchor_negative",
            "caption_echo_control_irrelevant"]
TEMPORAL = {"shuffle", "reverse", "freeze"}
SUPERFICIAL = [v for v in VARIANTS if v != "clean" and v not in TEMPORAL]
PHYJUDGE_LAWS = ["gravity", "inertia", "momentum", "impenetrability", "collision",
                 "material", "buoyancy", "displacement", "flow_dynamics",
                 "boundary_interaction", "fluid_continuity", "reflection", "shadow"]
GROUPS = {
    "phyjudge_9b": {"SA": lambda c: c == "SA", "PTV": lambda c: c == "PTV",
                    "laws13": lambda c: c in PHYJUDGE_LAWS},
    "vila_ewm": {"instruction": lambda c: c == "instruction",
                 "physical_laws_bool": lambda c: c.startswith("physical_laws_"),
                 "common_sense_bool": lambda c: c.startswith("common_sense_")},
    "videophy2_auto": {"SA": lambda c: c == "SA", "PC": lambda c: c == "PC"},
}
PHYSICS_GROUP = {"phyjudge_9b": "laws13", "vila_ewm": "physical_laws_bool",
                 "videophy2_auto": "PC"}
SA_GROUP = {"phyjudge_9b": "SA", "videophy2_auto": "SA"}
RNG = np.random.default_rng(0)
B = 2000


def numeric(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def load(run, judge, ds):
    """clip -> variant -> group -> mean numeric score."""
    out = {}
    for f in glob.glob(os.path.join(DATA, RUNS[run], judge, ds, "*.json")):
        d = json.load(open(f))
        per = {}
        for var, r in d.get("runs", {}).items():
            calls = r.get("calls", {})
            g = {}
            for gname, pred in GROUPS[judge].items():
                xs = [numeric(c.get("parsed")) for k, c in calls.items() if pred(k)]
                xs = [x for x in xs if x is not None]
                if xs:
                    g[gname] = float(np.mean(xs))
            per[var] = g
        out[d["clip"]] = per
    return out


def boot_ci(deltas):
    if len(deltas) < 2:
        return (float("nan"), float("nan"))
    a = np.array(deltas)
    idx = RNG.integers(0, len(a), (B, len(a)))
    means = a[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_deltas(clips, group, variant):
    ds = []
    for per in clips.values():
        c, v = per.get("clean", {}).get(group), per.get(variant, {}).get(group)
        if c is not None and v is not None:
            ds.append(v - c)
    return ds


def human_labels():
    labs = {}
    with open(os.path.join(DATA, "videophy2_test.csv")) as fh:
        for row in csv.DictReader(fh):
            stem = os.path.splitext(os.path.basename(row["video_url"]))[0]
            try:
                labs[stem] = (int(row["pc"]), int(row["sa"]))
            except (ValueError, KeyError):
                pass
    return labs


def main():
    rep, rows = [], []
    P = rep.append
    cache = {run: {(j, ds): load(run, j, ds) for j in JUDGES for ds in DATASETS}
             for run in RUNS}
    labs = human_labels()

    for judge in JUDGES:
        P(f"\n{'#'*78}\n# {judge}\n{'#'*78}")
        for ds in DATASETS:
            clips = cache["pass1"][(judge, ds)]
            P(f"\n== {ds}  (n={len(clips)} clips) ==")
            for group in GROUPS[judge]:
                P(f"\n  -- {group} --")
                P(f"  {'variant':38s} {'n':>4s} {'signed d':>9s} {'95% CI':>20s} {'mean|d|':>8s}")
                for var in VARIANTS[1:]:
                    d = paired_deltas(clips, group, var)
                    if not d:
                        continue
                    lo, hi = boot_ci(d)
                    a = np.array(d)
                    kind = "temporal" if var in TEMPORAL else "superficial"
                    flag = ""
                    if var in TEMPORAL and hi < 0:
                        flag = "  SENSITIVE"
                    if var not in TEMPORAL and lo > 0:
                        flag = "  INFLATION"
                    if var not in TEMPORAL and hi < 0:
                        flag = "  deflation"
                    P(f"  {var:38s} {len(d):4d} {a.mean():+9.4f} [{lo:+8.4f},{hi:+8.4f}] {np.abs(a).mean():8.4f}{flag}")
                    rows.append(dict(judge=judge, dataset=ds, group=group, variant=var,
                                     kind=kind, n=len(d), signed_delta=a.mean(),
                                     ci_lo=lo, ci_hi=hi, mean_abs_delta=np.abs(a).mean()))
                # gameability gap for this group
                # judge-internal contrast -- NOT the doc's Step 12 gameability
                # gap (that is dJ - dV vs the V-JEPA reference, reported below)
                t = [x for v in TEMPORAL for x in paired_deltas(clips, group, v)]
                s = [x for v in SUPERFICIAL for x in paired_deltas(clips, group, v)]
                if t and s:
                    gap = np.mean(s) - np.mean(t)
                    P(f"  temporal-superficial contrast (judge-internal): temporal {np.mean(t):+.4f}  superficial {np.mean(s):+.4f}  contrast {gap:+.4f}")

        # human tracking, test only
        ds = "test"
        clips = cache["pass1"][(judge, ds)]
        for hname, gmap, hidx in [("PC", PHYSICS_GROUP, 0), ("SA", SA_GROUP, 1)]:
            if judge not in gmap:
                continue
            group = gmap[judge]
            pairs = [(labs[c][hidx], per["clean"][group]) for c, per in clips.items()
                     if c.removesuffix("_result") in labs or c in labs
                     for labs_c in [labs.get(c.removesuffix("_result"), labs.get(c))]
                     if per.get("clean", {}).get(group) is not None]
            # simpler: rebuild
            pairs = []
            for c, per in clips.items():
                stem = c.removesuffix("_result")
                lab = labs.get(stem) or labs.get(c)
                v = per.get("clean", {}).get(group)
                if lab and v is not None:
                    pairs.append((lab[hidx], v))
            if len(pairs) > 5:
                h, m = zip(*pairs)
                rho = sps.spearmanr(h, m).statistic
                P(f"\n  human {hname} vs clean {group} (test, n={len(pairs)}): spearman {rho:+.3f}")
                by = defaultdict(list)
                for hh, mm in pairs:
                    by[hh].append(mm)
                for lvl in sorted(by):
                    P(f"    human {hname}={lvl}  n={len(by[lvl]):3d}  mean judge {np.mean(by[lvl]):.3f}")

    # prompt robustness
    P(f"\n{'#'*78}\n# prompt robustness: pass1 (native) vs p0 / p1 (paraphrases)\n{'#'*78}")
    for judge in JUDGES:
        for ds in DATASETS:
            base = cache["pass1"][(judge, ds)]
            P(f"\n== {judge} / {ds} ==")
            for group in GROUPS[judge]:
                # clean-score agreement across prompts
                line = [f"  {group:20s}"]
                for run in ("p0", "p1"):
                    other = cache[run][(judge, ds)]
                    pairs = [(per["clean"][group], other[c]["clean"][group])
                             for c, per in base.items()
                             if c in other and per.get("clean", {}).get(group) is not None
                             and other[c].get("clean", {}).get(group) is not None]
                    if len(pairs) > 5:
                        a, b = map(np.array, zip(*pairs))
                        rho = sps.spearmanr(a, b).statistic
                        line.append(f"{run}: rho {rho:+.3f} MAD {np.abs(a-b).mean():.3f}")
                P(" | ".join(line))
                # does each variant's signed delta replicate across prompts?
                P(f"    {'variant':38s} {'pass1':>8s} {'p0':>8s} {'p1':>8s} {'spread':>7s}")
                for var in VARIANTS[1:]:
                    ms = []
                    for run in RUNS:
                        d = paired_deltas(cache[run][(judge, ds)], group, var)
                        ms.append(np.mean(d) if d else float("nan"))
                    spread = np.nanmax(ms) - np.nanmin(ms)
                    P(f"    {var:38s} {ms[0]:+8.4f} {ms[1]:+8.4f} {ms[2]:+8.4f} {spread:7.4f}")

    # ---- Step 12 gameability gap: d = dJ - dV against the locked V-JEPA
    # reference (reference/probe_locked/dv.json), paired per clip x variant.
    # Both sides in normalised 0-1 units: dV ships d_norm; dJ is the physics-
    # group delta divided by its scale span (1-5 scales -> 4, booleans -> 1).
    dv_path = os.path.join(DATA, "dv.json")
    if os.path.exists(dv_path):
        dv = json.load(open(dv_path))["datasets"]
        DS_MAP = {"test": "videophy2_test", "implausibench_real": "implausibench_real",
                  "implausibench_implausible": "implausibench_implausible"}
        SPAN = {"phyjudge_9b": 4.0, "vila_ewm": 1.0, "videophy2_auto": 4.0}
        P(f"\n{'#'*78}\n# Step 12 gameability gap: d = dJ - dV  (normalised units, physics group)\n"
          f"# dJ from pass1, dV from the locked V-JEPA probe; positive d on a\n"
          f"# superficial variant = the judge moved when the reference did not\n{'#'*78}")
        for judge in JUDGES:
            group, span = PHYSICS_GROUP[judge], SPAN[judge]
            for ds in DATASETS:
                clips = cache["pass1"][(judge, ds)]
                ref = dv.get(DS_MAP[ds], {}).get("clips", {})
                P(f"\n== {judge} / {ds} ==")
                P(f"  {'variant':38s} {'n':>4s} {'dJ':>8s} {'dV':>8s} {'d=dJ-dV':>9s} {'95% CI':>20s}")
                for var in VARIANTS[1:]:
                    gaps = []
                    for c, per in clips.items():
                        jc = per.get("clean", {}).get(group)
                        jv = per.get(var, {}).get(group)
                        rv = ref.get(c, {}).get("variants", {}).get(var)
                        if jc is None or jv is None or rv is None:
                            continue
                        gaps.append(((jv - jc) / span) - rv["d_norm"])
                    if len(gaps) < 5:
                        continue
                    a = np.array(gaps)
                    lo, hi = boot_ci(gaps)
                    dj = np.mean([(per[var][group] - per["clean"][group]) / span
                                  for c, per in clips.items()
                                  if per.get(var, {}).get(group) is not None
                                  and per.get("clean", {}).get(group) is not None
                                  and c in ref and var in ref[c].get("variants", {})])
                    dvm = np.mean([ref[c]["variants"][var]["d_norm"] for c in clips
                                   if c in ref and var in ref[c].get("variants", {})])
                    flag = "  GAP>0" if lo > 0 else ("  GAP<0" if hi < 0 else "")
                    P(f"  {var:38s} {len(a):4d} {dj:+8.4f} {dvm:+8.4f} {a.mean():+9.4f} [{lo:+8.4f},{hi:+8.4f}]{flag}")
                    rows.append(dict(judge=judge, dataset=ds, group=f"step12_{group}",
                                     variant=var,
                                     kind="temporal" if var in TEMPORAL else "superficial",
                                     n=len(a), signed_delta=a.mean(), ci_lo=lo, ci_hi=hi,
                                     mean_abs_delta=np.abs(a).mean()))
    else:
        P("\n(no data/dv.json -- Step 12 dJ-dV section skipped)")

    txt = "\n".join(rep)
    open(os.path.join(HERE, "report.txt"), "w").write(txt)
    with open(os.path.join(HERE, "deltas.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(txt)
    print(f"\nwrote analysis/report.txt and analysis/deltas.csv ({len(rows)} delta rows)")


if __name__ == "__main__":
    main()
