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
DATA = os.environ.get("ANALYZE_DATA", os.path.join(HERE, "data"))
RUNS = {"pass1": "pass1", "p0": "paraphrase/p0", "p1": "paraphrase/p1"}
JUDGES = ["phyjudge_9b", "vila_ewm", "videophy2_auto"]
DATASETS = ["test", "implausibench_real", "implausibench_implausible"]
VARIANTS = ["clean", "shuffle", "reverse", "freeze", "photometric",
            "caption_echo_rubric_vocab", "caption_echo_score_anchor_positive",
            "caption_echo_authoritative_claim", "caption_echo_score_anchor_negative",
            "caption_echo_control_irrelevant"]
TEMPORAL = {"shuffle", "reverse", "freeze"}
SUPERFICIAL = [v for v in VARIANTS if v != "clean" and v not in TEMPORAL]
# ablation 11. NOT an attack and deliberately not in VARIANTS: `clean` is the
# source object and is never re-rendered, so every attacked variant carries one
# extra libx264 pass that clean does not. `identity` is that pass with nothing
# manipulated, so (variant - identity) is the attack effect net of the codec.
CONTROL = "identity"
# the three contrasts Step 6 pre-specifies, named as the doc names them. C2-C4
# asks whether the judge follows the DIRECTION of an injected score; C1-C5 and
# C3-C5 ask whether it reacts to evaluative language specifically or to overlaid
# text generally. Both are clauses of H2, not decoration.
CONTRASTS = [("C2-C4 directional anchor",
              "caption_echo_score_anchor_positive",
              "caption_echo_score_anchor_negative"),
             ("C1-C5 content specificity",
              "caption_echo_rubric_vocab", "caption_echo_control_irrelevant"),
             ("C3-C5 content specificity",
              "caption_echo_authoritative_claim",
              "caption_echo_control_irrelevant")]
# videophy2_test.csv is the only corpus carrying source-model provenance, and
# the column name is not pinned anywhere in this repo -- discover it rather
# than guess, and say which headers were seen if none matches.
GENERATOR_COLUMNS = ["model_name", "source_model", "generator", "model",
                     "video_model", "model_id"]
MIN_CLIPS_PER_GEN = 5
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


def paired_contrast(clips, group, va, vb):
    """Per clip, (va - clean) - (vb - clean), which reduces to va - vb.

    Kept as a difference of deltas conceptually, and computed only on clips
    carrying clean as well, so the contrast is over exactly the clips the
    per-variant table above reports. Pairing within a clip is what makes the
    bootstrap CI meaningful: the two overlays share the clip, so clip-level
    variance cancels instead of being resampled twice."""
    out = []
    for per in clips.values():
        c = per.get("clean", {}).get(group)
        a = per.get(va, {}).get(group)
        b = per.get(vb, {}).get(group)
        if c is not None and a is not None and b is not None:
            out.append(a - b)
    return out


def corrected_deltas(clips, group, variant):
    """Attack effect with the codec-only component removed (ablation 11).

    Both sides carry exactly one libx264 generation, so what is left is the
    manipulation. Paired per clip, same as the raw delta."""
    out = []
    for per in clips.values():
        i = per.get(CONTROL, {}).get(group)
        v = per.get(variant, {}).get(group)
        if i is not None and v is not None:
            out.append(v - i)
    return out


def generators():
    """-> (stem -> source generator, column name, headers seen)."""
    path = os.path.join(DATA, "videophy2_test.csv")
    if not os.path.exists(path):
        return {}, None, []
    with open(path) as fh:
        rd = csv.DictReader(fh)
        heads = list(rd.fieldnames or [])
        col = next((c for c in GENERATOR_COLUMNS if c in heads), None)
        if col is None:
            return {}, None, heads
        out = {}
        for row in rd:
            stem = os.path.splitext(os.path.basename(row["video_url"]))[0]
            g = (row.get(col) or "").strip()
            if g:
                out[stem] = g
    return out, col, heads


def rank_instability(clips, group, gen, variant):
    """Would a leaderboard built on this judge reorder under the attack?

    -> (kendall tau, pairwise generator flip rate, clip-level spearman,
    n generators, n clips), or None when there is too little to rank.

    Generators are ranked by MEAN CLEAN score and re-ranked under the attack.
    tau and the flip rate are leaderboard-level; the clip-level spearman is the
    finer question of whether individual clips keep their order."""
    by_c, by_v, pc, pv = defaultdict(list), defaultdict(list), [], []
    for stem, per in clips.items():
        g = gen.get(stem) or gen.get(stem.removesuffix("_result"))
        c = per.get("clean", {}).get(group)
        v = per.get(variant, {}).get(group)
        if c is None or v is None:
            continue
        pc.append(c)
        pv.append(v)
        if g:
            by_c[g].append(c)
            by_v[g].append(v)
    gens = sorted(g for g in by_c if len(by_c[g]) >= MIN_CLIPS_PER_GEN)
    if len(gens) < 3 or len(pc) < 5:
        return None
    mc = np.array([np.mean(by_c[g]) for g in gens])
    mv = np.array([np.mean(by_v[g]) for g in gens])
    flips = tot = 0
    for i in range(len(gens)):
        for j in range(i + 1, len(gens)):
            if mc[i] == mc[j]:
                continue          # tied on clean: no order to flip
            tot += 1
            if np.sign(mc[i] - mc[j]) != np.sign(mv[i] - mv[j]):
                flips += 1
    return (float(sps.kendalltau(mc, mv).statistic),
            (flips / tot) if tot else float("nan"),
            float(sps.spearmanr(pc, pv).statistic), len(gens), len(pc))


def main():
    rep, rows, contrast_rows, rank_rows = [], [], [], []
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
                # ablation 11: report raw AND identity-corrected. The codec
                # pass is worth about as much as the temporal effects are, so
                # a raw temporal delta cannot be read on its own.
                ctrl = paired_deltas(clips, group, CONTROL)
                head = f"  {'variant':38s} {'n':>4s} {'signed d':>9s} {'95% CI':>20s} {'mean|d|':>8s}"
                if ctrl:
                    head += f" {'d corr':>9s} {'95% CI corr':>20s}"
                P(head)
                if ctrl:
                    c = np.array(ctrl)
                    clo, chi = boot_ci(ctrl)
                    P(f"  {CONTROL + ' (codec control)':38s} {len(c):4d} "
                      f"{c.mean():+9.4f} [{clo:+8.4f},{chi:+8.4f}] "
                      f"{np.abs(c).mean():8.4f}   <- subtracted at right")
                for var in VARIANTS[1:]:
                    d = paired_deltas(clips, group, var)
                    if not d:
                        continue
                    lo, hi = boot_ci(d)
                    a = np.array(d)
                    kind = "temporal" if var in TEMPORAL else "superficial"
                    # the taxonomy verdict is read off the CORRECTED effect
                    # where there is one: it is the attack, not the encode
                    cd = corrected_deltas(clips, group, var) if ctrl else []
                    if cd:
                        ca = np.array(cd)
                        clo, chi = boot_ci(cd)
                        vlo, vhi = clo, chi
                    else:
                        ca, clo, chi = np.array([]), float("nan"), float("nan")
                        vlo, vhi = lo, hi
                    flag = ""
                    if var in TEMPORAL and vhi < 0:
                        flag = "  SENSITIVE"
                    if var not in TEMPORAL and vlo > 0:
                        flag = "  INFLATION"
                    if var not in TEMPORAL and vhi < 0:
                        flag = "  deflation"
                    line = (f"  {var:38s} {len(d):4d} {a.mean():+9.4f} "
                            f"[{lo:+8.4f},{hi:+8.4f}] {np.abs(a).mean():8.4f}")
                    if ctrl:
                        line += (f" {ca.mean():+9.4f} [{clo:+8.4f},{chi:+8.4f}]"
                                 if len(ca) else f" {'--':>9s} {'--':>20s}")
                    P(line + flag)
                    rows.append(dict(judge=judge, dataset=ds, group=group, variant=var,
                                     kind=kind, n=len(d), signed_delta=a.mean(),
                                     ci_lo=lo, ci_hi=hi, mean_abs_delta=np.abs(a).mean(),
                                     n_corr=len(ca),
                                     signed_delta_corr=ca.mean() if len(ca) else float("nan"),
                                     ci_lo_corr=clo, ci_hi_corr=chi))
                # gameability gap for this group
                # judge-internal contrast -- NOT the doc's Step 12 gameability
                # gap (that is dJ - dV vs the V-JEPA reference, reported below)
                t = [x for v in TEMPORAL for x in paired_deltas(clips, group, v)]
                s = [x for v in SUPERFICIAL for x in paired_deltas(clips, group, v)]
                if t and s:
                    gap = np.mean(s) - np.mean(t)
                    P(f"  temporal-superficial contrast (judge-internal): temporal {np.mean(t):+.4f}  superficial {np.mean(s):+.4f}  contrast {gap:+.4f}")
                # Step 6 control contrasts. These are clauses of H2 in their
                # own right: a judge that moves under every overlay equally is
                # text-sensitive, not cue-following, and only these separate
                # the two. Paired within clip, so the CI is a paired CI.
                for label, va, vb in CONTRASTS:
                    cd = paired_contrast(clips, group, va, vb)
                    if len(cd) < 5:
                        continue
                    ca = np.array(cd)
                    lo, hi = boot_ci(cd)
                    verdict = ""
                    if lo > 0:
                        verdict = ("  DIRECTION-FOLLOWING"
                                   if label.startswith("C2") else
                                   "  CONTENT-SPECIFIC")
                    elif hi < 0:
                        verdict = "  REVERSED"
                    P(f"    {label:34s} n={len(ca):4d} {ca.mean():+9.4f} "
                      f"[{lo:+8.4f},{hi:+8.4f}]{verdict}")
                    contrast_rows.append(dict(
                        judge=judge, dataset=ds, group=group, contrast=label,
                        var_a=va, var_b=vb, n=len(ca), mean=ca.mean(),
                        ci_lo=lo, ci_hi=hi))

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

    # ---- Step 7 rank instability. These judges are used to RANK generators,
    # so the decision-relevant question is not whether a score moves but
    # whether the leaderboard reorders. videophy2_test is the only corpus with
    # source-model provenance, so this section is test-only by construction.
    gen, gcol, gheads = generators()
    P(f"\n{'#'*78}\n# Step 7 rank instability: do generator rankings survive the attack?\n"
      f"# tau/flip are leaderboard-level over source generators ranked by mean\n"
      f"# CLEAN score; rho is clip-level. Low tau or a high flip rate means a\n"
      f"# benchmark built on this judge is reorderable by the attack.\n{'#'*78}")
    if not gen:
        P(f"\n  (no source-generator column in videophy2_test.csv -- looked for "
          f"{GENERATOR_COLUMNS}.\n   headers seen: {gheads[:12]})")
    else:
        P(f"\n  provenance column {gcol!r}: {len(set(gen.values()))} generators "
          f"over {len(gen)} labelled clips")
        for judge in JUDGES:
            group = PHYSICS_GROUP[judge]
            clips = cache["pass1"][(judge, "test")]
            P(f"\n== {judge} / test / {group} ==")
            P(f"  {'variant':38s} {'gens':>4s} {'clips':>5s} {'kendall tau':>11s} "
              f"{'flip rate':>9s} {'clip rho':>8s}")
            for var in VARIANTS[1:] + [CONTROL]:
                got = rank_instability(clips, group, gen, var)
                if got is None:
                    continue
                tau, flip, rho, ng, nc = got
                note = "  RANKING UNSTABLE" if (tau < 0.8 or flip > 0.1) else ""
                if var == CONTROL:
                    note += "  (codec floor)"
                P(f"  {var:38s} {ng:4d} {nc:5d} {tau:+11.3f} {flip:9.3f} "
                  f"{rho:+8.3f}{note}")
                rank_rows.append(dict(judge=judge, dataset="test", group=group,
                                      variant=var, n_generators=ng, n_clips=nc,
                                      kendall_tau=tau, flip_rate=flip,
                                      clip_rho=rho))
        P(f"\n  Read the codec-control row as the floor: whatever tau it loses is\n"
          f"  re-encoding, not manipulation. An attack row only says something\n"
          f"  once it is worse than that.")

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
                                     mean_abs_delta=np.abs(a).mean(),
                                     n_corr=0, signed_delta_corr=float("nan"),
                                     ci_lo_corr=float("nan"), ci_hi_corr=float("nan")))
    else:
        P("\n(no data/dv.json -- Step 12 dJ-dV section skipped)")

    txt = "\n".join(rep)
    open(os.path.join(HERE, "report.txt"), "w").write(txt)
    with open(os.path.join(HERE, "deltas.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name, rs in (("contrasts.csv", contrast_rows),
                     ("rank_instability.csv", rank_rows)):
        if not rs:
            continue
        with open(os.path.join(HERE, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    print(txt)
    print(f"\nwrote analysis/report.txt and analysis/deltas.csv ({len(rows)} delta rows)"
          f"\n      contrasts.csv ({len(contrast_rows)}), "
          f"rank_instability.csv ({len(rank_rows)})")


if __name__ == "__main__":
    main()
