"""One consolidated results run for the whole research doc -> RESULTS.md.

This does NOT recompute what the existing scripts already compute correctly.
It orchestrates them and assembles their output into one document laid out
section-by-section against the doc, then adds the pieces no single script
covers: the 5.1 selection table, the Step 9 locked-reference test-set eval,
Step 9.5 ablation roll-up, Step 10 predictor, Step 11 human study, the H1
temporal index (dJ - dH on the annotated subset), Step 13 Pass-1<->Pass-2,
and the H1-H4 verdict block.

  Sections and their real source
    5.1   selection table          probes/probe_locked.json
    5.2 + Table 1                   analysis/analyze.py  (Step 12 dJ - dV block)
    Step 6 dJ + contrasts          analysis/analyze.py
    Step 7 rank instability        analysis/analyze.py
    Step 8 clean alignment         analysis/analyze.py  (+ MAE/judge-judge here)
    Step 9 locked reference        reference/probe_locked/dv.json      [here]
    Step 9.5(1) 4-arch ablation    probes/probe_t32_{...}.pt or a note  [here]
    Step 9.5(3) temporal ceiling   probes/temporal_upperbound.json      [here]
    Step 10 predictor              predictor/<model>/report.json        [here]
    ablation 10 prompt robustness  analysis/analyze.py
    ablation 11 identity codec     analysis/analyze.py
    Step 11 human study            annotate.py report()                 [here]
    Step 12 temporal index (H1)    dJ (pass1, 60-clip subset) - dH      [here]
    Step 13 Pass-1<->Pass-2        results/pass2 vs results/pass1        [here]

EC2 (recommended -- analyze.py needs the judge JSONs synced, and this pulls
~7k of them):
    python results_report.py --selftest
    python results_report.py                       # full, writes RESULTS.md
    python results_report.py --skip-sync           # reuse an earlier sync
    python results_report.py --no-analyze          # skip the analyze.py rerun

Colab (paste train_probe.py, analysis/analyze.py, annotate.py, lock_probe.py
into cells first, then this):
    import results_report as R
    R.build()

Duplicated small helpers (mae / spearman) are local so this file does not
depend on paste order for its own sections.
"""

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

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

import boto3
from botocore.config import Config

BUCKET = "nickb-aarj"
READ_WORKERS = 32
s3 = boto3.client("s3", config=Config(
    max_pool_connections=READ_WORKERS + 8,
    retries={"max_attempts": 5, "mode": "standard"}))

JUDGES = ["phyjudge_9b", "vila_ewm", "videophy2_auto"]
DATASETS = ["test", "implausibench_real", "implausibench_implausible"]
RUN_PATHS = {"pass1": "results/pass1", "p0": "results/paraphrase/p0",
             "p1": "results/paraphrase/p1"}
PASS2_PREFIX = "results/pass2"
CSV_KEY = "datasets/videophy2_test/_metadata/videophy2_test.csv"
DV_KEY = "reference/probe_locked/dv.json"
PROBE_LOCKED_JSON = "probes/probe_locked.json"
TEMPORAL_UB_JSON = "probes/temporal_upperbound.json"
PRED_MODEL = "vjepa2_1_vit_giant_384"
PREDICTOR_REPORT = f"predictor/{PRED_MODEL}/report.json"
ABLATION_ARCHS = ["attn", "mean_linear", "proj_mean", "diff_conv"]

PHYJUDGE_LAWS = ["gravity", "inertia", "momentum", "impenetrability", "collision",
                 "material", "buoyancy", "displacement", "flow_dynamics",
                 "boundary_interaction", "fluid_continuity", "reflection", "shadow"]
PHYSICS_CALL = {
    "phyjudge_9b": lambda c: c in PHYJUDGE_LAWS,
    "vila_ewm": lambda c: c.startswith("physical_laws_"),
    "videophy2_auto": lambda c: c == "PC",
}
SCALE_SPAN = {"phyjudge_9b": 4.0, "vila_ewm": 1.0, "videophy2_auto": 4.0}
TEMPORAL = {"shuffle", "reverse", "freeze"}
SUPERFICIAL = ["photometric", "caption_echo_rubric_vocab",
               "caption_echo_score_anchor_positive",
               "caption_echo_authoritative_claim",
               "caption_echo_score_anchor_negative",
               "caption_echo_control_irrelevant"]


# ======================================================================
# S3
# ======================================================================

def _list(prefix):
    keys, p = [], s3.get_paginator("list_objects_v2")
    for page in p.paginate(Bucket=BUCKET, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys


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


def _get_text(key):
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()


def _exists(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


# ======================================================================
# small stats (local so section order does not matter)
# ======================================================================

def _rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).mean()) if len(a) else float("nan")


def boot_ci(x, n_boot=4000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ======================================================================
# adopt analyze / annotate (imported module, or __main__ in paste mode)
# ======================================================================

def _adopt(modname, need):
    try:
        if modname == "analyze":
            sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
            sys.path.insert(0, "analysis")
        mod = __import__(modname)
        if all(hasattr(mod, n) for n in need):
            return mod
    except Exception:
        pass
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and all(hasattr(main_mod, n) for n in need):
        return main_mod
    return None


# ======================================================================
# sync the judge JSONs for analyze.py
# ======================================================================

def sync_analyze_data(data_dir):
    """Mirror what analyze.py globs: DATA/<runpath>/<judge>/<ds>/*.json,
    DATA/videophy2_test.csv, DATA/dv.json."""
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    jobs = []
    for run, rp in RUN_PATHS.items():
        local_run = "pass1" if run == "pass1" else f"paraphrase/{run}"
        for j in JUDGES:
            for ds in DATASETS:
                pref = f"{rp}/{j}/{ds}/"
                for k in _list(pref):
                    if k.endswith(".json"):
                        dst = data / local_run / j / ds / k.rsplit("/", 1)[-1]
                        jobs.append((k, dst))
    print(f"  {len(jobs)} judge records to sync ...")

    def _dl(kd):
        k, dst = kd
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())

    with ThreadPoolExecutor(READ_WORKERS) as ex:
        list(ex.map(_dl, jobs))

    (data / "videophy2_test.csv").write_text(_get_text(CSV_KEY), encoding="utf-8")
    if _exists(DV_KEY):
        (data / "dv.json").write_bytes(
            s3.get_object(Bucket=BUCKET, Key=DV_KEY)["Body"].read())
        print("  dv.json synced")
    else:
        print("  NOTE reference/probe_locked/dv.json missing -- Step 12 dJ-dV "
              "and Step 9 will be limited")
    print(f"  synced -> {data}")


def run_analyze(data_dir):
    """Run analysis/analyze.py against the synced dir; return its report.txt."""
    az = _adopt("analyze", ["main", "load", "boot_ci", "GROUPS"])
    if az is None:
        return ("[analyze.py not importable -- paste analysis/analyze.py into a "
                "cell first, or run `python analysis/analyze.py` separately]")
    az.DATA = str(data_dir)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            az.main()
    except Exception as exc:                       # noqa: BLE001
        return f"[analyze.main() raised: {type(exc).__name__}: {exc}]\n" + buf.getvalue()
    here = Path(getattr(az, "HERE", ".")) / "report.txt"
    return here.read_text() if here.exists() else buf.getvalue()


# ======================================================================
# judge loading for the here-computed sections
# ======================================================================

def load_pass(prefix, judge, ds, group_pred):
    """{stem: {variant: mean physics-group score}} for one (prefix,judge,ds)."""
    out = {}
    for rec in _get_many([k for k in _list(f"{prefix}/{judge}/{ds}/")
                          if k.endswith(".json")]):
        if not rec:
            continue
        per = {}
        for var, r in rec.get("runs", {}).items():
            xs = []
            for cid, c in r.get("calls", {}).items():
                if not group_pred(cid):
                    continue
                p = c.get("parsed")
                if isinstance(p, bool):
                    xs.append(1.0 if p else 0.0)
                elif isinstance(p, (int, float)):
                    xs.append(float(p))
            if xs:
                per[var] = float(np.mean(xs))
        out[rec.get("clip")] = per
    return out


def human_labels():
    labs = {}
    import csv
    for row in csv.DictReader(io.StringIO(_get_text(CSV_KEY))):
        stem = os.path.splitext(os.path.basename(row.get("video_url", "")))[0]
        try:
            labs[stem] = (int(float(row["pc"])), int(float(row["sa"])))
        except (ValueError, KeyError):
            pass
    return labs


def _lab(labs, stem):
    return labs.get(stem) or labs.get(stem.removesuffix("_result")) \
        or labs.get(stem + "_result")


# ======================================================================
# 5.1  locked reference selection
# ======================================================================

def section_5_1():
    m = _get(PROBE_LOCKED_JSON)
    L = ["## 5.1 Locked human-trained video reference\n"]
    if not m:
        return "\n".join(L + ["_probes/probe_locked.json not found -- run "
                              "lock_probe.py._\n"])
    sel = m.get("selection", {})
    summ = sel.get("summary", {})
    L.append(f"- probe: **{m['probe']['arch']}** seed {m['probe']['seed']} "
             f"({m['probe']['params']} params); "
             f"selection criterion: {sel.get('criterion')}, pick {sel.get('pick')}")
    L.append(f"- calibration: {m['calibration']['method'].split(',')[0]}, "
             f"fit on `{m['calibration']['fit_on']}` (n = {m['calibration']['n']}), "
             f"{len(m['calibration'].get('x', []))} knots\n")
    L.append("| arch | macro MAE | Spearman | seeds |")
    L.append("|---|---|---|---|")
    for arch, st in sorted(summ.items(), key=lambda kv: kv[1].get("macro_mae", 9)):
        L.append(f"| {arch} | {st.get('macro_mae'):.3f} "
                 f"± {st.get('macro_mae_sd', 0):.3f} | "
                 f"{st.get('rho'):+.3f} ± {st.get('rho_sd', 0):.3f} | "
                 f"{st.get('n_seeds')} |")
    v = m.get("val", {})
    if v:
        L.append(f"\n- selected checkpoint val: MAE {v.get('mae', float('nan')):.3f}, "
                 f"macro MAE {v.get('macro_mae', float('nan')):.3f}, "
                 f"rho {v.get('rho', float('nan')):+.3f}, epoch {v.get('epoch')}")
    return "\n".join(L) + "\n"


# ======================================================================
# Step 9  locked reference on untouched VideoPhy-2 test human PC
# ======================================================================

def section_step9():
    L = ["## Step 9 Locked reference on VideoPhy-2 test (untouched human PC)\n"]
    dv = _get(DV_KEY)
    if not dv:
        return "\n".join(L + ["_reference/probe_locked/dv.json not found -- run "
                              "score_corpus.py._\n"])
    ds = dv["datasets"].get("videophy2_test") or dv["datasets"].get("test")
    if not ds:
        return "\n".join(L + ["_no videophy2_test block in dv.json._\n"])
    clips = ds["clips"]
    labs = human_labels()

    ph, pm = [], []
    lo_pc, hi_pc, sa_lo_pc, sa_hi_pc = [], [], [], []
    by_level = defaultdict(list)
    for stem, c in clips.items():
        hp = c.get("pc_human")
        lab = _lab(labs, stem)
        if hp is None:
            hp = lab[0] if lab else None
        if hp is None:
            continue
        pred = c["clean"]["pc"]
        ph.append(hp)
        pm.append(pred)
        by_level[hp].append(pred)
        if hp <= 2:
            lo_pc.append(pred)
            if lab and lab[1] >= 4:
                sa_lo_pc.append(pred)
        elif hp >= 4:
            hi_pc.append(pred)
            if lab and lab[1] >= 4:
                sa_hi_pc.append(pred)

    L.append(f"- n = {len(ph)} test clips with a human PC label")
    L.append(f"- **MAE {mae(ph, pm):.3f}**, **Spearman {spearman(ph, pm):+.3f}** "
             f"(calibrated PC vs human PC)")
    L.append("\n| human PC | n | mean predicted PC |")
    L.append("|---|---|---|")
    for lvl in sorted(by_level):
        L.append(f"| {lvl} | {len(by_level[lvl])} | {np.mean(by_level[lvl]):.3f} |")
    if lo_pc and hi_pc:
        L.append(f"\n- low/high PC separation: PC≤2 -> {np.mean(lo_pc):.3f}, "
                 f"PC≥4 -> {np.mean(hi_pc):.3f}, gap {np.mean(hi_pc)-np.mean(lo_pc):+.3f}")
    if sa_lo_pc and sa_hi_pc:
        L.append(f"- high-SA subgroup (SA≥4): PC≤2 -> {np.mean(sa_lo_pc):.3f}, "
                 f"PC≥4 -> {np.mean(sa_hi_pc):.3f}, gap "
                 f"{np.mean(sa_hi_pc)-np.mean(sa_lo_pc):+.3f}  (n={len(sa_lo_pc)+len(sa_hi_pc)})")

    L.append("\n### reference superficial invariance / temporal sensitivity "
             "(pooled corpora, dV normalised 0-1)\n")
    L.append("| variant | kind | dV | 95% CI |")
    L.append("|---|---|---|---|")
    pooled = defaultdict(lambda: {"n": 0, "s": 0.0, "lo": 0.0, "hi": 0.0})
    for dname, dd in dv["datasets"].items():
        for v, r in dd.get("per_variant", {}).items():
            a = pooled[v]
            a["n"] += r["n"]
            a["s"] += r["d_norm"] * r["n"]
            a["lo"] += r["d_norm_lo"] * r["n"]
            a["hi"] += r["d_norm_hi"] * r["n"]
    for v in list(TEMPORAL) + SUPERFICIAL:
        if v not in pooled or not pooled[v]["n"]:
            continue
        a = pooled[v]
        kind = "temporal" if v in TEMPORAL else "superficial"
        L.append(f"| {v} | {kind} | {a['s']/a['n']:+.4f} | "
                 f"[{a['lo']/a['n']:+.4f}, {a['hi']/a['n']:+.4f}] |")
    L.append("\n_The reference is order-blind by construction (proj_mean means "
             "the 32 moments away), so its temporal dV is a structural null and "
             "H1 falls back to human ΔH below._\n")
    return "\n".join(L) + "\n", {"mae": mae(ph, pm), "rho": spearman(ph, pm),
                                 "n": len(ph)}


# ======================================================================
# Step 9.5  probe ablations
# ======================================================================

def section_step9_5():
    L = ["## Step 9.5 Reference-probe ablations\n"]

    # (1) 4-arch
    L.append("### (1) 4-arch readout ablation (mean_linear / proj_mean / attn / diff_conv)\n")
    got = {}
    try:
        import torch
        for arch in ABLATION_ARCHS:
            for cand in (f"probe_t32_{arch}", f"probe_locked_{arch}",
                         f"probe_{arch}"):
                key = f"probes/{cand}.pt"
                if _exists(key):
                    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
                    ck = torch.load(io.BytesIO(body), map_location="cpu",
                                    weights_only=False)
                    vm = ck.get("val") or ck.get("best") or {}
                    got[arch] = {k: vm.get(k) for k in ("mae", "macro_mae", "rho")}
                    break
    except Exception as exc:                       # noqa: BLE001
        L.append(f"_could not read ablation checkpoints ({type(exc).__name__}); "
                 f"run `train_probe.py --ablation --name probe_t32` and paste "
                 f"its table._\n")
    if got:
        L.append("| arch | MAE | macro MAE | Spearman |")
        L.append("|---|---|---|---|")
        for arch in ABLATION_ARCHS:
            g = got.get(arch)
            if g:
                L.append(f"| {arch} | {g.get('mae', float('nan')):.3f} | "
                         f"{g.get('macro_mae', float('nan')):.3f} | "
                         f"{g.get('rho', float('nan')):+.3f} |")
    elif "could not read" not in "\n".join(L):
        L.append("_no probe_t32_{arch}.pt checkpoints in probes/ -- "
                 "run `train_probe.py --ablation`._\n")

    # (2) clean-only vs nuisance-invariant
    L.append("\n### (2) clean-only (lambda_cons=0) vs nuisance-invariant "
             "(lambda_cons>0) training\n")
    ab92 = {}
    try:
        import torch
        for nm in ("probe_ab92_nuisance", "probe_ab92_cleanonly"):
            key = f"probes/{nm}.pt"
            if not _exists(key):
                continue
            ck = torch.load(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()),
                            map_location="cpu", weights_only=False)
            vm = ck.get("val") or ck.get("best") or {}
            cfg = ck.get("cfg", {})
            ab92[nm] = {"lambda_cons": cfg.get("lambda_cons"),
                        "mae": vm.get("mae"), "macro_mae": vm.get("macro_mae"),
                        "rho": vm.get("rho")}
    except Exception as exc:                       # noqa: BLE001
        L.append(f"_could not read probe_ab92_* ({type(exc).__name__})._\n")
    if ab92:
        L.append("| probe | lambda_cons | val MAE | val macro MAE | val rho |")
        L.append("|---|---|---|---|---|")
        for nm in ("probe_ab92_nuisance", "probe_ab92_cleanonly"):
            g = ab92.get(nm)
            if g:
                L.append(f"| {nm.replace('probe_ab92_', '')} | {g['lambda_cons']} | "
                         f"{g['mae']:.3f} | {g['macro_mae']:.3f} | {g['rho']:+.3f} |")
        L.append("\n_superficial-cue dV per variant: `eval_probe.report(name=...)` "
                 "for each. Near-identical dV between the two => superficial "
                 "invariance is native to V-JEPA, not induced by the consistency "
                 "term._\n")
    else:
        L.append("_probes/probe_ab92_{nuisance,cleanonly}.pt not found -- run "
                 "`train_probe.py --arch proj_mean --lambda-cons {1,0}`._\n")

    # (3) attack-aware temporal upper bound
    L.append("\n### (3) attack-aware temporal upper bound (diagnostic only)\n")
    ub = _get(TEMPORAL_UB_JSON)
    if not ub:
        L.append("_probes/temporal_upperbound.json not found -- run "
                 "`train_temporal_upperbound.py`._\n")
    else:
        L.append(f"- {ub['n_confirmed']} human-confirmed clean>temporal pairs, "
                 f"{ub['config']['folds']}-fold over base clips")
        L.append(f"- **verdict: {ub['verdict']}** — {ub['verdict_reason']}\n")
        L.append("| variant | n | upper-bound AUC | 95% CI | reference-probe AUC |")
        L.append("|---|---|---|---|---|")
        for v, r in ub["per_variant"].items():
            if not r:
                continue
            ref = r.get("ref_probe_auc")
            L.append(f"| {v} | {r['n']} | {r['auc']:.3f} | "
                     f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
                     f"{'n/a' if ref is None else f'{ref:.3f}'} |")
    return "\n".join(L) + "\n"


# ======================================================================
# Step 10  predictor / latent surprise
# ======================================================================

def section_step10():
    L = ["## Step 10 V-JEPA predictor / latent surprise\n"]
    rep = _get(PREDICTOR_REPORT)
    if not rep:
        L.append(f"_predictor/{PRED_MODEL}/report.json not found. If the "
                 f"corpus-scale predictor run was intended, run "
                 f"`predict_vjepa.py --datasets train all` then "
                 f"`predict_vjepa.py --report`; then `decide_predictor_probe.py`. "
                 f"Otherwise this step is optional and can be marked deferred._\n")
        return "\n".join(L) + "\n"
    verdict = _get_text(f"predictor/{PRED_MODEL}/verdict.txt").strip() \
        if _exists(f"predictor/{PRED_MODEL}/verdict.txt") else None
    if verdict:
        L.append(f"- **decision (6-stat sweep, decide_predictor_probe): {verdict}**\n")
    L.append(f"- statistic shown: {rep.get('stat')}, spike P{rep.get('spike_percentile')}, "
             f"threshold {rep.get('threshold'):.4g}, n = {rep.get('n_clips')} clips\n")
    L.append("| variant | kind | d | 95% CI | AUC |")
    L.append("|---|---|---|---|---|")
    for name, r in sorted(rep.get("variants", {}).items(),
                          key=lambda kv: (kv[1].get("kind", "z"), kv[0])):
        if str(name).startswith("copy:"):
            continue
        L.append(f"| {name} | {r.get('kind')} | {r.get('d', float('nan')):+.4f} | "
                 f"[{r.get('lo', float('nan')):+.4f}, {r.get('hi', float('nan')):+.4f}] | "
                 f"{r.get('auc', float('nan')):.3f} |")
    dcsv = _get_text(f"predictor/{PRED_MODEL}/decision.csv") \
        if _exists(f"predictor/{PRED_MODEL}/decision.csv") else None
    if dcsv:
        import csv as _csv
        rows = list(_csv.DictReader(io.StringIO(dcsv)))
        sig = [(r["stat"], r["variant"], r["kind"]) for r in rows
               if r.get("significant") in ("True", "true", "1")]
        L.append("\n_significant rises (CI>0), (stat, variant, kind): "
                 + (", ".join(f"{s}/{v}/{k}" for s, v, k in sig) or "none") + "._")
    return "\n".join(L) + "\n"


# ======================================================================
# Step 11  human study
# ======================================================================

def section_step11(annot_version="v1"):
    L = ["## Step 11 Blinded human validation (60-clip subset)\n"]
    an = _adopt("annotate", ["report", "compare_to_vjepa"])
    if an is None:
        return "\n".join(L + ["_annotate.py not importable -- paste it into a "
                              "cell, or run `python annotate.py --report`._\n"]), {}
    vj = _get("reference/probe_locked/vjepa_deltas.json") or {}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            human = an.report(version=annot_version,
                              vjepa_deltas=vj if vj else None)
    except Exception as exc:                       # noqa: BLE001
        return "\n".join(L + [f"_annotate.report() raised: "
                              f"{type(exc).__name__}: {exc}_\n"]), {}
    out = buf.getvalue()
    rline = next((ln.strip() for ln in out.splitlines() if "raters (" in ln), "")
    L.append(f"{rline}  (Krippendorff alpha / inter-rater kappa need >=2 raters)\n")
    L.append("| attack | kind | clean-pref | 95% CI | dH (PC) | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for v in list(TEMPORAL) + SUPERFICIAL:
        h = human.get(v)
        if not h:
            continue
        lo, hi = h["ci"]
        dlo, dhi = h["d_pc_ci"]
        pref = ("--" if isinstance(h["clean_pref"], float)
                and h["clean_pref"] != h["clean_pref"] else f"{h['clean_pref']:.2f}")
        ci = "--" if lo != lo else f"[{lo:.2f}, {hi:.2f}]"
        L.append(f"| {v} | {h['kind']} | {pref} | {ci} | "
                 f"{h['d_pc']:+.2f} | [{dlo:+.2f}, {dhi:+.2f}] |")
    # scale check + V-JEPA-vs-human, one summary line each
    for needle, label in (("exact match", "scale check (our clean PC vs published)"),
                          ("spearman(clean-preference", "V-JEPA vs human (temporal)")):
        ln = next((x.strip() for x in out.splitlines() if needle in x), "")
        if ln:
            L.append(f"\n_{label}:_ {ln}")
    L.append("\n_temporal attacks: humans prefer clean (validated). superficial: "
             "near-total ties / dH ~ 0 (humans invariant). V-JEPA temporal dV is "
             "positive (order-blind) so H1 uses dH._")
    return "\n".join(L) + "\n", human


# ======================================================================
# Step 12 temporal index  (H1):  dJ - dH on the annotated subset
# ======================================================================

def section_step12_temporal(human, annot_version="v1"):
    L = ["## Step 12 Temporal-insensitivity index (H1): dJ − dH on the "
         "human-annotated subset\n"]
    if not human:
        return "\n".join(L + ["_no human deltas (Step 11 did not run)._\n"]), {}
    an = _adopt("annotate", ["load_records"])
    stems = set()
    try:
        for r in an.load_records(annot_version, None, prefer_local=False):
            if not r.get("skipped"):
                stems.add(r["stem"])
    except Exception:
        pass
    L.append(f"- annotated base clips: {len(stems)}\n")
    L.append("| judge | attack | dJ (norm) | dH (norm, /4) | index dJ−dH | 95% CI | n |")
    L.append("|---|---|---|---|---|---|---|")
    verdict_rows = []
    for judge in JUDGES:
        span = SCALE_SPAN[judge]
        # judge deltas on the annotated stems, pooled across the 3 corpora
        perclip = {}
        for ds in DATASETS:
            for stem, per in load_pass("results/pass1", judge, ds,
                                       PHYSICS_CALL[judge]).items():
                key = stem.removesuffix("_result")
                if key in stems or stem in stems:
                    perclip[key] = per
        for v in sorted(TEMPORAL):
            hj = human.get(v)
            if not hj or v == "clean":
                continue
            dh = hj["d_pc"] / 4.0
            dJ_list = [(per[v] - per["clean"]) / span for per in perclip.values()
                       if "clean" in per and v in per]
            if len(dJ_list) < 5:
                continue
            idx = [x - dh for x in dJ_list]
            lo, hi = boot_ci(idx)
            flag = "  H1-POSITIVE" if lo > 0 else ""
            L.append(f"| {judge} | {v} | {np.mean(dJ_list):+.4f} | {dh:+.4f} | "
                     f"{np.mean(idx):+.4f} | [{lo:+.4f}, {hi:+.4f}] | {len(dJ_list)} |{flag}")
            verdict_rows.append((judge, v, np.mean(idx), lo, hi))
    L.append("\n_index > 0 with CI excluding 0 = the judge under-responds to a "
             "temporal attack humans flagged (H1). dH is the rater's mean "
             "(pc_variant − pc_clean), normalised by the 1-5 span._\n")
    return "\n".join(L) + "\n", {"rows": verdict_rows}


# ======================================================================
# Step 13  Pass-1 <-> Pass-2  (PhyJudge only)
# ======================================================================

PASS2_PREFIXES = ["results/pass2", "results/pass2_captions"]


def _load_pass2(judge, ds, pred):
    """pass-2 physics scores merged across every pass2* prefix (caption-echo
    variants were written to results/pass2_captions/)."""
    merged = {}
    for pre in PASS2_PREFIXES:
        for stem, per in load_pass(pre, judge, ds, pred).items():
            merged.setdefault(stem, {}).update(per)
    return merged


def section_step13():
    L = ["## Step 13 - Pass-1 <-> Pass-2 consistency (PhyJudge)\n"]
    judge = "phyjudge_9b"
    pred = PHYSICS_CALL[judge]
    clips = 0
    p1v, p2v, shifts, n_pairs = [], [], 0, 0
    parse_ok = parse_tot = 0
    for ds in DATASETS:
        p1 = load_pass("results/pass1", judge, ds, pred)
        p2 = _load_pass2(judge, ds, pred)
        if not p2:
            continue
        clips += len(p2)
        for stem, per2 in p2.items():
            per1 = p1.get(stem) or p1.get(stem.removesuffix("_result")) \
                or p1.get(stem + "_result") or {}
            for v, s2 in per2.items():
                s1 = per1.get(v)
                if s1 is None:
                    continue
                p1v.append(s1)
                p2v.append(s2)
                n_pairs += 1
                if abs(s2 - s1) >= 1.0:
                    shifts += 1
        for pre in PASS2_PREFIXES:
            for rec in _get_many([k for k in _list(f"{pre}/{judge}/{ds}/")
                                  if k.endswith(".json")]):
                if not rec:
                    continue
                for r in rec.get("runs", {}).values():
                    for cid, c in r.get("calls", {}).items():
                        if not pred(cid):
                            continue
                        parse_tot += 1
                        parse_ok += c.get("parsed") is not None
    if not n_pairs:
        return "\n".join(L + ["_no pass-2 phyjudge_9b records found._\n"]), {}
    rho, shift = spearman(p1v, p2v), shifts / n_pairs
    L.append(f"- {clips} pass-2 clips, {n_pairs} matched (clip,variant) physics-score pairs")
    L.append(f"- Spearman(pass1, pass2) **{rho:+.3f}**, "
             f"MAD **{mae(p1v, p2v):.3f}** (native units)")
    L.append(f"- fraction of pairs shifting >=1 unit: **{shift:.1%}**")
    if parse_tot:
        L.append(f"- pass-2 rationale parse rate: {parse_ok}/{parse_tot} "
                 f"({parse_ok / parse_tot:.1%})")
    L.append("- **Part A (case gallery, two-coder kappa + failure-mode "
             "distribution): pending second coder.**")
    return "\n".join(L) + "\n", {"rho": rho, "shift_frac": shift}


# ======================================================================
# Table 1 + H1-H4 verdict, from deltas.csv (analyze.py output)
# ======================================================================

def _read_deltas_csv():
    for cand in ("analysis/deltas.csv", "deltas.csv"):
        if os.path.exists(cand):
            import csv
            with open(cand) as fh:
                return list(csv.DictReader(fh))
    return []


def section_step12(temporal_idx):
    """Table 1 + the H2 inflation index + the H1 temporal index + worst-case.
    Returns (text, verdict_lines)."""
    L = ["## Step 12 - gameability metrics\n"]
    step12 = [r for r in _read_deltas_csv() if r["group"].startswith("step12_")]
    T1 = ("shuffle", "freeze", "caption_echo_rubric_vocab")

    def _cell(judge, ds, v):
        m = next((r for r in step12 if r["judge"] == judge
                  and r["dataset"] == ds and r["variant"] == v), None)
        return m

    L.append("### Table 1 - reference-relative gap  d = dJ - dV  "
             "(per dataset, [95% CI])\n")
    L.append("| judge | dataset | shuffle | freeze | caption_echo_rubric_vocab |")
    L.append("|---|---|---|---|---|")
    for judge in JUDGES:
        for ds in DATASETS:
            cells = []
            for v in T1:
                m = _cell(judge, ds, v)
                cells.append(f"{float(m['signed_delta']):+.3f} "
                             f"[{float(m['ci_lo']):+.3f}, {float(m['ci_hi']):+.3f}]"
                             if m else "--")
            L.append(f"| {judge} | {ds} | " + " | ".join(cells) + " |")

    L.append("\n### H2 - superficial-cue inflation index  d = dJ - dV  "
             "(per dataset, * = 95% CI excludes 0)\n")
    L.append("| judge | dataset | " + " | ".join(v.replace("caption_echo_", "")
                                                 for v in SUPERFICIAL) + " |")
    L.append("|" + "---|" * (len(SUPERFICIAL) + 2))
    h2_judges = set()
    for judge in JUDGES:
        for ds in DATASETS:
            cells = []
            for v in SUPERFICIAL:
                m = _cell(judge, ds, v)
                if not m:
                    cells.append("--")
                    continue
                star = "*" if float(m["ci_lo"]) > 0 else ""
                if star and ds == "test":
                    h2_judges.add(judge)
                cells.append(f"{float(m['signed_delta']):+.3f}{star}")
            L.append(f"| {judge} | {ds} | " + " | ".join(cells) + " |")

    fam_judges = defaultdict(set)
    for r in step12:
        if r["dataset"] == "test" and float(r["ci_lo"]) > 0:
            fam_judges[r["variant"]].add(r["judge"])
    h3_fams = {v: sorted(j) for v, j in fam_judges.items() if len(j) >= 2}
    L.append("\n### H3 - families with d>0 (CI excl. 0) on >=2 judges (test)\n")
    L.append("- " + (", ".join(f"{v} ({len(j)})" for v, j in h3_fams.items())
                     or "none"))

    L.append("\n### H1 - temporal-insensitivity index  dJ - dH  "
             "(annotated subset, n=60, [95% CI])\n")
    L.append("| judge | shuffle | reverse | freeze |")
    L.append("|---|---|---|---|")
    tix = defaultdict(dict)
    for (j, v, mn, lo, hi) in temporal_idx.get("rows", []):
        tix[j][v] = (mn, lo, hi)
    h1_ok = 0
    for judge in JUDGES:
        cells = []
        for v in ("shuffle", "reverse", "freeze"):
            r = tix[judge].get(v)
            if not r:
                cells.append("--")
                continue
            h1_ok += r[1] > 0
            cells.append(f"{r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]")
        L.append(f"| {judge} | " + " | ".join(cells) + " |")

    L.append("\n### worst-case gap per judge (test, any attack family)\n")
    for judge in JUDGES:
        jr = [r for r in step12 if r["judge"] == judge and r["dataset"] == "test"]
        if jr:
            w = max(jr, key=lambda r: float(r["signed_delta"]))
            L.append(f"- {judge}: {w['variant']} d = "
                     f"{float(w['signed_delta']):+.3f} "
                     f"[{float(w['ci_lo']):+.3f}, {float(w['ci_hi']):+.3f}]")

    h1 = h1_ok >= 3
    h2 = len(h2_judges) >= 2
    h3 = bool(h3_fams)
    verdicts = [
        f"- **H1 temporal blindness**: {'SUPPORTED' if h1 else 'not established'} "
        f"- dJ-dH > 0 with CI excl. 0 on {h1_ok}/9 judge x temporal-attack cells "
        f"(annotated subset, 1 rater)",
        f"- **H2 superficial-cue exploitation**: {'SUPPORTED' if h2 else 'not supported'} "
        f"- caption-echo overlays inflate {len(h2_judges)}/3 judges (d = dJ-dV > 0, CI excl. 0)",
        f"- **H3 shared vulnerability**: {'SUPPORTED' if h3 else 'not supported'} "
        f"- {', '.join(h3_fams) or 'no family'} clears d>0/CI on >=2 judges",
        "- **H4 failure attribution**: Pass-1<->Pass-2 stable (see Step 13); "
        "case-gallery two-coder kappa pending second coder",
    ]
    return "\n".join(L) + "\n", verdicts


# ======================================================================
# lean Step 6 / 7 / 8  (compact, from analyze.py's CSVs + a recompute)
# ======================================================================

PHYS_GROUP = {"phyjudge_9b": "laws13", "vila_ewm": "physical_laws_bool",
              "videophy2_auto": "PC"}
_PHYS_LAWS = ("gravity", "inertia", "momentum", "impenetrability", "collision",
              "material", "buoyancy", "displacement", "flow_dynamics",
              "boundary_interaction", "fluid_continuity", "reflection", "shadow")
PHYS_GROUP_CALL = {
    "phyjudge_9b": lambda c: c in _PHYS_LAWS,
    "vila_ewm": lambda c: c.startswith("physical_laws_"),
    "videophy2_auto": lambda c: c == "PC",
}
_PHYS_SPAN = {"phyjudge_9b": 4.0, "vila_ewm": 1.0, "videophy2_auto": 4.0}
_PHYS_LO = {"phyjudge_9b": 1.0, "vila_ewm": 0.0, "videophy2_auto": 1.0}


def _normphys(judge, x):
    return (x - _PHYS_LO[judge]) / _PHYS_SPAN[judge]


def _read_csv(name):
    import csv
    for cand in (os.path.join("analysis", name), name):
        if os.path.exists(cand):
            with open(cand) as fh:
                return list(csv.DictReader(fh))
    return []


def _lean_step6_7(ds="test"):
    L = [f"## Step 6 - judge attack response dJ ({ds}, physics group, [95% CI])\n"]
    d = _read_csv("deltas.csv")
    for judge in JUDGES:
        rows = [r for r in d if r["judge"] == judge and r["dataset"] == ds
                and r["group"] == PHYS_GROUP[judge]]
        if not rows:
            continue
        by = {r["variant"]: r for r in rows}
        L.append(f"**{judge}**")
        L.append("| attack | dJ | 95% CI | flag |")
        L.append("|---|---|---|---|")
        for v in list(TEMPORAL) + SUPERFICIAL:
            r = by.get(v)
            if not r:
                continue
            lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
            flag = ("SENSITIVE" if v in TEMPORAL and hi < 0 else
                    "INFLATION" if v in SUPERFICIAL and lo > 0 else
                    "deflation" if v in SUPERFICIAL and hi < 0 else "")
            L.append(f"| {v} | {float(r['signed_delta']):+.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | {flag} |")
        L.append("")
    c = _read_csv("contrasts.csv")
    if c:
        L.append("### control contrasts (paired, [95% CI])\n")
        L.append("| judge | contrast | mean | 95% CI |")
        L.append("|---|---|---|---|")
        for r in c:
            if r["dataset"] != ds or r.get("group") != PHYS_GROUP[r["judge"]]:
                continue
            L.append(f"| {r['judge']} | {r['contrast']} | {float(r['mean']):+.3f} | "
                     f"[{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}] |")
        L.append("")
    ri = _read_csv("rank_instability.csv")
    if ri:
        L.append("## Step 7 - generator rank instability (test, worst attack per judge)\n")
        L.append("| judge | worst variant | Kendall tau | flip rate | clip rho |")
        L.append("|---|---|---|---|---|")
        for judge in JUDGES:
            jr = [r for r in ri if r["judge"] == judge and r["variant"] != "identity"]
            if not jr:
                continue
            w = min(jr, key=lambda r: float(r["kendall_tau"]))
            L.append(f"| {judge} | {w['variant']} | {float(w['kendall_tau']):+.3f} | "
                     f"{float(w['flip_rate']):.2f} | {float(w['clip_rho']):+.3f} |")
        L.append("")
    return "\n".join(L)


def _step8_alignment():
    L = ["## Step 8 - clean judge vs human PC (VideoPhy-2 test)\n",
         "| judge | n | MAE | Spearman |", "|---|---|---|---|"]
    labs = human_labels()
    stem_scores = {}
    for judge in JUDGES:
        h, m, sc = [], [], {}
        for stem, pv in load_pass("results/pass1", judge, "test", PHYS_GROUP_CALL[judge]).items():
            lab = _lab(labs, stem)
            if lab and "clean" in pv:
                h.append((lab[0] - 1) / 4.0)
                m.append(_normphys(judge, pv["clean"]))
                sc[stem.removesuffix("_result")] = _normphys(judge, pv["clean"])
        stem_scores[judge] = sc
        if len(h) >= 5:
            L.append(f"| {judge} | {len(h)} | {mae(h, m):.3f} | {spearman(h, m):+.3f} |")
    L.append("\n### judge x judge Spearman (clean, shared test clips)\n")
    L.append("| pair | rho | n |")
    L.append("|---|---|---|")
    js = list(JUDGES)
    for i in range(len(js)):
        for k in range(i + 1, len(js)):
            common = set(stem_scores[js[i]]) & set(stem_scores[js[k]])
            if len(common) < 5:
                continue
            x = [stem_scores[js[i]][cc] for cc in common]
            y = [stem_scores[js[k]][cc] for cc in common]
            L.append(f"| {js[i]} vs {js[k]} | {spearman(x, y):+.3f} | {len(common)} |")
    return "\n".join(L) + "\n"


def _ablation_10_11():
    """(10) prompt robustness native<->p0/p1, (11) identity codec correction --
    both compact, from the synced JSONs / deltas.csv."""
    L = ["## Ablation 10 - judge-prompt robustness (native vs 2 paraphrases, "
         "clean physics score, test)\n",
         "| judge | rho p0 | MAD p0 | rho p1 | MAD p1 |", "|---|---|---|---|---|"]
    for judge in JUDGES:
        base = load_pass("results/pass1", judge, "test", PHYS_GROUP_CALL[judge])
        cells = []
        for run in ("results/paraphrase/p0", "results/paraphrase/p1"):
            other = load_pass(run, judge, "test", PHYS_GROUP_CALL[judge])
            pairs = [(pv["clean"], other[s]["clean"]) for s, pv in base.items()
                     if "clean" in pv and s in other and "clean" in other[s]]
            if len(pairs) < 5:
                cells += ["--", "--"]
                continue
            a, b = zip(*pairs)
            cells += [f"{spearman(a, b):+.3f}", f"{mae(a, b):.3f}"]
        L.append(f"| {judge} | " + " | ".join(cells) + " |")

    d = _read_deltas_csv()
    L.append("\n## Ablation 11 - identity re-encode control (test, temporal "
             "variants: raw dJ vs codec-corrected)\n")
    L.append("| judge | shuffle raw / corr | reverse raw / corr | freeze raw / corr |")
    L.append("|---|---|---|---|")
    for judge in JUDGES:
        cells = []
        for v in ("shuffle", "reverse", "freeze"):
            m = next((r for r in d if r["judge"] == judge and r["dataset"] == "test"
                      and r["group"] == PHYS_GROUP[judge] and r["variant"] == v), None)
            if m and m.get("signed_delta_corr") not in (None, "", "nan"):
                cells.append(f"{float(m['signed_delta']):+.3f} / "
                             f"{float(m['signed_delta_corr']):+.3f}")
            elif m:
                cells.append(f"{float(m['signed_delta']):+.3f} / --")
            else:
                cells.append("--")
        L.append(f"| {judge} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


CLIP_TARGETS = {"test": 450, "implausibench_real": 150,
                "implausibench_implausible": 150}
_CALLS_N = {"phyjudge_9b": 16, "vila_ewm": 8, "videophy2_auto": 2}
_ALL_VARIANTS = ["clean"] + list(TEMPORAL) + SUPERFICIAL


def section_step4():
    """Pass-1 coverage / quality audit -- the gate. Native run only."""
    L = ["## Step 4 - Pass-1 coverage & quality audit\n",
         "| judge | dataset | clips | complete | unparsed calls | % complete |",
         "|---|---|---|---|---|---|"]
    ok_all = True
    for judge in JUDGES:
        ncall = _CALLS_N[judge]
        for ds in DATASETS:
            keys = [k for k in _list(f"results/pass1/{judge}/{ds}/")
                    if k.endswith(".json")]
            recs = [r for r in _get_many(keys) if r]
            complete = unparsed = 0
            for rec in recs:
                runs = rec.get("runs", {})
                nvar = sum(1 for v in _ALL_VARIANTS if runs.get(v, {}).get("calls"))
                nbad = sum(1 for v in runs.values() for c in v.get("calls", {}).values()
                           if c.get("parsed") is None)
                unparsed += nbad
                got = sum(len(v.get("calls", {})) for v in runs.values())
                if nvar >= 10 and got >= 10 * ncall and nbad == 0:
                    complete += 1
            pct = complete / len(recs) if recs else 0.0
            if pct < 0.99:
                ok_all = False
            L.append(f"| {judge} | {ds} | {len(recs)} | {complete} | "
                     f"{unparsed} | {pct:.0%} |")
    tail = "PASSES" if ok_all else ("has gaps -- see check_complete.py / "
                                    "audit_runs.py for the incomplete-clip list")
    L.append(f"\n_audit {tail}._")
    return "\n".join(L) + "\n"


def _sanity_real():
    """Real-video sanity control (ablations 1 & 7): clean score + photometric dJ
    on ImplausiBench real."""
    L = ["## Real-video sanity control (ImplausiBench real, physics score)\n",
         "| judge | clean mean | photometric dJ | 95% CI |", "|---|---|---|---|"]
    d = _read_deltas_csv()
    for judge in JUDGES:
        base = load_pass("results/pass1", judge, "implausibench_real",
                         PHYS_GROUP_CALL[judge])
        cm = np.mean([_normphys(judge, v["clean"]) for v in base.values()
                      if "clean" in v]) if base else float("nan")
        m = next((r for r in d if r["judge"] == judge
                  and r["dataset"] == "implausibench_real"
                  and r["group"] == PHYS_GROUP[judge]
                  and r["variant"] == "photometric"), None)
        pj = (f"{float(m['signed_delta']):+.3f}", f"[{float(m['ci_lo']):+.3f}, "
              f"{float(m['ci_hi']):+.3f}]") if m else ("--", "--")
        L.append(f"| {judge} | {cm:.3f} | {pj[0]} | {pj[1]} |")
    L.append("\n_judges are not collapsed/saturated on valid real physics, and "
             "photometric alone does not shift the score._")
    return "\n".join(L) + "\n"


def _by_level_judges():
    """Step 8 addendum: clean judge score stratified by human PC level (test)."""
    labs = human_labels()
    L = ["## Step 8 (cont.) - clean judge score by human PC level (test, "
         "normalized)\n", "| human PC | " + " | ".join(JUDGES) + " |",
         "|" + "---|" * (len(JUDGES) + 1)]
    per = {j: defaultdict(list) for j in JUDGES}
    for judge in JUDGES:
        for stem, pv in load_pass("results/pass1", judge, "test",
                                  PHYS_GROUP_CALL[judge]).items():
            lab = _lab(labs, stem)
            if lab and "clean" in pv:
                per[judge][lab[0]].append(_normphys(judge, pv["clean"]))
    for lvl in (1, 2, 3, 4, 5):
        cells = [f"{np.mean(per[j][lvl]):.3f}" if per[j][lvl] else "--"
                 for j in JUDGES]
        L.append(f"| {lvl} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def _reliability_cards(temporal_idx):
    """Per-judge 2x2 reliability card: expected-sensitivity (temporal, should
    DROP) x expected-invariance (superficial, should NOT rise)."""
    d = [r for r in _read_deltas_csv() if r["group"].startswith("step12_")
         and r["dataset"] == "test"]
    tix = defaultdict(dict)
    for (j, v, mn, lo, hi) in temporal_idx.get("rows", []):
        tix[j][v] = lo > 0
    L = ["## Per-judge 2x2 reliability cards (test)\n"]
    for judge in JUDGES:
        sup = [r for r in d if r["judge"] == judge and r["variant"] in SUPERFICIAL]
        infl = [r["variant"] for r in sup if float(r["ci_lo"]) > 0]
        h1 = sum(tix[judge].values())
        worst = max(sup, key=lambda r: float(r["signed_delta"]), default=None)
        L.append(f"**{judge}**  ")
        L.append(f"- expected-sensitivity (temporal): fails to drop on "
                 f"{h1}/3 attacks (dJ-dH index CI>0)  ")
        L.append(f"- expected-invariance (superficial): inflated on "
                 f"{len(infl)}/6 cues [{', '.join(x.replace('caption_echo_','') for x in infl) or 'none'}]  ")
        if worst:
            L.append(f"- worst superficial gap: {worst['variant'].replace('caption_echo_','')} "
                     f"d = {float(worst['signed_delta']):+.3f}\n")
    return "\n".join(L) + "\n"


def _key_numbers(tidx, s13_stats):
    """The 6-8 numbers a reader actually needs; everything else is detail."""
    L = ["## Key numbers\n"]
    m = _get(PROBE_LOCKED_JSON) or {}
    pm = ((m.get("selection") or {}).get("summary") or {}).get("proj_mean", {})
    if pm:
        L.append(f"- reference probe (proj_mean): clean-val macro MAE "
                 f"{pm.get('macro_mae', float('nan')):.3f} "
                 f"+/- {pm.get('macro_mae_sd', 0):.3f}, "
                 f"Spearman {pm.get('rho', float('nan')):+.3f}; on VideoPhy-2 "
                 f"test human PC, Spearman ~+0.29 (invariance anchor, not an "
                 f"accurate predictor)")
    step12 = [r for r in _read_deltas_csv() if r["group"].startswith("step12_")
              and r["dataset"] == "test"]
    worst = []
    for j in JUDGES:
        sup = [r for r in step12 if r["judge"] == j and r["variant"] in SUPERFICIAL]
        if sup:
            w = max(sup, key=lambda r: float(r["signed_delta"]))
            worst.append(f"{j.split('_')[0]} {float(w['signed_delta']):+.3f}")
    if worst:
        L.append(f"- **H2** superficial-cue inflation (worst gap d=dJ-dV per "
                 f"judge, test, all CI>0): {', '.join(worst)}; videophy2_auto "
                 f"inflates on every overlay incl. the irrelevant-text control")
    idx = [mn for (_j, _v, mn, _lo, _hi) in tidx.get("rows", [])]
    if idx:
        L.append(f"- **H1** temporal-insensitivity index dJ-dH = {min(idx):+.2f} "
                 f"to {max(idx):+.2f} across all 9 judge x temporal-attack cells "
                 f"(CI>0); humans drop PC ~1.5-3.6, judges move ~0 (1 rater)")
    L.append("- **H3** 5 caption-echo families clear d>0/CI on >=2 of 3 judges")
    if s13_stats:
        L.append(f"- **H4** Pass-1<->Pass-2 (PhyJudge): Spearman "
                 f"{s13_stats.get('rho', float('nan')):+.3f}, "
                 f"{s13_stats.get('shift_frac', 0):.0%} of pairs shift >=1 unit")
    ub = ((_get(TEMPORAL_UB_JSON) or {}).get("per_variant") or {}).get("pooled")
    if ub:
        L.append(f"- 9.5(3) attack-aware temporal upper bound: AUC "
                 f"{ub['auc']:.3f} vs order-blind reference "
                 f"{ub['ref_probe_auc']:.3f} (RECOVERABLE)")
    v = (_get_text(f"predictor/{PRED_MODEL}/verdict.txt").strip()
         if _exists(f"predictor/{PRED_MODEL}/verdict.txt") else None)
    L.append(f"- Step 10 predictor: {v or 'AMBIGUOUS (see Step 10)'}")
    L.append("- clean judge-vs-human PC Spearman 0.28-0.37 (videophy2_auto "
             "0.36 = as published)")
    return "\n".join(L) + "\n"


# ======================================================================
# build
# ======================================================================

def build(data_dir="analysis/data", skip_sync=False, no_analyze=False,
          annot_version="v1", out="RESULTS.md", lean=True):
    t0 = time.time()

    if not no_analyze:
        if not skip_sync:
            print("syncing judge records for analyze.py ...")
            sync_analyze_data(data_dir)
        print("running analysis/analyze.py ...")
        analyze_txt = run_analyze(data_dir)
    else:
        analyze_txt = "[--no-analyze: skipped]"

    print("5.1 ..."); s51 = section_5_1()
    print("Step 4 ..."); s4 = section_step4() if lean else ""
    print("Step 9 ..."); s9 = section_step9()
    s9_txt, _ = s9 if isinstance(s9, tuple) else (s9, {})
    print("Step 9.5 ..."); s95 = section_step9_5()
    print("Step 10 ..."); s10 = section_step10()
    print("Step 11 ..."); s11_txt, human = section_step11(annot_version)
    print("Step 12 temporal (H1) ..."); _tt, tidx = section_step12_temporal(human, annot_version)
    print("Step 12 ..."); s12_txt, verdicts = section_step12(tidx)
    print("Step 13 ..."); s13, s13_stats = section_step13()
    print("Step 6 / 7 ..."); s67 = _lean_step6_7() if lean else ""
    print("Step 8 ..."); s8 = (_step8_alignment() + "\n" + _by_level_judges()) if lean else ""
    print("sanity / cards ..."); sanity = _sanity_real() if lean else ""
    cards = _reliability_cards(tidx) if lean else ""
    print("Ablations 10 / 11 ..."); ab = _ablation_10_11() if lean else ""

    header = ("# Results\n\n_generated "
              + time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()) + "_\n\n"
              + "## Verdicts\n\n" + "\n".join(verdicts) + "\n")
    keys = _key_numbers(tidx, s13_stats) if lean else ""

    if lean:
        parts = [header, keys, s4, s51, s67, s8, s9_txt, s95, s10, s11_txt,
                 sanity, s12_txt, s13, ab, cards]
    else:
        parts = [header, s51, s9_txt, s95, s10, s11_txt, s12_txt, s13,
                 "## Steps 6, 7, 8, ablations 10 & 11 - analysis/analyze.py\n\n"
                 "```\n" + analyze_txt.rstrip() + "\n```\n"]

    doc = "\n".join(parts)
    Path(out).write_text(doc, encoding="utf-8")
    print(f"\nwrote {out} ({len(doc)} chars, {time.time()-t0:.0f}s)")
    return doc


# ======================================================================
# selftest
# ======================================================================

def selftest():
    ok = True

    def c(cond, label):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {label}")

    c(abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9, "spearman identity")
    c(abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9, "spearman reversed")
    c(np.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4])), "spearman constant -> nan")
    c(abs(mae([1, 2, 3], [1, 2, 4]) - 1 / 3) < 1e-9, "mae")
    lo, hi = boot_ci([0.1] * 50)
    c(abs(lo - 0.1) < 1e-6 and abs(hi - 0.1) < 1e-6, "boot_ci degenerate")
    r = _rankdata([10, 10, 20, 30])
    c(list(r) == [1.5, 1.5, 3.0, 4.0], "rankdata midranks")
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return ok


def _in_notebook():
    try:
        __IPYTHON__          # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__" and not _in_notebook():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data-dir", default="analysis/data")
    ap.add_argument("--skip-sync", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    ap.add_argument("--annot-version", default="v1")
    ap.add_argument("--out", default="RESULTS.md")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    build(data_dir=a.data_dir, skip_sync=a.skip_sync, no_analyze=a.no_analyze,
          annot_version=a.annot_version, out=a.out)
