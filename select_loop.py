"""Reward selection / optimisation loop over the judges (reviewer request).

The main experiment shows that an overlay MOVES a judge. The reviewer's ask is
the operational consequence: put a small attacker in front of the judge, let it
pick the candidate that scores best, and show that **judge reward rises while
human physical plausibility stays flat or gets worse**.

Nothing here needs a GPU or a new render. Every candidate is a variant already
scored in `results/pass1`, so the loop is a re-reading of those records under
an argmax. That is the point: the attacker is not training anything, it is
choosing, and choosing is all it takes.

Three selection settings, and the difference between them is the whole
credibility of the section:

  oracle    per clip, take the best-scoring candidate. This is what an
            attacker with query access to the judge actually gets, and it is
            an upper bound: some of the gain is selection on the judge's own
            noise rather than on a real effect.
  transfer  choose ONE variant on a fit half of the clips by mean judge gain,
            then apply that single fixed variant to the held-out half. No
            per-clip judge access at evaluation time, so no selection on noise
            is possible. This is the honest headline number.
  null      the same argmax over a pool of candidates that differ by NOTHING
            but a re-encode ({clean, identity}). Whatever this returns is the
            selection-on-noise floor; an oracle gain is only interesting to the
            extent it clears it.

Three candidate pools:

  superficial  clean + the 6 expected-invariance variants. The attacker is
               constrained to manipulations that do not change the physics of
               the depicted event, so the human score is unchanged by
               construction and "human stays flat" is a claim about the attack
               taxonomy, not a hope about the data.
  all          clean + all 9 attacked variants. The attacker is unconstrained,
               so it can and does pick temporal attacks the judge tolerates --
               and there the human score genuinely GETS WORSE.
  null         clean + identity. The floor described above.

The human side is `clean human PC + dH(variant)`, where dH is the measured
per-attack mean human delta from the Step 11 annotation study (PC points), and
dH(clean) = 0. When the annotations are unavailable the superficial pool falls
back to dH = 0 -- correct by construction, and stated in the output -- and the
`all` pool is skipped entirely, because a temporal variant's human score is
exactly the thing that cannot be assumed.

Best-of-K is computed EXACTLY, not by resampling: over a uniformly random
K-subset of an n-candidate pool the probability that the i-th smallest
candidate is the max is C(i-1,K-1)/C(n,K), so one weighted sum over the sorted
candidates gives the expectation. The same weights applied to the HUMAN value
paired with each candidate give the expected human score of whatever the
attacker picked -- which is the number the whole section is about.

Judge deltas are reported in the judge's NATIVE units alongside the normalised
ones, because an equal normalised movement across two different scales is not
an equal behavioural movement.

    python select_loop.py --selftest
    python select_loop.py                        # -> analysis/select_loop.{md,json}
    python select_loop.py --judges phyjudge_9b --datasets test
    python select_loop.py --push                 # also -> s3://<bucket>/analysis/

In Colab: import select_loop; select_loop.run()
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
PASS1 = "results/pass1"
CSV_KEY = "datasets/videophy2_test/_metadata/videophy2_test.csv"
OUT_JSON = "analysis/select_loop.json"
OUT_MD = "analysis/select_loop.md"
S3_OUT_PREFIX = "analysis"

JUDGES = ["phyjudge_9b", "vila_ewm", "videophy2_auto"]
DATASETS = ["test", "implausibench_real", "implausibench_implausible"]

_PHYS_LAWS = ("gravity", "inertia", "momentum", "impenetrability", "collision",
              "material", "buoyancy", "displacement", "flow_dynamics",
              "boundary_interaction", "fluid_continuity", "reflection", "shadow")
# The physics construct group per judge. Calls are NEVER pooled across
# constructs -- mixing semantic adherence into a physics score is the mistake
# stats.py's GROUPS exists to prevent. Duplicated from there and from
# analysis/analyze.py on purpose: this file imports nothing local so it runs in
# a bare Colab runtime, and the cost is that the four lists must move together.
PHYS_CALL = {
    "phyjudge_9b": lambda c: c in _PHYS_LAWS,
    "vila_ewm": lambda c: c.startswith("physical_laws_"),
    "videophy2_auto": lambda c: c == "PC",
}
# Native scale of that group. vila's physics calls are booleans, so its
# "score" is a pass fraction and is already on 0-1 -- which is exactly why the
# native column matters: +0.05 means something different on each of these.
PHYS_LO = {"phyjudge_9b": 1.0, "vila_ewm": 0.0, "videophy2_auto": 1.0}
PHYS_SPAN = {"phyjudge_9b": 4.0, "vila_ewm": 1.0, "videophy2_auto": 4.0}
PHYS_UNIT = {"phyjudge_9b": "law pts (1-5)", "vila_ewm": "pass frac (0-1)",
             "videophy2_auto": "PC pts (1-5)"}

TEMPORAL = ["shuffle", "reverse", "freeze"]
SUPERFICIAL = ["photometric", "caption_echo_rubric_vocab",
               "caption_echo_score_anchor_positive",
               "caption_echo_authoritative_claim",
               "caption_echo_score_anchor_negative",
               "caption_echo_control_irrelevant"]
CONTROL = "identity"
# pool[0] is ALWAYS "clean" -- the gain columns difference against it by index
POOLS = {
    "superficial": ["clean"] + SUPERFICIAL,
    "all": ["clean"] + SUPERFICIAL + TEMPORAL,
    "null": ["clean", CONTROL],
}
HUMAN_SPAN = 4.0          # PC is an integer 1-5
TOP_Q = (0.10, 0.25, 0.50)
N_BOOT = 2000
SEED = 0

s3 = boto3.client("s3", config=Config(
    max_pool_connections=READ_WORKERS + 8,
    retries={"max_attempts": 5, "mode": "standard"}))


# ======================================================================
# S3 / loading
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


def load_scores(judge, ds, prefix=PASS1):
    """{stem: {variant: native physics-group mean}} for one judge x dataset."""
    pred, out = PHYS_CALL[judge], {}
    for rec in _get_many([k for k in _list(f"{prefix}/{judge}/{ds}/")
                          if k.endswith(".json")]):
        if not rec:
            continue
        per = {}
        for var, r in (rec.get("runs") or {}).items():
            xs = []
            for cid, c in (r.get("calls") or {}).items():
                if not pred(cid):
                    continue
                p = c.get("parsed")
                if isinstance(p, bool):
                    xs.append(1.0 if p else 0.0)
                elif isinstance(p, (int, float)):
                    xs.append(float(p))
            if xs:
                per[var] = float(np.mean(xs))
        if per:
            out[rec.get("clip")] = per
    return out


def human_pc():
    """{stem: human PC 1-5} from videophy2_test.csv.

    VideoPhy-2 test is the only corpus carrying paired human labels, so the
    human half of this loop is test-only by construction; the other two
    corpora report the judge side and say so.
    """
    import csv
    import io as _io
    try:
        txt = s3.get_object(Bucket=BUCKET, Key=CSV_KEY)["Body"].read().decode()
    except Exception:
        return {}
    labs = {}
    for row in csv.DictReader(_io.StringIO(txt)):
        stem = os.path.splitext(os.path.basename(row.get("video_url", "")))[0]
        try:
            labs[stem] = float(row["pc"])
        except (ValueError, KeyError, TypeError):
            pass
    return labs


def human_deltas(version="v1"):
    """{variant: mean human dPC in PC points} from the Step 11 annotations.

    Returns {} when the study has not been read. Callers then fall back to the
    by-construction dH = 0 for superficial variants and drop the temporal half
    entirely -- stated in the output rather than silently assumed.
    """
    per = defaultdict(list)
    for key in _list(f"annotations/{version}/"):
        if not key.endswith(".jsonl"):
            continue
        try:
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()
        except Exception:
            continue
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("skipped"):
                continue
            a, b = r.get("pc_clean"), r.get("pc_variant")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                per[r.get("variant")].append(float(b) - float(a))
    return {v: float(np.mean(d)) for v, d in per.items() if d and v}


# ======================================================================
# best-of-K, exactly
# ======================================================================

def bok_weights(n, k):
    """P(the i-th smallest of n candidates is the max of a random K-subset).

    Exact, so the best-of-K curve carries no Monte-Carlo error of its own and
    two runs of this file agree to the last digit.
    """
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k} n={n}")
    denom = math.comb(n, k)
    return np.array([math.comb(i, k - 1) / denom for i in range(n)])


def best_of_k(judge_vals, human_vals, k):
    """-> (E[judge score of the pick], E[human score of the pick]).

    The human value is carried through the SAME sort and the SAME weights, so
    it answers "what did the attacker actually get" rather than "what is the
    average human score of the pool" -- which is the difference between a
    gameability result and a tautology. Ties in the judge value break by pool
    order (stable sort), so the answer is deterministic.
    """
    j = np.asarray(judge_vals, dtype=float)
    order = np.argsort(j, kind="stable")
    w = bok_weights(len(j), k)
    e_j = float(np.dot(w, j[order]))
    if human_vals is None:
        return e_j, None
    h = np.asarray(human_vals, dtype=float)
    return e_j, float(np.dot(w, h[order]))


# ======================================================================
# the loop
# ======================================================================

def _fit_half(stem):
    """Deterministic 50/50 split for the transfer policy, on the clip stem.

    Hash-based rather than rng.shuffle so a rerun, a different judge and a
    different dataset all agree on which clips are fit and which are eval --
    the transfer variant is chosen once per (judge, dataset) and has to be
    evaluated on clips it was not chosen on.
    """
    return int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16) % 2 == 0


def _boot_ci(x, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap resampling CLIPS -- the right unit, since every
    quantity here is a within-clip paired difference."""
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def build_candidates(scores, pool, labs, dh, require_human=False):
    """-> [(stem, pool, [judge native], [human PC] or None)].

    A clip contributes only if EVERY variant in the pool was scored: the
    attacker's budget is the pool size, so a clip with a hole in it is a
    different (smaller) experiment silently averaged into the same number.
    """
    out = []
    for stem, per in scores.items():
        if not stem or not all(v in per for v in pool):
            continue
        key = stem.removesuffix("_result")
        pc = labs.get(key, labs.get(stem))
        if require_human and pc is None:
            continue
        jv = [per[v] for v in pool]
        hv = None if pc is None else [pc + dh.get(v, 0.0) for v in pool]
        out.append((stem, pool, jv, hv))
    return out


def oracle_curve(cands, judge):
    """Judge reward and human PC as a function of the attacker's budget K.

    -> (curve rows, per-clip judge gains at full budget, per-clip human gains).
    The gains are returned per clip rather than averaged so the caller can
    bootstrap them with the pairing intact.
    """
    lo, span = PHYS_LO[judge], PHYS_SPAN[judge]
    rows, n = [], len(cands[0][1])
    for k in range(1, n + 1):
        js, hs = [], []
        for _stem, _pool, jv, hv in cands:
            ej, eh = best_of_k(jv, hv, k)
            js.append(ej)
            if eh is not None:
                hs.append(eh)
        jlo, jhi = _boot_ci(js)
        rows.append(dict(k=k, n=len(js),
                         judge_native=float(np.mean(js)),
                         judge_norm=(float(np.mean(js)) - lo) / span,
                         judge_ci_native=[jlo, jhi],
                         human_pc=float(np.mean(hs)) if hs else None,
                         n_human=len(hs)))
    gains_j, gains_h = [], []
    for _stem, _pool, jv, hv in cands:
        ej, eh = best_of_k(jv, hv, len(jv))
        gains_j.append((ej - jv[0]) / span)          # pool[0] is clean
        if hv is not None:
            gains_h.append((eh - hv[0]) / HUMAN_SPAN)
    return rows, gains_j, gains_h


def transfer_policy(cands, judge):
    """Pick ONE variant on the fit half, evaluate it on the held-out half.

    No per-clip judge access at evaluation, so nothing here can be selection
    on the judge's own noise -- which is exactly the objection the oracle
    number cannot answer for itself. `clean` is excluded from the pick: an
    attack policy that submits the unmodified clip is not an attack, and
    letting it win would report a 0 gain as a finding.
    """
    span, pool = PHYS_SPAN[judge], cands[0][1]
    fit = [c for c in cands if _fit_half(c[0])]
    ev = [c for c in cands if not _fit_half(c[0])]
    if len(fit) < 5 or len(ev) < 5:
        return None
    means = []
    for i, v in enumerate(pool):
        if v == "clean":
            means.append(-np.inf)
            continue
        means.append(float(np.mean([jv[i] - jv[0] for _s, _p, jv, _h in fit])))
    if not np.isfinite(max(means)):
        return None
    pick = int(np.argmax(means))
    gj = [(jv[pick] - jv[0]) / span for _s, _p, jv, _h in ev]
    gh = [(hv[pick] - hv[0]) / HUMAN_SPAN for _s, _p, _j, hv in ev
          if hv is not None]
    jlo, jhi = _boot_ci(gj)
    hlo, hhi = _boot_ci(gh) if gh else (float("nan"), float("nan"))
    return dict(variant=pool[pick], n_fit=len(fit), n_eval=len(ev),
                fit_gain_native=float(means[pick]),
                judge_gain_norm=float(np.mean(gj)),
                judge_gain_native=float(np.mean(gj) * span),
                judge_ci=[jlo, jhi],
                human_gain_norm=float(np.mean(gh)) if gh else None,
                human_gain_pc=float(np.mean(gh) * HUMAN_SPAN) if gh else None,
                human_ci=[hlo, hhi] if gh else None,
                n_human=len(gh))


def video_selection(cands, judge, qs=TOP_Q):
    """Leaderboard gaming: submit the top-q clips by judge score.

    Honest: rank by the judge's CLEAN score. Gamed: rank by the judge's score
    on the best overlay for that clip, which is what a submitter optimising
    against the judge would hand in. The decision-relevant number is the mean
    HUMAN PC of the selected set -- if that does not rise with the judge
    score, the judge is selecting on the attack rather than on physics.
    """
    lo, span = PHYS_LO[judge], PHYS_SPAN[judge]
    ranked = [(jv[0], max(jv), hv[0]) for _s, _p, jv, hv in cands
              if hv is not None]
    if len(ranked) < 20:
        return []
    clean_j = np.array([r[0] for r in ranked])
    gamed_j = np.array([r[1] for r in ranked])
    pc = np.array([r[2] for r in ranked])
    rows = []
    for q in qs:
        m = max(5, int(round(q * len(ranked))))
        ci = np.argsort(-clean_j, kind="stable")[:m]
        gi = np.argsort(-gamed_j, kind="stable")[:m]
        rows.append(dict(
            q=q, n_selected=m, n_pool=len(ranked),
            clean_judge_native=float(clean_j[ci].mean()),
            clean_judge_norm=(float(clean_j[ci].mean()) - lo) / span,
            clean_human_pc=float(pc[ci].mean()),
            gamed_judge_native=float(gamed_j[gi].mean()),
            gamed_judge_norm=(float(gamed_j[gi].mean()) - lo) / span,
            gamed_human_pc=float(pc[gi].mean()),
            overlap=float(len(set(ci.tolist()) & set(gi.tolist())) / m)))
    return rows


def run_cell(judge, ds, pool_name, scores, labs, dh):
    pool = POOLS[pool_name]
    cands = build_candidates(scores, pool, labs, dh)
    if len(cands) < 10:
        return None
    curve, gains_j, gains_h = oracle_curve(cands, judge)
    jlo, jhi = _boot_ci(gains_j)
    hlo, hhi = _boot_ci(gains_h) if gains_h else (float("nan"), float("nan"))
    return dict(
        judge=judge, dataset=ds, pool=pool_name, pool_variants=pool,
        n_clips=len(cands), curve=curve,
        oracle=dict(judge_gain_norm=float(np.mean(gains_j)),
                    judge_gain_native=float(np.mean(gains_j) * PHYS_SPAN[judge]),
                    judge_ci=[jlo, jhi],
                    human_gain_norm=float(np.mean(gains_h)) if gains_h else None,
                    human_gain_pc=(float(np.mean(gains_h) * HUMAN_SPAN)
                                   if gains_h else None),
                    human_ci=[hlo, hhi] if gains_h else None,
                    n_human=len(gains_h)),
        transfer=transfer_policy(cands, judge),
        video=video_selection(cands, judge))


def run(judges=None, datasets=None, pools=None, annot_version="v1",
        out_json=OUT_JSON, out_md=OUT_MD, push=False):
    judges = judges or JUDGES
    datasets = datasets or DATASETS
    pools = pools or list(POOLS)

    dh = human_deltas(annot_version)
    labs = human_pc()
    src = ("from annotations/" + annot_version if dh else
           "none found -- superficial falls back to dH=0 by construction, "
           "and the unconstrained pool is skipped")
    print(f"human PC labels: {len(labs)} clips; measured dH for "
          f"{len(dh)} variants ({src})")
    if dh:
        print("  " + "  ".join(f"{v}:{d:+.2f}" for v, d in sorted(dh.items())))

    have_temporal_dh = any(v in dh for v in TEMPORAL)
    cells = []
    for judge in judges:
        for ds in datasets:
            scores = load_scores(judge, ds)
            if not scores:
                print(f"  {judge}/{ds}: no pass-1 records")
                continue
            for p in pools:
                # `all` crosses the taxonomy, so it needs a MEASURED human
                # delta for the temporal half. Without one there is no honest
                # human number for a scrambled clip, and reporting it with dH
                # assumed 0 would claim humans do not mind a shuffled video.
                if p == "all" and not have_temporal_dh:
                    print(f"  {judge}/{ds}/all: skipped (no measured temporal dH)")
                    continue
                cell = run_cell(judge, ds, p, scores, labs, dh)
                if cell:
                    cells.append(cell)
                    print(f"  {judge}/{ds}/{p}: n={cell['n_clips']}")

    doc = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        annot_version=annot_version, human_deltas=dh,
        native_units=PHYS_UNIT, cells=cells)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    md = markdown(doc)
    Path(out_md).write_text(md, encoding="utf-8")
    print(f"\nwrote {out_json} and {out_md}")
    if push:
        for path in (out_json, out_md):
            s3.put_object(Bucket=BUCKET,
                          Key=f"{S3_OUT_PREFIX}/{Path(path).name}",
                          Body=Path(path).read_bytes())
        print(f"pushed -> s3://{BUCKET}/{S3_OUT_PREFIX}/")
    return doc


# ======================================================================
# reporting
# ======================================================================

def _fmt(x, digits=4, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def markdown(doc):
    L = ["## Reward selection loop (judge reward vs human plausibility)\n",
         "_An attacker picks, per clip, the already-scored variant the judge "
         "likes best. No retraining and no new renders -- selection alone._\n"]
    dh = doc.get("human_deltas") or {}
    if dh:
        L.append("Human side: clean human PC + the measured per-attack mean dH "
                 "from the Step 11 study ("
                 + ", ".join(f"{v} {d:+.2f}" for v, d in sorted(dh.items()))
                 + ").\n")
    else:
        L.append("Human side: no Step 11 deltas available, so superficial "
                 "variants use dH = 0 -- correct by construction, since they "
                 "do not change the depicted physics -- and the unconstrained "
                 "pool is omitted rather than assumed.\n")

    heads = {"superficial": "attacker constrained to physics-preserving cues",
             "all": "attacker unconstrained (temporal attacks allowed)",
             "null": "floor: candidates differ only by a re-encode"}
    for pool in ("superficial", "all", "null"):
        cells = [c for c in doc["cells"] if c["pool"] == pool]
        if not cells:
            continue
        L.append(f"### pool = `{pool}` -- {heads[pool]}\n")
        L.append("| judge | dataset | n | oracle dJ (norm) | 95% CI | "
                 "oracle dJ (native) | oracle dHuman (PC pts) | transfer pick | "
                 "transfer dJ (norm) | transfer dJ (native) | 95% CI |")
        L.append("|" + "---|" * 11)
        for c in cells:
            o, t = c["oracle"], c["transfer"]
            unit = doc["native_units"][c["judge"]]
            tr = ["--", "--", "--", "--"]
            if t:
                tr = [t["variant"].replace("caption_echo_", ""),
                      _fmt(t["judge_gain_norm"]),
                      f"{_fmt(t['judge_gain_native'])} {unit}",
                      f"[{_fmt(t['judge_ci'][0])}, {_fmt(t['judge_ci'][1])}]"]
            L.append(f"| {c['judge']} | {c['dataset']} | {c['n_clips']} | "
                     f"{_fmt(o['judge_gain_norm'])} | "
                     f"[{_fmt(o['judge_ci'][0])}, {_fmt(o['judge_ci'][1])}] | "
                     f"{_fmt(o['judge_gain_native'])} {unit} | "
                     f"{_fmt(o['human_gain_pc'], 3)} | "
                     + " | ".join(tr) + " |")
        L.append("")

    L.append("### best-of-K curve (VideoPhy-2 test, physics-preserving pool)\n")
    L.append("_K = how many candidate overlays the attacker may try; exact "
             "expectation over uniformly random K-subsets. Judge is "
             "normalised 0-1, human is native PC 1-5._\n")
    kmax = 7
    L.append("| judge | row | " + " | ".join(f"K={k}" for k in range(1, kmax + 1))
             + " |")
    L.append("|" + "---|" * (kmax + 2))
    for c in doc["cells"]:
        if c["dataset"] != "test" or c["pool"] != "superficial":
            continue
        cur = c["curve"][:kmax]
        L.append(f"| {c['judge']} | judge (norm) | "
                 + " | ".join(_fmt(r["judge_norm"], 4, False) for r in cur) + " |")
        if cur[0]["human_pc"] is not None:
            L.append(f"| {c['judge']} | human PC | "
                     + " | ".join(_fmt(r["human_pc"], 3, False) for r in cur) + " |")
    L.append("")

    vid = [c for c in doc["cells"] if c["video"] and c["pool"] == "superficial"]
    if vid:
        L.append("### video selection: submit the top-q clips by judge score\n")
        L.append("| judge | dataset | q | n | judge norm (clean rank) | "
                 "judge norm (gamed rank) | human PC (clean rank) | "
                 "human PC (gamed rank) | overlap |")
        L.append("|" + "---|" * 9)
        for c in vid:
            for r in c["video"]:
                L.append(f"| {c['judge']} | {c['dataset']} | {r['q']:.2f} | "
                         f"{r['n_selected']} | "
                         f"{_fmt(r['clean_judge_norm'], 4, False)} | "
                         f"{_fmt(r['gamed_judge_norm'], 4, False)} | "
                         f"{_fmt(r['clean_human_pc'], 3, False)} | "
                         f"{_fmt(r['gamed_human_pc'], 3, False)} | "
                         f"{r['overlap']:.2f} |")
        L.append("\n_Reward rises from clean rank to gamed rank; the human PC "
                 "of the selected set is the column that must NOT rise with "
                 "it. `overlap` is the fraction of the honest top-q that "
                 "survives into the gamed one._\n")

    L.append("### how to read this\n")
    L.append("- `oracle` is an upper bound: it selects using the same scores "
             "it reports, so part of the gain is the judge's own noise. The "
             "`null` pool measures exactly that part -- subtract it.")
    L.append("- `transfer` selects one variant on a fit half and reports the "
             "held-out half, so no selection-on-noise is possible. It is the "
             "number to quote.")
    L.append("- The claim is the PAIR of columns: dJ > 0 while dHuman is 0 "
             "(constrained pool) or negative (unconstrained pool). Either "
             "column alone shows nothing.")
    L.append("- Native and normalised are both given because the three judges "
             "do not share a scale: vila's physics score is a pass fraction "
             "on 0-1, the other two are 1-5.\n")
    return "\n".join(L) + "\n"


# ======================================================================
# selftest
# ======================================================================

def selftest():
    ok = True

    def c(cond, label):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {label}")

    # ---- best-of-K weights
    c(np.allclose(bok_weights(5, 1), 0.2), "bok K=1 is uniform")
    c(np.allclose(bok_weights(5, 5), [0, 0, 0, 0, 1]),
      "bok K=n puts all mass on the max")
    sums_ok = all(np.isclose(bok_weights(n, k).sum(), 1.0)
                  for n in (2, 3, 7, 11) for k in range(1, n + 1))
    c(sums_ok, "bok weights sum to 1 over n=2..11, every k")
    w = bok_weights(6, 3)
    c(w[0] == 0 and w[1] == 0 and np.all(np.diff(w[2:]) > 0),
      "bok K=3 zeroes the bottom K-1 and increases with rank")
    try:
        bok_weights(3, 4)
        c(False, "bok rejects k>n")
    except ValueError:
        c(True, "bok rejects k>n")

    # ---- best_of_k
    j = [1.0, 2.0, 3.0, 4.0]
    c(abs(best_of_k(j, None, 4)[0] - 4.0) < 1e-12, "best_of_n of 1..4 is 4")
    c(abs(best_of_k(j, None, 1)[0] - 2.5) < 1e-12, "best_of_1 is the mean")
    vals = [best_of_k(j, None, k)[0] for k in range(1, 5)]
    c(all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])),
      "best-of-K is monotone non-decreasing in K")
    ej, eh = best_of_k([1.0, 5.0], [9.0, 0.0], 2)
    c(abs(ej - 5.0) < 1e-12 and abs(eh - 0.0) < 1e-12,
      "human value is carried by the JUDGE's ranking, not its own")
    c(abs(best_of_k([5.0, 5.0], [1.0, 2.0], 2)[1] - 2.0) < 1e-12,
      "judge ties break by pool order (stable sort)")

    # ---- fit/eval split
    stems = [f"clip_{i:04d}" for i in range(2000)]
    a = [_fit_half(s) for s in stems]
    c(a == [_fit_half(s) for s in stems], "fit split is deterministic")
    c(0.45 < sum(a) / len(a) < 0.55, "fit split is ~balanced")

    lo, hi = _boot_ci([0.25] * 40)
    c(abs(lo - 0.25) < 1e-9 and abs(hi - 0.25) < 1e-9, "boot_ci degenerate")

    # ---- candidate construction
    sc = {"a": {"clean": 3, "photometric": 3}, "b": {"clean": 3}}
    got = build_candidates(sc, ["clean", "photometric"], {"a": 4.0, "b": 4.0}, {})
    c(len(got) == 1 and got[0][0] == "a",
      "a clip missing a pool member is dropped, not padded")
    c(got[0][3] == [4.0, 4.0], "dH=0 leaves the human value flat")
    got = build_candidates(sc, ["clean", "photometric"], {"a": 4.0},
                           {"photometric": -0.5})
    c(got[0][3] == [4.0, 3.5], "measured dH applies to the variant only")
    got = build_candidates({"a_result": {"clean": 3}}, ["clean"],
                           {"a": 2.0}, {})
    c(got and got[0][3] == [2.0], "the _result stem suffix still finds a label")

    # ---- a pool of identical candidates must show exactly zero gain
    flat = [(f"s{i}", ["clean", "identity"], [3.0, 3.0], [4.0, 4.0])
            for i in range(30)]
    rows, gj, gh = oracle_curve(flat, "videophy2_auto")
    c(abs(np.mean(gj)) < 1e-12 and abs(np.mean(gh)) < 1e-12,
      "identical candidates give exactly zero gain at every K")
    c(rows[0]["judge_norm"] == 0.5, "native -> norm uses the judge's own scale")
    c(rows[0]["judge_native"] == 3.0, "native column stays in native units")

    # ---- transfer picks on fit, reports on eval, never picks clean
    rng = np.random.default_rng(0)
    cands = []
    for i in range(200):
        base = float(rng.normal(3, 0.3))
        cands.append((f"t{i:04d}", ["clean", "photometric", "caption_echo_x"],
                      [base, base + 0.05, base + 0.40], [4.0, 4.0, 4.0]))
    t = transfer_policy(cands, "videophy2_auto")
    c(t["variant"] == "caption_echo_x", "transfer picks the strongest variant")
    c(abs(t["judge_gain_norm"] - 0.10) < 1e-9, "transfer gain is normalised")
    c(abs(t["judge_gain_native"] - 0.40) < 1e-9, "transfer gain is also native")
    c(t["human_gain_norm"] == 0.0, "transfer human gain is flat when dH=0")
    c(t["n_fit"] + t["n_eval"] == 200 and t["n_fit"] > 50,
      "transfer fit/eval partition the clips")
    only_clean = [(s, ["clean"], [v[0]], h) for s, _p, v, h in cands]
    c(transfer_policy(only_clean, "videophy2_auto") is None,
      "transfer returns None when clean is the only candidate")

    # ---- video selection
    v = video_selection(cands, "videophy2_auto", qs=(0.25,))[0]
    c(v["gamed_judge_native"] > v["clean_judge_native"],
      "gamed ranking beats clean ranking on judge reward")
    c(abs(v["gamed_human_pc"] - v["clean_human_pc"]) < 1e-12,
      "gamed ranking does not improve human PC when dH=0")
    c(0.0 <= v["overlap"] <= 1.0, "overlap is a fraction")

    # ---- an attacker that can pick a temporal attack must LOWER human PC
    tc = [(f"u{i:04d}", ["clean", "shuffle"], [3.0, 3.4], [4.0, 4.0 - 1.5])
          for i in range(40)]
    _rows, gj2, gh2 = oracle_curve(tc, "videophy2_auto")
    c(np.mean(gj2) > 0 and np.mean(gh2) < 0,
      "unconstrained pool: judge up, human down")

    c(markdown(dict(human_deltas={}, native_units=PHYS_UNIT,
                    cells=[])).startswith("## Reward selection loop"),
      "markdown renders with no cells")

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return ok


def _in_notebook():
    try:
        __IPYTHON__          # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__" and not _in_notebook():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--judges", nargs="+", default=None, choices=JUDGES)
    ap.add_argument("--datasets", nargs="+", default=None, choices=DATASETS)
    ap.add_argument("--pools", nargs="+", default=None, choices=list(POOLS))
    ap.add_argument("--annot-version", default="v1")
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--push", action="store_true",
                    help="also upload both outputs to s3://<bucket>/analysis/")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run(judges=a.judges, datasets=a.datasets, pools=a.pools,
        annot_version=a.annot_version, out_json=a.out_json, out_md=a.out_md,
        push=a.push)
