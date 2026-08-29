"""Step 13 Part A -- the failure-mode case gallery and its two-coder interface.

Three stages, each idempotent, all read-only except the writes named:

  1. select()   picks the ~15-20 highest-divergence (clip, attack) cases plus
                the worst clean-clip over/under-scorers, per judge. Writes the
                FROZEN case set to s3://.../case_gallery/cases_<v>.json -- build
                it ONCE, both coders derive from it.
  2. code()     the ipywidgets UI. Each coder sees every case (clean video +,
                for an attack case, the attacked variant; PhyJudge's Pass-2
                rationale where it exists) and labels the judge's failure as
                temporal/perceptual or superficial-cue/linguistic. Blinded only
                to the OTHER coder -- not to the attack, which the video shows
                anyway. Resumable; one S3 PUT per submit.
                Writes s3://.../case_codes/<v>/<coder>.jsonl
  3. report()   Cohen's kappa between the two coders, the failure-mode
                distribution by judge / track / attack family, and the
                disagreement list for adjudication.
     adjudicate()  a third pass over disagreements only -> final labels.

Self-contained (no import from annotate.py / judge_harness.py) so it pastes
into one Colab cell. Needs AWS creds (Colab secrets or ambient). Selection
needs pandas-free numpy only; the UI needs ipywidgets.

    python case_gallery.py --selftest
    python case_gallery.py --select                 # writes cases_v1.json
    python case_gallery.py --report
    In Colab:  import case_gallery as G
               G.select()                            # once
               G.code(coder="c1")  /  G.code(coder="c2")
               G.report()
               G.adjudicate(adjudicator="c3")        # if there are disagreements
"""

import argparse
import hashlib
import io
import itertools
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
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
s3 = boto3.client("s3", config=Config(max_pool_connections=40,
                                      retries={"max_attempts": 5, "mode": "standard"}))

JUDGES = ["phyjudge_9b", "vila_ewm", "videophy2_auto"]
DATASETS = ["test", "implausibench_real", "implausibench_implausible"]
DS_TO_DVKEY = {"test": "videophy2_test", "implausibench_real": "implausibench_real",
               "implausibench_implausible": "implausibench_implausible"}
DS_TO_ATTACKDIR = {"test": "test", "implausibench_real": "implausibench_real",
                   "implausibench_implausible": "implausibench_implausible"}
CSV_KEY = "datasets/videophy2_test/_metadata/videophy2_test.csv"
DATASET_PREFIXES = {
    "test": "datasets/videophy2_test/",
    "implausibench_real": "datasets/implausibench/ImplausiBench/real/",
    "implausibench_implausible": "datasets/implausibench/ImplausiBench/implausible/",
}
VIDEO_SUFFIXES = (".mp4", ".webm", ".avi", ".mov", ".mkv")
_SRC_CACHE = {}
DV_KEY = "reference/probe_locked/dv.json"
PASS1 = "results/pass1"
PASS2 = "results/pass2"

PHYJUDGE_LAWS = ["gravity", "inertia", "momentum", "impenetrability", "collision",
                 "material", "buoyancy", "displacement", "flow_dynamics",
                 "boundary_interaction", "fluid_continuity", "reflection", "shadow"]
PHYSICS_CALL = {
    "phyjudge_9b": lambda c: c in PHYJUDGE_LAWS,
    "vila_ewm": lambda c: c.startswith("physical_laws_"),
    "videophy2_auto": lambda c: c == "PC",
}
SCALE_SPAN = {"phyjudge_9b": 4.0, "vila_ewm": 1.0, "videophy2_auto": 4.0}
SCALE_LO = {"phyjudge_9b": 1.0, "vila_ewm": 0.0, "videophy2_auto": 1.0}
TEMPORAL = ["shuffle", "reverse", "freeze"]
SUPERFICIAL = ["photometric", "caption_echo_rubric_vocab",
               "caption_echo_score_anchor_positive",
               "caption_echo_authoritative_claim",
               "caption_echo_score_anchor_negative",
               "caption_echo_control_irrelevant"]
ATTACKS = TEMPORAL + SUPERFICIAL

CASES_KEY = "case_gallery/cases_{v}.json"
CODES_KEY = "case_codes/{v}/{coder}.jsonl"
TMP = Path(os.environ.get("CASE_TMP", "./tmp_cases"))

FAILURE_MODES = [
    ("temporal_perceptual",
     "Temporal / perceptual -- the judge missed what the video actually shows "
     "(scrambled order, frozen motion, a physical event it did not see)."),
    ("superficial_cue_linguistic",
     "Superficial-cue / linguistic shortcut -- the judge followed the caption, "
     "an overlay, a score anchor, or wording rather than the dynamics."),
    ("both",
     "Both are clearly present."),
    ("neither_ambiguous",
     "Neither / ambiguous -- cannot attribute the failure to one mode."),
]
FM_KEYS = [k for k, _ in FAILURE_MODES]


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


def _get_text(key):
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()


def _get_many(keys):
    if not keys:
        return []
    with ThreadPoolExecutor(min(32, len(keys))) as ex:
        return list(ex.map(_get, keys))


def _exists(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def _u01(*parts):
    h = hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


# ======================================================================
# inputs for selection
# ======================================================================

def _judge_physics(prefix, judge, ds):
    """{stem: {variant: mean physics-group parsed score}}."""
    pred = PHYSICS_CALL[judge]
    out = {}
    for rec in _get_many([k for k in _list(f"{prefix}/{judge}/{ds}/")
                          if k.endswith(".json")]):
        if not rec:
            continue
        per = {}
        for var, r in rec.get("runs", {}).items():
            xs = []
            for cid, c in r.get("calls", {}).items():
                if not pred(cid):
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


def _human_labels():
    import csv
    labs = {}
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


def _dv_index():
    """{ds: {stem: {variant: d_norm}}} from the locked reference."""
    dv = _get(DV_KEY)
    if not dv:
        return {}
    out = {}
    for dvkey, block in dv.get("datasets", {}).items():
        ds = next((k for k, v in DS_TO_DVKEY.items() if v == dvkey), dvkey)
        out[ds] = {stem: {v: r["d_norm"] for v, r in c.get("variants", {}).items()}
                   for stem, c in block.get("clips", {}).items()}
    return out


# ======================================================================
# selection
# ======================================================================

def _norm(judge, x):
    return (x - SCALE_LO[judge]) / SCALE_SPAN[judge]


def select(version="v1", n_superficial=2, n_temporal=2, n_clean_over=1,
           n_clean_under=1, seed="cg-2026", push_to_s3=True):
    """Freeze the case set. Attack track: per judge, the biggest ΔJ−ΔV on a
    superficial attack (inflation) and the smallest score drop on a temporal
    one (blindness). Clean track: per judge, the worst over- and under-scorers
    vs human PC. ~15-20 cases total."""
    labs = _human_labels()
    dv = _dv_index()
    if not dv:
        print("NOTE reference/probe_locked/dv.json missing -- attack-track "
              "gaps will use ΔJ alone (no reference subtraction)")

    cases = []
    for judge in JUDGES:
        span = SCALE_SPAN[judge]
        rows_sup, rows_tmp, rows_clean = [], [], []
        for ds in DATASETS:
            jp = _judge_physics(PASS1, judge, ds)
            for stem, per in jp.items():
                jc = per.get("clean")
                if jc is None:
                    continue
                # clean track
                lab = _lab(labs, stem)
                if lab is not None:
                    resid = _norm(judge, jc) - (lab[0] - 1) / 4.0
                    rows_clean.append((abs(resid), resid, ds, stem, jc, lab))
                # attack track
                for v in ATTACKS:
                    jv = per.get(v)
                    if jv is None:
                        continue
                    dJ = (jv - jc) / span
                    dV = dv.get(ds, {}).get(stem, {}).get(v)
                    gap = dJ - dV if dV is not None else dJ
                    row = (ds, stem, v, jc, jv, dJ, dV, gap)
                    if v in SUPERFICIAL:
                        rows_sup.append((gap, row))         # want most positive
                    else:
                        rows_tmp.append((dJ, row))          # want least negative

        rows_sup.sort(key=lambda t: -t[0])
        rows_tmp.sort(key=lambda t: -t[0])          # dJ descending: barely moved
        rows_clean_over = sorted((r for r in rows_clean if r[1] > 0),
                                 key=lambda t: -t[1])
        rows_clean_under = sorted((r for r in rows_clean if r[1] < 0),
                                  key=lambda t: t[1])

        picks = ([("attack_superficial", r) for _, r in rows_sup[:n_superficial]]
                 + [("attack_temporal", r) for _, r in rows_tmp[:n_temporal]]
                 + [("clean_over", r) for r in rows_clean_over[:n_clean_over]]
                 + [("clean_under", r) for r in rows_clean_under[:n_clean_under]])

        for track, r in picks:
            if track.startswith("attack"):
                ds, stem, v, jc, jv, dJ, dV, gap = r
                cases.append(dict(
                    case_id=f"{judge}:{stem}:{v}", track=track, judge=judge,
                    dataset=ds, stem=stem, variant=v,
                    judge_clean=round(jc, 4), judge_variant=round(jv, 4),
                    dJ_norm=round(dJ, 4),
                    dV_norm=None if dV is None else round(dV, 4),
                    gap=round(gap, 4), human_pc=None, human_sa=None,
                    source_key=_resolve_source(ds, stem)))
            else:
                _, resid, ds, stem, jc, lab = r
                cases.append(dict(
                    case_id=f"{judge}:{stem}:clean", track=track, judge=judge,
                    dataset=ds, stem=stem, variant="clean",
                    judge_clean=round(jc, 4), judge_variant=None,
                    dJ_norm=None, dV_norm=None,
                    gap=round(resid, 4),
                    human_pc=lab[0], human_sa=lab[1],
                    source_key=_resolve_source(ds, stem)))

    # de-dup by case_id, then present in a fixed but non-informative order
    uniq = {c["case_id"]: c for c in cases}
    cases = sorted(uniq.values(), key=lambda c: _u01(seed, c["case_id"]))
    for i, c in enumerate(cases):
        c["order"] = i

    doc = {"version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seed": seed, "n_cases": len(cases),
           "params": dict(n_superficial=n_superficial, n_temporal=n_temporal,
                          n_clean_over=n_clean_over, n_clean_under=n_clean_under),
           "failure_modes": FM_KEYS, "cases": cases}

    by = Counter((c["judge"], c["track"]) for c in cases)
    print(f"selected {len(cases)} cases:")
    for (j, t), n in sorted(by.items()):
        print(f"  {j:15s} {t:20s} {n}")
    key = CASES_KEY.format(v=version)
    if push_to_s3:
        s3.put_object(Bucket=BUCKET, Key=key,
                      Body=json.dumps(doc, indent=2).encode())
        print(f"\nwrote s3://{BUCKET}/{key}")
    else:
        Path(f"cases_{version}.json").write_text(json.dumps(doc, indent=2))
        print(f"\nwrote ./cases_{version}.json (not pushed)")
    return doc


def load_cases(version="v1"):
    doc = _get(CASES_KEY.format(v=version))
    if not doc:
        raise RuntimeError(f"no case set at s3://{BUCKET}/{CASES_KEY.format(v=version)}"
                           f" -- run select() first")
    return doc


# ======================================================================
# media + rationale
# ======================================================================

_MIME = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}


def _stem_of(key):
    return os.path.splitext(os.path.basename(key.split("?")[0]))[0]


def _source_map(dataset):
    """{stem -> source object key} from datasets/<prefix>/. The clean clip IS
    the source object; attacks/ holds only the rendered variants."""
    if dataset not in _SRC_CACHE:
        m = {}
        for k in _list(DATASET_PREFIXES[dataset]):
            if k.lower().endswith(VIDEO_SUFFIXES) and "/_metadata/" not in k:
                m[_stem_of(k)] = k
        _SRC_CACHE[dataset] = m
    return _SRC_CACHE[dataset]


def _resolve_source(dataset, stem):
    m = _source_map(dataset)
    return (m.get(stem) or m.get(stem.removesuffix("_result"))
            or m.get(stem + "_result"))


def _clean_key(case):
    """clean video: the stored source_key, else look it up, else the (rare)
    attacks/.../clean.mp4."""
    return (case.get("source_key") or _resolve_source(case["dataset"], case["stem"])
            or _clip_key(case["dataset"], case["stem"], "clean"))


def _clip_key(dataset, stem, variant):
    return f"attacks/{DS_TO_ATTACKDIR[dataset]}/{stem}/{variant}.mp4"


def _alt_keys(key):
    """Same object under a toggled `_result` stem segment -- attacks/ dir names
    and pass-1 clip stems have historically differed on that suffix."""
    yield key
    parts = key.split("/")
    for i, seg in enumerate(parts):
        if seg.endswith("_result"):
            alt = parts[:]; alt[i] = seg[:-7]; yield "/".join(alt)
        elif i and parts[i - 1] in ("test", "implausibench_real",
                                    "implausibench_implausible") and "." not in seg:
            alt = parts[:]; alt[i] = seg + "_result"; yield "/".join(alt)


def _download(key):
    if key is None:
        return None
    dst = TMP / key.replace("/", "__")
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    for k in _alt_keys(key):
        try:
            body = s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()
        except Exception:
            continue
        dst.write_bytes(body)
        return dst
    return None


def _video_html(path, label, width=360):
    if path is None:
        return f"<div style='color:#c00'>[{label}: video not found]</div>"
    import base64
    mime = _MIME.get(path.suffix.lower(), "video/mp4")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f"<div style='display:inline-block;margin:4px;text-align:center;"
            f"font:12px sans-serif'>{label}<br>"
            f"<video width='{width}' controls autoplay loop muted playsinline>"
            f"<source src='data:{mime};base64,{b64}' type='{mime}'></video></div>")


def _phyjudge_rationale(dataset, stem, variant):
    """Concatenate PhyJudge's Pass-2 physics-law rationale text, if it exists."""
    rec = _get(f"{PASS2}/phyjudge_9b/{dataset}/{stem}.json")
    if not rec:
        return None
    calls = rec.get("runs", {}).get(variant, {}).get("calls", {})
    bits = []
    for cid, c in calls.items():
        raw = (c.get("raw") or "").strip()
        if raw:
            bits.append(f"[{cid}] {raw}")
    return "\n\n".join(bits) if bits else None


# ======================================================================
# coding UI
# ======================================================================

def _codes_path(version, coder):
    return TMP / f"codes_{version}__{coder}.jsonl"


def _codes_key(version, coder):
    return CODES_KEY.format(v=version, coder=coder)


def load_codes(version="v1", coder=None):
    if coder:
        keys = [_codes_key(version, coder)] if _exists(_codes_key(version, coder)) else []
    else:
        keys = [k for k in _list(f"case_codes/{version}/") if k.endswith(".jsonl")]
    out = []
    for k in keys:
        try:
            for line in _get_text(k).splitlines():
                if line.strip():
                    out.append(json.loads(line))
        except Exception:
            pass
    return out


_UPLOAD = ThreadPoolExecutor(max_workers=1)


def _flush(version, coder):
    path = _codes_path(version, coder)
    body = path.read_bytes()
    _UPLOAD.submit(lambda: s3.put_object(Bucket=BUCKET,
                                         Key=_codes_key(version, coder), Body=body))


def code(coder, version="v1"):
    """The rater UI. Resumable per case; a submit writes locally (sync) then
    queues an S3 PUT."""
    import ipywidgets as W
    from IPython.display import display, clear_output

    doc = load_cases(version)
    cases = sorted(doc["cases"], key=lambda c: c["order"])
    TMP.mkdir(parents=True, exist_ok=True)
    path = _codes_path(version, coder)
    if not path.exists():
        for r in load_codes(version, coder):        # restore from S3
            path.open("a").write(json.dumps(r) + "\n")
    done = {r["case_id"]: r for r in load_codes(version, coder)}

    state = {"i": next((k for k, c in enumerate(cases)
                        if c["case_id"] not in done), len(cases))}

    head = W.HTML()
    vids = W.HTML()
    ctx = W.HTML()
    rat = W.Accordion(children=[W.HTML()])
    rat.set_title(0, "PhyJudge Pass-2 rationale (click)")
    rat.selected_index = None
    mode = W.RadioButtons(options=[(d, k) for k, d in FAILURE_MODES],
                          description="", layout=W.Layout(width="95%"))
    note = W.Textarea(placeholder="evidence / reasoning (optional)",
                      layout=W.Layout(width="95%", height="60px"))
    back = W.Button(description="< Back")
    subm = W.Button(description="Submit >", button_style="primary")
    msg = W.HTML()
    box = W.VBox([head, vids, ctx, rat, W.HTML("<b>Failure mode</b>"), mode,
                  note, W.HBox([back, subm]), msg])
    display(box)

    def render():
        i = state["i"]
        if i >= len(cases):
            head.value = "<h3>All cases coded.</h3>"
            vids.value = ctx.value = ""
            rat.children[0].value = ""
            mode.disabled = note.disabled = subm.disabled = True
            msg.value = (f"{len(cases)} cases -> "
                         f"s3://{BUCKET}/{_codes_key(version, coder)}")
            return
        c = cases[i]
        head.value = (f"<h3>Case {i+1}/{len(cases)} &nbsp; "
                      f"<code>{c['case_id']}</code></h3>"
                      f"<div style='font:13px sans-serif'>judge <b>{c['judge']}</b>"
                      f" &nbsp; track <b>{c['track']}</b></div>")
        clean_p = _download(_clean_key(c))
        html = _video_html(clean_p, "clean")
        if c["variant"] != "clean":
            var_p = _download(_clip_key(c["dataset"], c["stem"], c["variant"]))
            html += _video_html(var_p, c["variant"])
        vids.value = html
        lines = [f"judge score on clean: <b>{c['judge_clean']}</b>"]
        if c["variant"] != "clean":
            lines.append(f"judge score on {c['variant']}: <b>{c['judge_variant']}</b>"
                         f" &nbsp; ΔJ(norm) <b>{c['dJ_norm']}</b>"
                         + (f" &nbsp; ΔV(ref) {c['dV_norm']} &nbsp; "
                            f"gap ΔJ−ΔV <b>{c['gap']}</b>" if c['dV_norm'] is not None
                            else f" &nbsp; gap <b>{c['gap']}</b>"))
        else:
            lines.append(f"human PC <b>{c['human_pc']}</b> &nbsp; SA {c['human_sa']}"
                         f" &nbsp; residual judge−human(norm) <b>{c['gap']}</b>")
        ctx.value = "<div style='font:13px sans-serif;line-height:1.5'>" \
                    + "<br>".join(lines) + "</div>"
        r = (_phyjudge_rationale(c["dataset"], c["stem"], c["variant"])
             if c["judge"] == "phyjudge_9b" else None)
        rat.children[0].value = (f"<pre style='white-space:pre-wrap;font:12px "
                                 f"monospace'>{r}</pre>" if r else
                                 "<i>no Pass-2 rationale for this case</i>")
        prev = done.get(c["case_id"])
        mode.value = prev["failure_mode"] if prev else None
        note.value = prev.get("note", "") if prev else ""
        mode.disabled = note.disabled = subm.disabled = False
        msg.value = "<i>revising a coded case</i>" if prev else ""

    def on_submit(_):
        i = state["i"]
        c = cases[i]
        if mode.value is None:
            msg.value = "<span style='color:#c00'>pick a failure mode</span>"
            return
        rec = dict(coder=coder, version=version, case_id=c["case_id"],
                   judge=c["judge"], track=c["track"], dataset=c["dataset"],
                   stem=c["stem"], variant=c["variant"],
                   failure_mode=mode.value, note=note.value.strip(),
                   ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        # rewrite the file so a revision replaces the old line
        done[c["case_id"]] = rec
        with path.open("w") as fh:
            for cc in cases:
                if cc["case_id"] in done:
                    fh.write(json.dumps(done[cc["case_id"]]) + "\n")
        _flush(version, coder)
        state["i"] = i + 1
        render()

    def on_back(_):
        state["i"] = max(0, state["i"] - 1)
        render()

    subm.on_click(on_submit)
    back.on_click(on_back)
    render()


# ======================================================================
# report + adjudication
# ======================================================================

def cohen_kappa(a, b, levels=None):
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    levels = levels or sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(levels)}
    n = len(a)
    obs = sum(x == y for x, y in zip(a, b)) / n
    pa = Counter(a)
    pb = Counter(b)
    exp = sum((pa[v] / n) * (pb[v] / n) for v in levels)
    if abs(1 - exp) < 1e-12:
        return float("nan")
    return (obs - exp) / (1 - exp)


def _pairwise_kappa(by_coder, coders, cases):
    """Mean pairwise Cohen's kappa over every coder pair, on cases all pairs
    share. Returns (mean_kappa, [(a, b, k, n), ...])."""
    rows = []
    for a, b in itertools.combinations(coders, 2):
        shared = [cid for cid in cases if cid in by_coder[a] and cid in by_coder[b]]
        if not shared:
            continue
        la = [by_coder[a][cid]["failure_mode"] for cid in shared]
        lb = [by_coder[b][cid]["failure_mode"] for cid in shared]
        rows.append((a, b, cohen_kappa(la, lb, levels=FM_KEYS), len(shared)))
    ks = [k for _a, _b, k, _n in rows if k == k]
    return (sum(ks) / len(ks) if ks else float("nan")), rows


def report(version="v1"):
    doc = load_cases(version)
    cases = {c["case_id"]: c for c in doc["cases"]}
    recs = load_codes(version)
    if not recs:
        print(f"no codes under s3://{BUCKET}/case_codes/{version}/")
        return {}
    by_coder = defaultdict(dict)
    for r in recs:
        if r.get("coder", "").startswith("adj") or r.get("adjudicated"):
            continue
        by_coder[r["coder"]][r["case_id"]] = r
    coders = sorted(by_coder)
    print(f"== {len(cases)} cases, coders: {', '.join(coders)} ==")
    for c in coders:
        miss = [cid for cid in cases if cid not in by_coder[c]]
        print(f"  {c}: {len(by_coder[c])}/{len(cases)} coded"
              + (f"  (missing {len(miss)})" if miss else ""))

    if len(coders) < 2:
        print("\nneed >=2 coders for kappa; distribution so far:")
        _dist(recs, cases)
        return {"coders": coders}

    mean_k, pairs = _pairwise_kappa(by_coder, coders, cases)
    print(f"\n== inter-coder agreement ({len(coders)} coders) ==")
    for a, b, k, n in pairs:
        print(f"  {a} vs {b}: kappa {k:+.3f}  (n={n})")
    print(f"  mean pairwise Cohen's kappa = {mean_k:+.3f}"
          + ("  (n small -- read with the majority table)" if len(cases) < 30 else ""))

    # per-case majority
    need = len(coders) // 2 + 1
    unan = maj = nomaj = 0
    splits = []
    for cid in cases:
        labs = [by_coder[c][cid]["failure_mode"] for c in coders if cid in by_coder[c]]
        if len(labs) < 2:
            continue
        cnt = Counter(labs)
        top_n = cnt.most_common(1)[0][1]
        if top_n == len(labs):
            unan += 1
        elif top_n * 2 > len(labs):
            maj += 1
        else:
            nomaj += 1
            splits.append((cid, labs))
    print(f"\n== per-case consensus ==")
    print(f"  unanimous {unan}   plurality/majority {maj}   "
          f"exact tie (auto-broken by mode priority) {nomaj}")
    for cid, labs in splits:
        print(f"  TIE {cid}: {' / '.join(labs)}")
        for c in coders:
            if cid in by_coder[c] and by_coder[c][cid].get("note"):
                print(f"     {c}: {by_coder[c][cid]['note'][:80]}")

    print("\n== failure-mode distribution (final label: adjudicated > plurality) ==")
    fin = final_labels(version)
    _dist([dict(cases[cid], failure_mode=fin[cid]["failure_mode"])
           for cid in cases if cid in fin], cases, from_final=True)
    return {"coders": coders, "mean_kappa": mean_k, "pairwise": pairs,
            "unanimous": unan, "majority": maj, "no_majority": nomaj,
            "splits": [cid for cid, _ in splits]}


def _dist(recs, cases, from_final=False):
    by = defaultdict(Counter)
    fam = defaultdict(Counter)
    for r in recs:
        fm = r.get("failure_mode")
        if fm is None:
            continue
        by[r["judge"]][fm] += 1
        c = cases.get(r["case_id"], {})
        key = ("temporal" if c.get("variant") in TEMPORAL else
               "superficial" if c.get("variant") in SUPERFICIAL else "clean")
        fam[key][fm] += 1
    for j, cnt in sorted(by.items()):
        print(f"  {j:15s} " + "  ".join(f"{m}:{cnt[m]}" for m in FM_KEYS if cnt[m]))
    print("  ---")
    for key, cnt in sorted(fam.items()):
        print(f"  {key:15s} " + "  ".join(f"{m}:{cnt[m]}" for m in FM_KEYS if cnt[m]))


def final_labels(version="v1"):
    """Per case: adjudicated label if one exists, else the PLURALITY label
    (the mode of the coders' picks). An exact tie (e.g. 1-1-1) is broken
    deterministically by FM_KEYS order; `source` records which case it was."""
    recs = load_codes(version)
    by_coder = defaultdict(dict)
    adj = {}
    for r in recs:
        if r.get("coder", "").startswith("adj") or r.get("adjudicated"):
            adj[r["case_id"]] = r
        else:
            by_coder[r["coder"]][r["case_id"]] = r
    coders = sorted(by_coder)
    out = {}
    for cid in {cid for d in by_coder.values() for cid in d}:
        if cid in adj:
            out[cid] = dict(adj[cid], source="adjudicated")
            continue
        labs = [by_coder[c][cid]["failure_mode"] for c in coders if cid in by_coder[c]]
        if not labs:
            continue
        cnt = Counter(labs)
        top_n = cnt.most_common(1)[0][1]
        tied = sorted((m for m in cnt if cnt[m] == top_n),
                      key=lambda m: FM_KEYS.index(m) if m in FM_KEYS else 99)
        pick = tied[0]
        src = ("unanimous" if top_n == len(labs)
               else "plurality" if len(tied) == 1 else "tie_broken")
        out[cid] = {"case_id": cid, "failure_mode": pick, "source": src,
                    "n_agree": top_n, "n_coders": len(labs)}
    return out


def adjudicate(version="v1", adjudicator="adj"):
    """Third pass over disagreements only."""
    import ipywidgets as W
    from IPython.display import display

    doc = load_cases(version)
    cases = {c["case_id"]: c for c in doc["cases"]}
    recs = load_codes(version)
    by_coder = defaultdict(dict)
    for r in recs:
        if not r.get("coder", "").startswith("adj"):
            by_coder[r["coder"]][r["case_id"]] = r
    coders = sorted(by_coder)
    if len(coders) < 2:
        print("need 2 coders before adjudication")
        return
    ca, cb = coders[:2]
    todo = [cid for cid in cases
            if cid in by_coder[ca] and cid in by_coder[cb]
            and by_coder[ca][cid]["failure_mode"] != by_coder[cb][cid]["failure_mode"]]
    already = {r["case_id"] for r in load_codes(version)
              if r.get("coder", "").startswith("adj")}
    todo = [c for c in todo if c not in already]
    if not todo:
        print("no un-adjudicated disagreements")
        return
    TMP.mkdir(parents=True, exist_ok=True)
    path = _codes_path(version, adjudicator)
    st = {"i": 0}
    head, vids, ctx = W.HTML(), W.HTML(), W.HTML()
    mode = W.RadioButtons(options=[(d, k) for k, d in FAILURE_MODES])
    note = W.Textarea(layout=W.Layout(width="95%", height="60px"))
    nxt = W.Button(description="Save >", button_style="primary")
    msg = W.HTML()
    display(W.VBox([head, vids, ctx, mode, note, nxt, msg]))

    def render():
        if st["i"] >= len(todo):
            head.value = "<h3>Adjudication done.</h3>"
            nxt.disabled = mode.disabled = True
            return
        cid = todo[st["i"]]
        c = cases[cid]
        head.value = (f"<h3>{st['i']+1}/{len(todo)} &nbsp; <code>{cid}</code></h3>"
                      f"{ca}: <b>{by_coder[ca][cid]['failure_mode']}</b> &nbsp; "
                      f"{cb}: <b>{by_coder[cb][cid]['failure_mode']}</b>")
        cp = _download(_clean_key(c))
        h = _video_html(cp, "clean")
        if c["variant"] != "clean":
            h += _video_html(_download(_clip_key(c["dataset"], c["stem"], c["variant"])),
                             c["variant"])
        vids.value = h
        ctx.value = (f"<pre>{ca} note: {by_coder[ca][cid].get('note','')}\n"
                     f"{cb} note: {by_coder[cb][cid].get('note','')}</pre>")
        mode.value = None
        note.value = ""

    def on_next(_):
        if mode.value is None:
            msg.value = "<span style='color:#c00'>pick one</span>"
            return
        cid = todo[st["i"]]
        rec = dict(coder=adjudicator, adjudicated=True, version=version,
                   case_id=cid, judge=cases[cid]["judge"],
                   track=cases[cid]["track"], dataset=cases[cid]["dataset"],
                   stem=cases[cid]["stem"], variant=cases[cid]["variant"],
                   failure_mode=mode.value, note=note.value.strip(),
                   ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        _flush(version, adjudicator)
        st["i"] += 1
        msg.value = ""
        render()

    nxt.on_click(on_next)
    render()


# ======================================================================
# selftest
# ======================================================================

def selftest():
    ok = True

    def c(cond, label):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {label}")

    c(abs(cohen_kappa("aabb", "aabb") - 1.0) < 1e-12, "kappa perfect = 1")
    c(cohen_kappa("abab", "baba") < 0.0, "kappa below chance < 0")
    c(np.isnan(cohen_kappa([], [])), "kappa empty = nan")
    c(np.isnan(cohen_kappa(["x", "x"], ["x", "x"])), "kappa all-one-class = nan")
    # obs 3/4, exp 0.3125 -> (0.75-0.3125)/(1-0.3125) = 0.6364
    c(abs(cohen_kappa(["a", "a", "b", "c"], ["a", "b", "b", "c"],
                      levels=["a", "b", "c"]) - 0.6364) < 0.01, "kappa partial")

    # deterministic ordering + de-dup
    u1 = _u01("s", "abc")
    c(0.0 <= u1 < 1.0 and u1 == _u01("s", "abc"), "hash in [0,1) and stable")
    c(_u01("s", "abc") != _u01("s", "abd"), "hash separates keys")

    c(_clip_key("test", "st", "shuffle") == "attacks/test/st/shuffle.mp4",
      "clip key layout")
    c(_norm("videophy2_auto", 3.0) == 0.5 and _norm("vila_ewm", 0.5) == 0.5,
      "score normalisation endpoints")

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
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--n-superficial", type=int, default=2)
    ap.add_argument("--n-temporal", type=int, default=2)
    ap.add_argument("--n-clean-over", type=int, default=1)
    ap.add_argument("--n-clean-under", type=int, default=1)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.select:
        select(version=a.version, n_superficial=a.n_superficial,
               n_temporal=a.n_temporal, n_clean_over=a.n_clean_over,
               n_clean_under=a.n_clean_under, push_to_s3=not a.no_push)
    if a.report:
        report(version=a.version)
    if not (a.select or a.report):
        ap.print_help()
