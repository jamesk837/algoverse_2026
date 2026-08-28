"""Ablation 9.5(3) -- the attack-aware temporal upper-bound probe.

Diagnostic ONLY. Never feeds a headline gameability number. It answers one
question the doc poses in Step 9.5 / Ablation (3):

    Does the frozen V-JEPA 2.1 representation contain *recoverable* temporal
    information that the clean-PC-supervised reference probe simply failed to
    extract?

The locked reference (`proj_mean`) is order-blind by construction -- it means
the 32 moments away before the head sees them -- so its response to
shuffle/reverse/freeze is a structural null (see locked-probe-is-order-blind).
That is a property of the readout, not proof that the *representation* is
temporally empty. This probe puts an upper bound on the latter: give a small
readout WITH temporal capacity EXPLICIT `clean > human-confirmed-temporal`
supervision -- the exact signal the reference probe never gets -- and see how
well it can then separate clean from temporally-corrupted clips on held-out
base clips.

  - if it separates well (AUC well above 0.5 and above the reference probe's
    AUC on the same clips): V-JEPA *does* carry the temporal signal linearly;
    clean-PC training just does not surface it.
  - if it cannot separate even with direct supervision: the signal is not
    linearly recoverable from these representations at this pooling.

Supervision is the blinded pairwise human judgment from the 60-clip study
(Step 11): a (clip, temporal-variant) pair is "confirmed" when the rater
picked the CLEAN clip as more physically plausible. Eligibility is defined by
humans, never by V-JEPA or the judges -- that is the doc's rule.

n is small (~60 base clips x 3 temporal variants, minus unconfirmed), so this
is a wide-error-bar diagnostic. Grouped k-fold over BASE CLIPS keeps a clip's
clean and attacked rows in the same fold. Everything is reported with a
clip-resampled bootstrap CI.

    python train_temporal_upperbound.py --selftest
    python train_temporal_upperbound.py                 # full run, writes S3
    python train_temporal_upperbound.py --no-push
    In Colab (after train_probe.py, and lock_probe.py if you want the
    reference-probe baseline):
        import train_temporal_upperbound as T
        T.run()

Writes probes/temporal_upperbound.json -- per-variant and pooled AUC, CIs,
the reference-probe and untrained-distance baselines, and the fold config --
which results_report.py reads for the Step 9.5(3) section.
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ adopt

# Bound by _adopt(): the names this file borrows from train_probe. Same
# module-or-__main__ dance lock_probe.py uses, and it fails LOUDLY rather than
# leaving a name unbound to surface 200 lines later.
_FROM_TRAIN_PROBE = ("BUCKET", "PROBE_PREFIX", "PACK_PREFIXES", "PACK_KIND",
                     "N_TEMPORAL", "TEMPORAL_VARIANTS", "load_pack",
                     "group_by_stem", "_boot_ci", "_ensure_s3")

BUCKET = PROBE_PREFIX = PACK_PREFIXES = PACK_KIND = None
N_TEMPORAL = None
TEMPORAL_VARIANTS = ("shuffle", "reverse", "freeze")
load_pack = group_by_stem = _boot_ci = _ensure_s3 = None

ANNOT_PREFIX = "annotations"
TASK_KEY_FMT = "annotation/tasks_{version}.json"
# datasets the 60-clip task set can draw from -> their consolidated t32 packs
BENCHMARK_DATASETS = ("videophy2_test", "implausibench_real",
                      "implausibench_implausible")
# annotate.py's dataset names -> pack names
DS_ALIAS = {"test": "videophy2_test", "videophy2_test": "videophy2_test",
            "implausibench_real": "implausibench_real",
            "implausibench_implausible": "implausibench_implausible"}

OUT_NAME = "temporal_upperbound"


def _adopt(verbose=True):
    """Bind train_probe's names, from the imported module or from __main__."""
    g = globals()
    src = None
    try:
        import train_probe as _tp
        src = _tp
        if verbose:
            print("adopting train_probe (imported module)")
    except Exception as exc:                      # noqa: BLE001
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and hasattr(main_mod, "load_pack"):
            src = main_mod
            if verbose:
                print("adopting train_probe names from __main__ (paste mode)")
        else:
            raise ImportError(
                "train_temporal_upperbound needs train_probe's helpers. On "
                "EC2 `import train_probe` must work (it imports boto3/torch); "
                "in Colab paste train_probe.py into a cell FIRST. Underlying "
                f"error: {exc!r}") from exc
    missing = [n for n in _FROM_TRAIN_PROBE if not hasattr(src, n)]
    if missing:
        raise ImportError(f"train_probe is missing {missing} -- version skew")
    for n in _FROM_TRAIN_PROBE:
        g[n] = getattr(src, n)


# ------------------------------------------------------------------ S3

def _get_json(key):
    body = _ensure_s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return json.loads(body)


def _list(prefix):
    keys, p = [], _ensure_s3().get_paginator("list_objects_v2")
    for page in p.paginate(Bucket=BUCKET, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys


# ------------------------------------------------------------------ data

def _stem_variants(stem_map, want=("clean",) + tuple(TEMPORAL_VARIANTS)):
    """Restrict a group_by_stem map to stems carrying clean + >=1 temporal."""
    out = {}
    for stem, d in stem_map.items():
        if "clean" not in d:
            continue
        temp = [v for v in TEMPORAL_VARIANTS if v in d]
        if temp:
            out[stem] = {v: d[v] for v in want if v in d}
    return out


def load_features(datasets=BENCHMARK_DATASETS):
    """-> {stem: {variant: (32,1024) float32}} over the benchmark corpora.

    Pooling is deliberately NOT applied here -- the whole point is to keep the
    time axis so a probe with temporal capacity can use it.
    """
    feats = {}
    for ds in datasets:
        try:
            pack = load_pack(ds)
        except Exception as exc:                  # noqa: BLE001
            print(f"  {ds}: no t32 pack ({type(exc).__name__}); skipped")
            continue
        X = pack["X"].astype(np.float32)
        sm = _stem_variants(group_by_stem(pack))
        for stem, d in sm.items():
            feats[stem] = {v: X[i] for v, i in d.items()}
        print(f"  {ds}: {len(sm)} stems with clean + a temporal variant")
    return feats


def _match(stem, feats):
    """annotate `clip` -> a key in `feats`, tolerating a `_result` suffix."""
    if stem in feats:
        return stem
    alt = stem[:-7] if stem.endswith("_result") else stem + "_result"
    return alt if alt in feats else None


def human_confirmed(version="v1"):
    """-> {(stem, variant): {"pref_clean": int, "n": int, "margin": float}}

    A pair is counted when a blinded rater expressed a preference (not tie /
    not skipped) on a temporal variant. `pref_clean` is how many raters picked
    the clean clip; `margin` is mean (pc_clean - pc_variant) where both were
    rated. A pair is "confirmed" downstream when pref_clean / n > 0.5.
    """
    keys = _list(f"{ANNOT_PREFIX}/{version}/")
    keys = [k for k in keys if k.endswith(".jsonl")]
    if not keys:
        raise RuntimeError(f"no annotation records under {ANNOT_PREFIX}/{version}/")
    acc = {}
    raters = set()
    for k in keys:
        body = _ensure_s3().get_object(Bucket=BUCKET, Key=k)["Body"].read()
        for line in body.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("skipped") or r.get("variant") not in TEMPORAL_VARIANTS:
                continue
            pref = r.get("preference")
            if pref not in ("clean", "variant", "tie"):
                continue
            raters.add(r.get("rater"))
            key = (r["clip"], r["variant"])
            a = acc.setdefault(key, {"pref_clean": 0, "n": 0, "margins": []})
            a["n"] += 1
            if pref == "clean":
                a["pref_clean"] += 1
            pc_c, pc_v = r.get("pc_clean"), r.get("pc_variant")
            if isinstance(pc_c, (int, float)) and isinstance(pc_v, (int, float)):
                a["margins"].append(float(pc_c) - float(pc_v))
    out = {}
    for key, a in acc.items():
        out[key] = {"pref_clean": a["pref_clean"], "n": a["n"],
                    "margin": float(np.mean(a["margins"])) if a["margins"]
                    else float("nan")}
    print(f"  {len(out)} (clip, temporal-variant) pairs rated by "
          f"{len(raters)} rater(s): {sorted(r for r in raters if r)}")
    return out


def assemble(feats, confirmed, require_confirmed=True):
    """-> list of pairs: dict(stem, variant, clean(32,1024), attk(32,1024),
    confirmed:bool, margin:float). One row per (annotated clip, temporal
    variant) that also has V-JEPA features.
    """
    rows, no_feat, no_anno = [], 0, 0
    for (stem, variant), c in confirmed.items():
        fk = _match(stem, feats)
        if fk is None or variant not in feats[fk] or "clean" not in feats[fk]:
            no_feat += 1
            continue
        is_conf = c["n"] > 0 and (c["pref_clean"] / c["n"]) > 0.5
        rows.append(dict(stem=fk, variant=variant,
                         clean=feats[fk]["clean"], attk=feats[fk][variant],
                         confirmed=is_conf, margin=c["margin"],
                         pref_clean=c["pref_clean"], n_rat=c["n"]))
    for stem in feats:
        if not any(_match(stem, feats) == r["stem"] for r in rows):
            no_anno += 1
    n_conf = sum(r["confirmed"] for r in rows)
    print(f"  assembled {len(rows)} annotated pairs with features "
          f"({n_conf} human-confirmed clean>temporal); "
          f"{no_feat} annotated pairs had no features")
    if require_confirmed and n_conf < 6:
        raise RuntimeError(
            f"only {n_conf} human-confirmed pairs -- too few to train even a "
            "diagnostic probe. Lower --min-confirmed to force it, but the "
            "result will be noise.")
    return rows


# ------------------------------------------------------------------ probe

def _make_probe(hidden, seed):
    """LayerNorm -> 1024->H -> first difference over the 32 moments -> per-unit
    MEAN ABSOLUTE difference over the 31 gaps -> H->1.

    The first difference cancels static appearance; taking |.| before pooling
    turns it into per-channel temporal-variation ENERGY, which is what
    shuffle (frame-to-frame jumps grow) and freeze (they collapse) move. A
    signed mean would telescope to (z_T - z_1)/(T-1) and see neither. This is
    train_probe's `diff_conv` intuition with the conv replaced by |.| -- n
    here is far too small to fit a Conv1d. Magnitude-based, so `reverse`
    (adjacent-frame similarity ~unchanged) is expected near chance; that is a
    finding, not a bug, and it is why the table is per variant."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class TemporalReadout(nn.Module):
        def __init__(self, d=1024, h=hidden):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.proj = nn.Linear(d, h)
            self.head = nn.Linear(h, 1)

        def forward(self, x):                      # x: (B, T, D)
            z = self.proj(self.norm(x))           # (B, T, H)
            dz = (z[:, 1:, :] - z[:, :-1, :]).abs()   # (B, T-1, H)
            return self.head(dz.mean(dim=1)).squeeze(-1)   # (B,)

    return TemporalReadout()


def _fit_fold(train_rows, hidden, epochs, lr, wd, margin, seed):
    import torch

    model = _make_probe(hidden, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    clean = torch.tensor(np.stack([r["clean"] for r in train_rows]))
    attk = torch.tensor(np.stack([r["attk"] for r in train_rows]))
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        fc, fa = model(clean), model(attk)
        # want f(clean) > f(attk) + margin
        loss = torch.relu(margin - (fc - fa)).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _score(model, rows):
    import torch
    with torch.no_grad():
        clean = torch.tensor(np.stack([r["clean"] for r in rows]))
        attk = torch.tensor(np.stack([r["attk"] for r in rows]))
        return (model(clean) - model(attk)).cpu().numpy()   # >0 == clean higher


# ------------------------------------------------------------------ metrics

def auc_gt_zero(scores):
    """P(score > 0) with ties at 0.5 -- 'clean ranked above attacked'.

    scores = f(clean) - f(attacked), one per pair. This is the paired form of
    AUC: each pair is its own clean/attacked contrast, so the statistic is the
    fraction of pairs the probe got the right way round.
    """
    s = np.asarray(scores, dtype=float)
    if not len(s):
        return float("nan")
    return float((s > 0).mean() + 0.5 * (s == 0).mean())


def _grouped_folds(rows, k, seed):
    """k folds over BASE CLIPS, so a clip's temporal variants never straddle
    the split."""
    stems = sorted({r["stem"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)
    k = min(k, len(stems))
    assign = {s: i % k for i, s in enumerate(stems)}
    return [[r for r in rows if assign[r["stem"]] != f] for f in range(k)], \
           [[r for r in rows if assign[r["stem"]] == f] for f in range(k)], k


def cv_auc(rows, k, hidden, epochs, lr, wd, margin, seed, eval_confirmed_only):
    """Grouped k-fold. -> (per-pair held-out scores dict keyed by (stem,variant),
    n_folds). Every pair is scored exactly once, by a model that never saw its
    clip."""
    tr_folds, te_folds, k = _grouped_folds(rows, k, seed)
    held = {}
    for tr, te in zip(tr_folds, te_folds):
        tr_use = [r for r in tr if r["confirmed"]]   # ALWAYS train on confirmed
        if len(tr_use) < 4 or not te:
            continue
        model = _fit_fold(tr_use, hidden, epochs, lr, wd, margin, seed)
        te_use = [r for r in te if r["confirmed"]] if eval_confirmed_only else te
        if not te_use:
            continue
        for r, sc in zip(te_use, _score(model, te_use)):
            held[(r["stem"], r["variant"])] = float(sc)
    return held, k


# ------------------------------------------------- reference-probe baseline

def reference_probe_auc(rows, name="probe_locked"):
    """The locked PC probe's own clean>temporal separation on the SAME pairs,
    the number this upper bound is compared against. Returns None if
    lock_probe / the checkpoint is unavailable."""
    try:
        import lock_probe as LP
    except Exception:                             # noqa: BLE001
        LP = sys.modules.get("__main__")
        if LP is None or not hasattr(LP, "load_locked"):
            return None
    try:
        model, calib, _ = LP.load_locked(name)
    except Exception as exc:                      # noqa: BLE001
        print(f"  (reference-probe baseline skipped: {type(exc).__name__}: {exc})")
        return None
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()

    def pc(batch):
        with torch.no_grad():
            x = torch.tensor(np.stack(batch)).to(dev)
            raw = LP._predict(model, x, dev) if hasattr(LP, "_predict") \
                else model(x).cpu().numpy()
        return LP.isotonic_apply(calib, np.asarray(raw))

    scores = pc([r["clean"] for r in rows]) - pc([r["attk"] for r in rows])
    per = {(r["stem"], r["variant"]): float(s) for r, s in zip(rows, scores)}
    return per


def distance_auc(rows):
    """Untrained floor: does raw feature distance ||pool(clean)-pool(attk)||
    even tell clean from attacked apart? It has no sign, so we score whether
    the attacked clip's per-moment trajectory moved -- report |.| only, as an
    AUC of attacked-vs-a-null of zero is meaningless. Kept as a magnitude."""
    mags = []
    for r in rows:
        d = r["clean"].mean(0) - r["attk"].mean(0)
        mags.append(float(np.linalg.norm(d)))
    return float(np.mean(mags)), float(np.std(mags))


# ------------------------------------------------------------------ run

def run(datasets=BENCHMARK_DATASETS, version="v1", k=5, hidden=32,
        epochs=300, lr=1e-3, wd=1e-3, margin=0.5, seed=0,
        min_confirmed=6, name=OUT_NAME, push_to_s3=True):
    _adopt()
    t0 = time.time()

    print("task set / selection:")
    try:
        task = _get_json(TASK_KEY_FMT.format(version=version))
        print(f"  quotas   {task.get('quotas')}")
        print(f"  selection {json.dumps(task.get('selection'), default=str)[:300]}")
        print(f"  raters   {task.get('raters')}  overlap {task.get('overlap')}")
    except Exception as exc:                      # noqa: BLE001
        print(f"  (could not read {TASK_KEY_FMT.format(version=version)}: {exc})")

    print("\nfeatures:")
    feats = load_features(datasets)
    if not feats:
        raise RuntimeError("no benchmark t32 packs found -- embed_vjepa the "
                           "benchmark corpora first (score_corpus.py needs "
                           "them too)")

    print("\nhuman labels:")
    conf = human_confirmed(version)
    rows = assemble(feats, conf, require_confirmed=(min_confirmed > 0))
    n_conf_total = sum(r["confirmed"] for r in rows)
    if n_conf_total < min_confirmed:
        raise RuntimeError(f"{n_conf_total} confirmed pairs < min_confirmed "
                           f"{min_confirmed}")

    dist_mean, dist_sd = distance_auc(rows)
    ref_per = reference_probe_auc([r for r in rows if r["confirmed"]], name="probe_locked")

    print("\ngrouped k-fold (train on confirmed pairs only, every fold):")
    per_variant, pooled_scores = {}, []
    held, kk = cv_auc(rows, k, hidden, epochs, lr, wd, margin, seed,
                      eval_confirmed_only=True)
    print(f"  {kk} folds, {len(held)} confirmed pairs scored held-out")

    def block(pairs, label):
        s = np.array([held[p] for p in pairs if p in held])
        if len(s) < 3:
            print(f"  {label:<26} n={len(s):3d}   (too few)")
            return None
        a = auc_gt_zero(s)
        lo, hi = _boot_ci((s > 0).astype(float), n_boot=4000, seed=seed)
        ref = None
        if ref_per is not None:
            rs = np.array([ref_per[p] for p in pairs if p in ref_per])
            ref = auc_gt_zero(rs) if len(rs) >= 3 else None
        refstr = f"   ref-probe {ref:.3f}" if ref is not None else ""
        print(f"  {label:<26} n={len(s):3d}   upper-bound AUC {a:.3f} "
              f"[{lo:.3f}, {hi:.3f}]{refstr}")
        return dict(n=int(len(s)), auc=a, ci_lo=lo, ci_hi=hi,
                    ref_probe_auc=ref)

    all_pairs = list(held)
    per_variant["pooled"] = block(all_pairs, "pooled (all temporal)")
    for v in TEMPORAL_VARIANTS:
        vp = [p for p in all_pairs if p[1] == v]
        per_variant[v] = block(vp, v)

    pooled = per_variant.get("pooled") or {}
    verdict = "INCONCLUSIVE"
    why = "too few held-out pairs"
    if pooled.get("n", 0) >= 10 and not np.isnan(pooled.get("auc", np.nan)):
        a, lo = pooled["auc"], pooled["ci_lo"]
        ref = pooled.get("ref_probe_auc")
        if lo > 0.5 and (ref is None or a - ref > 0.1):
            verdict = "RECOVERABLE"
            why = ("with explicit supervision a temporal readout separates "
                   "clean from attacked on held-out clips (CI above 0.5)"
                   + ("" if ref is None else
                      f" and beats the order-blind reference probe "
                      f"({a:.3f} vs {ref:.3f})")
                   + " -- V-JEPA carries temporal signal that clean-PC "
                   "training does not surface.")
        elif pooled["ci_hi"] <= 0.5:
            verdict = "NOT RECOVERABLE"
            why = ("even with direct clean>temporal supervision the readout "
                   "cannot separate them (CI at or below chance) -- the "
                   "signal is not linearly recoverable at this pooling.")
        else:
            verdict = "WEAK"
            why = (f"AUC {a:.3f} CI [{pooled['ci_lo']:.3f}, "
                   f"{pooled['ci_hi']:.3f}] straddles or barely clears "
                   "chance; treat as no strong evidence either way.")

    print(f"\nVERDICT: {verdict}\n  {why}")
    print(f"  untrained feature distance ||pool(clean)-pool(attk)||: "
          f"{dist_mean:.3f} +/- {dist_sd:.3f} (magnitude only, context)")

    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "diagnostic_only": True,
        "purpose": "Step 9.5(3) / ablation (3): upper bound on recoverable "
                   "temporal information in frozen V-JEPA 2.1 given explicit "
                   "clean>human-confirmed-temporal supervision. Never used for "
                   "headline gameability results.",
        "config": dict(datasets=list(datasets), annotation_version=version,
                       folds=kk, hidden=hidden, epochs=epochs, lr=lr,
                       weight_decay=wd, margin=margin, seed=seed,
                       readout="LayerNorm->1024xH->first-diff over 32 moments"
                               "->mean over 31->Hx1, margin ranking loss"),
        "n_pairs_total": len(rows),
        "n_confirmed": int(n_conf_total),
        "per_variant": per_variant,
        "verdict": verdict,
        "verdict_reason": why,
        "baselines": {"untrained_feature_distance_mean": dist_mean,
                      "untrained_feature_distance_sd": dist_sd,
                      "reference_probe": "per-variant ref_probe_auc above"},
        "seconds": round(time.time() - t0, 1),
    }
    Path(f"./{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> ./{name}.json")
    if push_to_s3:
        _ensure_s3().put_object(Bucket=BUCKET,
                                Key=f"{PROBE_PREFIX}/{name}.json",
                                Body=json.dumps(payload, indent=2).encode())
        print(f"uploaded -> s3://{BUCKET}/{PROBE_PREFIX}/{name}.json")
    return payload


# ------------------------------------------------------------------ selftest

def selftest():
    ok = True

    def check(cond, label):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {label}")

    check(abs(auc_gt_zero([1, 2, 3]) - 1.0) < 1e-12, "auc all-positive = 1")
    check(abs(auc_gt_zero([-1, -2, -3]) - 0.0) < 1e-12, "auc all-negative = 0")
    check(abs(auc_gt_zero([0, 0]) - 0.5) < 1e-12, "auc all-tied = 0.5")
    check(abs(auc_gt_zero([1, -1]) - 0.5) < 1e-12, "auc half = 0.5")

    # grouped folds never split a stem
    rows = [dict(stem=f"s{i}", variant=v, clean=None, attk=None,
                 confirmed=True, margin=0.0)
            for i in range(10) for v in TEMPORAL_VARIANTS]
    tr, te, k = _grouped_folds(rows, 5, 0)
    leak = any({r["stem"] for r in tr_f} & {r["stem"] for r in te_f}
               for tr_f, te_f in zip(tr, te))
    check(not leak, "grouped folds keep a stem on one side")
    check(k == 5, "k honoured when enough stems")
    _, _, k2 = _grouped_folds(rows[:6], 5, 0)
    check(k2 == 2, "k clamped to stem count (2 stems -> 2 folds)")

    # every pair scored at most once, by a fold that excludes its stem
    seen = [p for tr_f, te_f in zip(tr, te) for p in
            [(r["stem"], r["variant"]) for r in te_f]]
    check(len(seen) == len(set(seen)) == len(rows),
          "every pair is a held-out pair exactly once")

    # tiny end-to-end fit on separable synthetic data
    try:
        import torch                              # noqa: F401
        rng = np.random.default_rng(0)
        syn = []
        for i in range(24):
            base = rng.normal(size=(32, 1024)).astype(np.float32)
            clean = base.copy()
            attk = base.copy()
            attk[1:] = attk[:-1]                  # freeze-like: kill the diffs
            syn.append(dict(stem=f"c{i}", variant="freeze", clean=clean,
                            attk=attk, confirmed=True, margin=1.0))
        held, kk = cv_auc(syn, 4, 16, 150, 3e-3, 1e-4, 0.5, 0,
                          eval_confirmed_only=True)
        a = auc_gt_zero(list(held.values()))
        check(a > 0.8, f"learns a separable freeze signal (AUC {a:.2f})")
    except ImportError:
        print("  [skip] torch not present; end-to-end fit not exercised")

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
    ap.add_argument("--datasets", nargs="+", default=list(BENCHMARK_DATASETS))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-confirmed", type=int, default=6)
    ap.add_argument("--name", default=OUT_NAME)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run(datasets=a.datasets, version=a.version, k=a.folds, hidden=a.hidden,
        epochs=a.epochs, lr=a.lr, wd=a.weight_decay, margin=a.margin,
        seed=a.seed, min_confirmed=a.min_confirmed, name=a.name,
        push_to_s3=not a.no_push)
