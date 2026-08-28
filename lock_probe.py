"""Stage 5b -- select ONE reference probe on clean validation, then freeze it.

This is the doc's Step 9 selection protocol, and the whole point of the file is
that the choice is auditable:

  1. train mean_linear and proj_mean for 3 seeds each, everything else held
     identical (same packs, split, targets, loss, weighting, consistency,
     optimizer, early stopping);
  2. select on CLEAN VideoPhy-2 validation only -- macro MAE primary, Spearman
     secondary -- never on how a probe responds to the benchmark attacks;
  3. fit a monotonic calibration on the reserved clean `cal` split;
  4. freeze weights, calibration, encoder checkpoint, preprocessing and frame
     sampling into one manifest.

Colab:
    !wget -q -O lock_probe.py <raw url>
    import lock_probe                      # after train_probe.py's cell
    r = lock_probe.lock(name="probe_locked")
    lock_probe.selftest()

EC2:
    python lock_probe.py --name probe_locked
    python lock_probe.py --selftest
"""

import argparse
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

# On EC2 this imports from the sibling file. In Colab train_probe.py is pasted
# into an earlier cell, so these names are already global and the import fails
# harmlessly -- same pattern as eval_probe.py / diagnose_probe.py.
try:
    from train_probe import (BUCKET, PROBE_PREFIX, PACK_PREFIXES, PACK_KIND,
                             EMBED_DIM, N_TEMPORAL, N_THRESH, build_eval,
                             expected_pc, load_pack, make_model, train,
                             _ensure_s3, _rankdata, _spearman)
except ImportError:
    pass

# Duplicated rather than imported, deliberately: the import above is
# best-effort (it fails by design in Colab), and a default argument binds at
# `def` time, so anything used as one cannot depend on that import having
# worked. Same reasoning as eval_probe.HELD_OUT.
HELD_OUT = ("val", "cal")
CAL_SPLIT = "cal"
VAL_SPLIT = "val"
SELECT_ARCHS = ("mean_linear", "proj_mean")
SEEDS = (0, 1, 2)
LEVELS = (1, 2, 3, 4, 5)
PC_LO, PC_HI = 2, 4          # the doc's PC<=2 vs PC>=4 separation
SCALE_LO, SCALE_HI = 1.0, 5.0

# What embed_vjepa.py actually did, pinned here so the manifest is a complete
# description of the instrument rather than a pointer at code that can move.
FROZEN_ENCODER = {
    "hub_repo": "facebookresearch/vjepa2",
    "hub_model": "vjepa2_1_vit_large_384",
    "predictor": "discarded",
    "requires_grad": False,
}
FROZEN_PREPROCESSING = {
    "crop_size": 384,
    "color": "cv2 decodes BGR; converted to RGB before the ImageNet "
             "mean/std normalisation the preprocessor applies",
    "normalisation": "ImageNet mean/std (via vjepa2_preprocessor)",
    "autocast": "bfloat16 on cuda",
}
FROZEN_SAMPLING = {
    "num_frames": 64,
    "indices": "np.linspace over the clip's own decoded length (not "
               "CAP_PROP_FRAME_COUNT, which lies on re-encoded files)",
    "pooling": "18432 tokens -> (32, 576, 1024), mean over the 576 spatial "
               "positions only; temporal-major flattening, checked by "
               "embed_vjepa.verify_temporal_axis()",
    "n_temporal": 32,
    "embed_dim": 1024,
    "dtype": "float32 accumulation, stored float16",
}

try:
    from google.colab import userdata
    for _k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        try:
            os.environ[_k] = userdata.get(_k)
        except Exception:
            pass
except Exception:
    pass
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


# ------------------------------------------------ clean-only selection view
#
# Selection must never see an attack response. That is enforced structurally
# rather than by comment: select() is only ever handed records built by
# clean_only(), and _assert_clean() raises if an attack key ever reaches it.
# Getting this wrong is invisible in the output and fatal to the paper -- a
# reviewer's first question is whether the reference was chosen because it
# produced the desired attack result.

_CLEAN_KEYS = ("mae", "macro_mae", "rho", "rmse", "epoch")
_ATTACK_KEYS = ("variant_stats", "signed", "gaps", "mean_gap", "compression",
                "mean_temporal_signed", "heldout", "abs_ci", "signed_ci")


def _assert_clean(rec, where="selection"):
    bad = sorted(set(rec) & set(_ATTACK_KEYS))
    if bad:
        raise ValueError(
            f"attack-response keys reached {where}: {bad}. Probe selection is "
            "defined on clean validation performance only; passing these in "
            "would make the reference selectable on its attack response.")
    return rec


def clean_only(best, **extra):
    """Strip a train() record down to its clean validation numbers."""
    rec = {k: best[k] for k in _CLEAN_KEYS if k in best}
    rec.update(extra)
    return _assert_clean(rec, where="clean_only")


# ------------------------------------------------------------- statistics

def auc(pos, neg):
    """P(a random positive scores above a random negative), tie-corrected.

    Midranks, for the same reason _spearman uses them: predictions saturate
    onto near-integers, so ties are common and argsort-ranking invents
    separation that is not there.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if not len(pos) or not len(neg):
        return float("nan")
    r = _rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def separation(pred, y, lo=PC_LO, hi=PC_HI):
    """The doc's PC<=2 vs PC>=4 separation: gap in predicted PC, Cohen's d,
    and the tie-corrected AUC."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y)
    a, b = pred[y <= lo], pred[y >= hi]
    if not len(a) or not len(b):
        return {"gap": float("nan"), "d": float("nan"), "auc": float("nan"),
                "n_lo": int(len(a)), "n_hi": int(len(b))}
    va = a.var(ddof=1) if len(a) > 1 else 0.0
    vb = b.var(ddof=1) if len(b) > 1 else 0.0
    sd = np.sqrt((va + vb) / 2.0)
    return {"gap": float(b.mean() - a.mean()),
            "d": float((b.mean() - a.mean()) / sd) if sd > 0 else float("nan"),
            "auc": auc(b, a), "n_lo": int(len(a)), "n_hi": int(len(b))}


def clean_scores(pred, y):
    """MAE, macro MAE (the selector), Spearman, and the PC<=2/PC>=4 split."""
    pred = np.asarray(pred, dtype=np.float64)
    yf = np.asarray(y, dtype=np.float64)
    per_level = [np.abs(pred[yf == lv] - lv).mean()
                 for lv in LEVELS if (yf == lv).any()]
    return {"mae": float(np.abs(pred - yf).mean()),
            "macro_mae": float(np.mean(per_level)) if per_level else float("nan"),
            "rho": float(_spearman(pred, yf)),
            "sep": separation(pred, yf),
            "n": int(len(yf))}


def _mean_sd(xs):
    xs = [x for x in xs if x == x]        # drop NaN
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs, ddof=1) if len(xs) > 1 else 0.0)


# ------------------------------------------------- monotonic calibration
#
# The probe's raw E[PC] is compressed (2.7-4.0 against a 1-5 label range), so
# an uncalibrated ABSOLUTE score is not comparable with a human PC. Isotonic
# regression is the standard monotonic fix: it moves the values onto the label
# scale and, being monotonic, leaves every ranking exactly as it was --
# asserted below rather than assumed.
#
# Fit on `cal` ONLY. val already picked the epoch and the architecture, so
# calibrating on it would be fitting to a split the model was selected on.

def _pava(y, w):
    """Pool-adjacent-violators. -> the non-decreasing least-squares fit."""
    vals, wts, cnts = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi))
        wts.append(float(wi))
        cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            nw = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / nw)
            wts.append(nw)
            cnts.append(c1 + c2)
    out = []
    for v, c in zip(vals, cnts):
        out.extend([v] * c)
    return np.array(out, dtype=np.float64)


def isotonic_fit(x, y, lo=SCALE_LO, hi=SCALE_HI):
    """-> {"x": knots, "y": values}, a non-decreasing piecewise-linear map.

    Duplicate x are collapsed to one weighted knot first, so the knots are
    strictly increasing and np.interp is well defined.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(f"x and y differ in length: {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise ValueError("need at least 2 points to fit a calibration")

    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    ux, start = np.unique(xs, return_index=True)
    counts = np.diff(np.append(start, len(ys))).astype(np.float64)
    means = np.add.reduceat(ys, start) / counts

    fitted = np.clip(_pava(means, counts), lo, hi)
    return {"x": ux.tolist(), "y": fitted.tolist(), "lo": float(lo),
            "hi": float(hi), "n": int(len(x)),
            "method": "isotonic (PAVA), linear interpolation between knots, "
                      "clipped at the endpoints"}


def isotonic_apply(cal, x):
    """Apply a fitted calibration. Flat outside the fitted range, by design:
    extrapolating a monotone fit past its support invents resolution."""
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(cal["x"], dtype=np.float64)
    fp = np.asarray(cal["y"], dtype=np.float64)
    if len(xp) == 1:
        return np.full_like(x, fp[0])
    return np.clip(np.interp(x, xp, fp), cal["lo"], cal["hi"])


def check_calibration(cal, pred, y, tol=1e-9):
    """A calibration that inverts any pair is a bug, not a calibration.

    The guarantee to test is NO INVERSION -- sort by the raw prediction and the
    calibrated values must never go down. It is deliberately not "Spearman is
    1.0": isotonic pooling merges violators, so distinct raw values map onto
    one calibrated value and the rank correlation legitimately falls below 1.
    Requiring rho == 1 would demand a strictly increasing map, which no
    isotonic fit on real data is, and would fail every healthy calibration.
    `ties_created` is reported so the pooling is visible rather than silent.
    """
    pred = np.asarray(pred, dtype=np.float64)
    out = isotonic_apply(cal, pred)
    ordered = out[np.argsort(pred, kind="mergesort")]
    ok = bool((np.diff(ordered) >= -tol).all())
    yf = np.asarray(y, dtype=np.float64)
    return {"monotone": ok,
            "rank_rho": float(_spearman(_rankdata(pred), _rankdata(out))),
            "ties_created": int(len(np.unique(pred)) - len(np.unique(out))),
            "mae_before": float(np.abs(pred - yf).mean()),
            "mae_after": float(np.abs(out - yf).mean()),
            "span_before": float(np.ptp(pred)),
            "span_after": float(np.ptp(out))}


# ------------------------------------------------------------ the runs

def _predict(model, X, device, batch=1024):
    """E[PC] = 1 + sum_k P(PC>k). Local copy so this file does not import
    eval_probe -- in Colab that would pin the paste order three files deep."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            chunk = torch.as_tensor(X[i:i + batch], device=device)
            out.append(expected_pc(model(chunk)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _rebuild(result, device):
    model = make_model(result["cfg"]).to(device)
    model.load_state_dict(result["best"]["state"])
    model.eval()
    return model


def run_seeds(archs=SELECT_ARCHS, seeds=SEEDS, device=None, packs=None,
              verbose=True, **overrides):
    """One training per (arch, seed) on shared packs and one shared cfg.

    Returns {(arch, seed): train() result}. The cfg is printed once so
    "identical hyperparameters" is checkable rather than asserted -- same
    reasoning as train_probe.ablation().
    """
    import torch
    clash = sorted({"arch", "seed"} & set(overrides))
    if clash:
        raise ValueError(f"{clash} is what this sweep varies; pass archs= / "
                         "seeds= instead of overriding it per run")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if packs is None:
        print("loading packs ...")
        packs = {s: load_pack(s) for s in ("train",) + HELD_OUT}

    runs = {}
    for arch in archs:
        for seed in seeds:
            print(f"\n=== {arch}  seed {seed} ===")
            runs[(arch, seed)] = train(device=device, verbose=False,
                                       packs=packs, arch=arch, seed=seed,
                                       **overrides)
    if verbose and runs:
        cfg = next(iter(runs.values()))["cfg"]
        shared = {k: v for k, v in sorted(cfg.items())
                  if k not in ("arch", "seed", "n_temporal")}
        print("\nshared cfg (identical across every run above):")
        print("  " + json.dumps(shared, default=str))
    return runs, packs, device


def score_runs(runs, packs, device):
    """Clean val + clean cal numbers for every run. Attack stats are never
    read here -- clean_only() is what select() is allowed to see."""
    val_clean, val_y, _, _ = build_eval(packs[VAL_SPLIT])
    cal_clean, cal_y, _, _ = build_eval(packs[CAL_SPLIT])

    scored = {}
    for key, r in runs.items():
        model = _rebuild(r, device)
        v = clean_scores(_predict(model, val_clean, device), val_y)
        c = clean_scores(_predict(model, cal_clean, device), cal_y)
        scored[key] = {"val": clean_only(r["best"], **v), "cal": c,
                       "epochs": r["epochs_run"],
                       "early_stopped": r["early_stopped"],
                       "params": r["params"]}
    return scored, (val_clean, val_y), (cal_clean, cal_y)


def select(scored, archs=SELECT_ARCHS, seeds=SEEDS, pick="median"):
    """Pick the architecture on mean val macro MAE across seeds, then pick the
    seed that ships.

    The architecture is chosen on the ACROSS-SEED MEAN, not on the single best
    run: with 6 runs, taking the minimum would mostly be selecting seed noise,
    and the seeds exist to give error bars in the first place. Spearman breaks
    a tie. `pick="median"` then ships the median seed -- deliberately not the
    lucky one; `pick="best"` ships the best-on-val seed and is the opt-in.
    """
    for key, s in scored.items():
        _assert_clean(s["val"], where=f"select({key})")

    summary = {}
    for arch in archs:
        rows = [scored[(arch, sd)] for sd in seeds if (arch, sd) in scored]
        if not rows:
            continue
        macro_m, macro_sd = _mean_sd([r["val"]["macro_mae"] for r in rows])
        mae_m, mae_sd = _mean_sd([r["val"]["mae"] for r in rows])
        rho_m, rho_sd = _mean_sd([r["val"]["rho"] for r in rows])
        gap_m, gap_sd = _mean_sd([r["val"]["sep"]["gap"] for r in rows])
        auc_m, auc_sd = _mean_sd([r["val"]["sep"]["auc"] for r in rows])
        summary[arch] = {
            "n_seeds": len(rows),
            "macro_mae": macro_m, "macro_mae_sd": macro_sd,
            "mae": mae_m, "mae_sd": mae_sd,
            "rho": rho_m, "rho_sd": rho_sd,
            "sep_gap": gap_m, "sep_gap_sd": gap_sd,
            "sep_auc": auc_m, "sep_auc_sd": auc_sd,
            "params": rows[0]["params"],
        }

    if not summary:
        raise ValueError("no runs to select from")

    # primary macro MAE (lower better), secondary Spearman (higher better)
    winner = min(summary, key=lambda a: (summary[a]["macro_mae"],
                                         -summary[a]["rho"]))

    cand = sorted((sd for sd in seeds if (winner, sd) in scored),
                  key=lambda sd: scored[(winner, sd)]["val"]["macro_mae"])
    seed = cand[len(cand) // 2] if pick == "median" else cand[0]

    return {"arch": winner, "seed": int(seed), "pick": pick,
            "summary": summary,
            "criterion": "clean val macro MAE (primary), Spearman (secondary); "
                         "architecture chosen on the across-seed mean",
            "seed_order": [int(s) for s in cand]}


def print_selection(scored, sel, seeds=SEEDS):
    print("\n=== per-run clean validation ===")
    print(f"  {'arch':<13} {'seed':>5} {'macro MAE':>10} {'MAE':>9} "
          f"{'rho':>8} {'sep gap':>9} {'sep AUC':>8} {'epoch':>6} {'stop':>6}")
    for (arch, sd) in sorted(scored):
        s = scored[(arch, sd)]
        v = s["val"]
        print(f"  {arch:<13} {sd:>5} {v['macro_mae']:>10.4f} {v['mae']:>9.4f} "
              f"{v['rho']:>+8.3f} {v['sep']['gap']:>9.3f} "
              f"{v['sep']['auc']:>8.3f} {v['epoch']:>6} "
              f"{'yes' if s['early_stopped'] else 'CAP':>6}")

    print("\n=== across-seed summary (the selection table) ===")
    print(f"  {'arch':<13} {'macro MAE':>18} {'MAE':>17} {'rho':>16} "
          f"{'sep gap':>16} {'params':>8}")
    for arch, m in sorted(sel["summary"].items(),
                          key=lambda kv: kv[1]["macro_mae"]):
        mark = "  <-- selected" if arch == sel["arch"] else ""
        print(f"  {arch:<13} "
              f"{m['macro_mae']:>10.4f} +/-{m['macro_mae_sd']:<5.4f} "
              f"{m['mae']:>9.4f} +/-{m['mae_sd']:<5.4f} "
              f"{m['rho']:>+8.3f} +/-{m['rho_sd']:<5.3f} "
              f"{m['sep_gap']:>8.3f} +/-{m['sep_gap_sd']:<5.3f} "
              f"{m['params']:>8}" + mark)
    print(f"\n  selected {sel['arch']} seed {sel['seed']} "
          f"({sel['pick']} of seeds {sel['seed_order']})")
    print(f"  criterion: {sel['criterion']}")
    print("  NOTE selection read clean validation only. No attack response, "
          "no cal, no test.")


# ------------------------------------------------------------- freezing

def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def _pin(key):
    """ETag + size for an S3 object, so the manifest pins the exact bytes the
    probe was fitted on. Best effort -- a missing pin must not fail a lock."""
    try:
        h = _ensure_s3().head_object(Bucket=BUCKET, Key=key)
        return {"key": key, "etag": h["ETag"].strip('"'),
                "size": int(h["ContentLength"]),
                "last_modified": h["LastModified"].isoformat()}
    except Exception as e:
        return {"key": key, "error": f"{type(e).__name__}: {e}"}


def build_manifest(sel, scored, cal, cal_check, cfg, packs_kind=None):
    kind = packs_kind or PACK_KIND
    prefix = PACK_PREFIXES[kind]
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "probe": {"arch": sel["arch"], "seed": sel["seed"], "cfg": cfg,
                  "params": scored[(sel["arch"], sel["seed"])]["params"]},
        "selection": {
            "archs": sorted({a for a, _ in scored}),
            "seeds": sorted({s for _, s in scored}),
            "criterion": sel["criterion"],
            "pick": sel["pick"],
            "seed_order": sel["seed_order"],
            "summary": sel["summary"],
            "per_run_val": {f"{a}_seed{s}": scored[(a, s)]["val"]
                            for (a, s) in scored},
            "rule": "clean VideoPhy-2 validation only; the probe was never "
                    "selected on its response to the benchmark perturbations",
        },
        "calibration": {"fit_on": CAL_SPLIT, "n": cal["n"],
                        "method": cal["method"], "x": cal["x"], "y": cal["y"],
                        "lo": cal["lo"], "hi": cal["hi"], "check": cal_check},
        "encoder": dict(FROZEN_ENCODER),
        "preprocessing": dict(FROZEN_PREPROCESSING),
        "frame_sampling": dict(FROZEN_SAMPLING),
        "data": {
            "pack_kind": kind,
            "packs": {s: _pin(f"{prefix}/{s}.npz")
                      for s in ("train",) + HELD_OUT},
            "split": _pin("splits/videophy2_train/split_v1.json"),
        },
        "held_out_from_everything": [
            "videophy2_test", "implausibench_real", "implausibench_implausible",
        ],
    }


def freeze(result, sel, manifest, name="probe_locked", push_to_s3=True):
    """Write the locked instrument: weights + calibration + manifest."""
    import torch

    payload = {
        "locked": True,
        "state_dict": result["best"]["state"],
        "cfg": result["cfg"],
        "calibration": manifest["calibration"],
        "manifest": manifest,
        "val": {k: result["best"][k] for k in
                ("mae", "rmse", "macro_mae", "rho", "epoch")
                if k in result["best"]},
        "embed_dim": EMBED_DIM,
        "n_thresh": N_THRESH,
        "n_temporal": N_TEMPORAL,
    }
    buf = io.BytesIO()
    torch.save(payload, buf)

    pt, js = Path(f"./{name}.pt"), Path(f"./{name}.json")
    pt.write_bytes(buf.getvalue())
    js.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {pt}\nsaved -> {js}")

    if push_to_s3:
        s3 = _ensure_s3()
        s3.put_object(Bucket=BUCKET, Key=f"{PROBE_PREFIX}/{name}.pt",
                      Body=buf.getvalue())
        s3.put_object(Bucket=BUCKET, Key=f"{PROBE_PREFIX}/{name}.json",
                      Body=js.read_bytes())
        print(f"uploaded -> s3://{BUCKET}/{PROBE_PREFIX}/{name}.{{pt,json}}")
    return payload


def load_locked(name="probe_locked", device=None, local_first=True):
    """-> (model, calibration, manifest). The downstream entry point: this is
    what Step 12 uses to get dV without re-deriving anything."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    local = Path(f"./{name}.pt")
    if local_first and local.exists():
        ckpt = torch.load(local, map_location="cpu", weights_only=False)
        print(f"loading {local}")
    else:
        key = f"{PROBE_PREFIX}/{name}.pt"
        body = _ensure_s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
        ckpt = torch.load(io.BytesIO(body), map_location="cpu",
                          weights_only=False)
        print(f"loading s3://{BUCKET}/{key}")
    if not ckpt.get("locked"):
        raise ValueError(f"{name} is not a locked probe; run lock() first")

    model = make_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["calibration"], ckpt["manifest"]


def score(model, calibration, X, device, batch=1024):
    """Calibrated PC for a stack of embeddings. Use this everywhere downstream
    so no caller re-implements the calibration step."""
    return isotonic_apply(calibration, _predict(model, X, device, batch))


# --------------------------------------------------------------- the run

def lock(name="probe_locked", archs=SELECT_ARCHS, seeds=SEEDS, device=None,
         packs=None, pick="median", push_to_s3=True, **overrides):
    """Train, select, calibrate, freeze. The whole interface."""
    runs, packs, device = run_seeds(archs, seeds, device, packs, **overrides)
    scored, (val_X, val_y), (cal_X, cal_y) = score_runs(runs, packs, device)
    sel = select(scored, archs, seeds, pick=pick)
    print_selection(scored, sel, seeds)

    chosen = runs[(sel["arch"], sel["seed"])]
    model = _rebuild(chosen, device)

    # calibrate on cal ONLY: val picked the epoch and the architecture
    raw_cal = _predict(model, cal_X, device)
    cal = isotonic_fit(raw_cal, cal_y)
    chk = check_calibration(cal, raw_cal, cal_y)

    print(f"\n=== monotonic calibration (fit on {CAL_SPLIT}, "
          f"{cal['n']} clean clips) ===")
    print(f"  {len(cal['x'])} knots, {cal['method']}")
    print(f"  MAE  {chk['mae_before']:.4f} -> {chk['mae_after']:.4f}")
    print(f"  span {chk['span_before']:.3f} -> {chk['span_after']:.3f}  "
          f"(the labels span {SCALE_HI - SCALE_LO:.1f})")
    print(f"  no pair inverted: {'yes' if chk['monotone'] else 'NO -- BUG'}"
          f"   (rank rho {chk['rank_rho']:.4f}, "
          f"{chk['ties_created']} distinct values merged by pooling -- "
          "expected, and why rho is below 1)")
    if not chk["monotone"]:
        raise RuntimeError("the fitted calibration is not monotone; refusing "
                           "to freeze it")

    # val, calibrated -- reported, never used to select anything
    val_cal = clean_scores(isotonic_apply(cal, _predict(model, val_X, device)),
                           val_y)
    print(f"\n  val after calibration (reported, not used for selection): "
          f"macro MAE {val_cal['macro_mae']:.4f}  MAE {val_cal['mae']:.4f}  "
          f"rho {val_cal['rho']:+.3f}  sep gap {val_cal['sep']['gap']:.3f}")

    manifest = build_manifest(sel, scored, cal, chk, chosen["cfg"])
    manifest["selection"]["val_calibrated"] = val_cal
    payload = freeze(chosen, sel, manifest, name=name, push_to_s3=push_to_s3)

    print(f"\nLOCKED: {sel['arch']} seed {sel['seed']} -> {name}")
    print("  frozen: weights, calibration, encoder checkpoint, preprocessing, "
          "frame sampling")
    print("  next: run the locked probe over the benchmark corpus embeddings "
          "to get dV")
    return {"name": name, "selection": sel, "scored": scored,
            "calibration": cal, "calibration_check": chk,
            "manifest": manifest, "payload": payload, "runs": runs,
            "packs": packs, "device": device}


# --------------------------------------------------------------- selftest

def selftest():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}"
              + (f"   {detail}" if detail else ""))

    print("isotonic calibration")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = isotonic_fit(x, x)
    check("an already-monotone fit is reproduced",
          np.allclose(isotonic_apply(c, x), x))

    y = np.array([1.0, 3.0, 2.0, 4.0, 5.0])          # one violation
    c = isotonic_fit(x, y)
    out = isotonic_apply(c, x)
    check("violators are pooled", np.allclose(out, [1.0, 2.5, 2.5, 4.0, 5.0]),
          str(np.round(out, 3).tolist()))
    check("output is non-decreasing", bool((np.diff(out) >= -1e-12).all()))

    rng = np.random.default_rng(0)
    raw = rng.normal(3.4, 0.3, 400)                   # compressed, like ours
    lab = np.clip(np.round((raw - 3.4) * 4 + 3), 1, 5)
    c = isotonic_fit(raw, lab)
    chk = check_calibration(c, raw, lab)
    check("calibration inverts no pair", chk["monotone"],
          f"rho {chk['rank_rho']:.4f}, {chk['ties_created']} values merged")
    check("pooling merges values, so rho below 1 is expected not broken",
          chk["ties_created"] > 0 and chk["rank_rho"] < 1.0,
          f"rho {chk['rank_rho']:.4f}")
    # the regression test for the check itself: a deliberately inverted map
    # must be caught, or `monotone` is decorative
    flipped = {"x": c["x"], "y": sorted(c["y"], reverse=True),
               "lo": c["lo"], "hi": c["hi"]}
    check("an inverted map is caught",
          not check_calibration(flipped, raw, lab)["monotone"])
    check("calibration cannot worsen fit-set MAE",
          chk["mae_after"] <= chk["mae_before"] + 1e-9,
          f"{chk['mae_before']:.4f} -> {chk['mae_after']:.4f}")
    check("calibration widens a compressed span",
          chk["span_after"] > chk["span_before"],
          f"{chk['span_before']:.3f} -> {chk['span_after']:.3f}")
    cal_out = isotonic_apply(c, raw)
    check("calibrated output stays on the 1-5 scale",
          bool((cal_out >= 1.0).all() and (cal_out <= 5.0).all()))
    check("outside the fitted range the map is flat, not extrapolated",
          bool(np.isclose(isotonic_apply(c, np.array([-99.0]))[0], min(c["y"]))
               and np.isclose(isotonic_apply(c, np.array([99.0]))[0],
                              max(c["y"]))))
    dup = isotonic_fit(np.array([1., 1., 2., 2.]), np.array([1., 3., 2., 4.]))
    check("duplicate x collapse to strictly increasing knots",
          bool((np.diff(dup["x"]) > 0).all()), str(dup["x"]))

    print("separation and AUC")
    check("AUC is 1.0 when the classes are cleanly split",
          auc([5, 4, 6], [1, 2, 3]) == 1.0)
    check("AUC is 0.5 when every value is tied",
          auc([1, 1], [1, 1]) == 0.5, str(auc([1, 1], [1, 1])))
    s = separation(np.array([1., 1., 5., 5.]), np.array([1, 2, 4, 5]))
    check("separation reports the PC<=2 vs PC>=4 gap", s["gap"] == 4.0)
    check("separation counts both sides", s["n_lo"] == 2 and s["n_hi"] == 2)
    s2 = separation(np.array([1., 2., 3.]), np.array([3, 3, 3]))
    check("separation is NaN when a side is empty", s2["gap"] != s2["gap"])

    print("clean scores")
    cs = clean_scores(np.array([1., 2., 3., 4., 5.]), np.array([1, 2, 3, 4, 5]))
    # rho is compared with a tolerance: _spearman returns 0.9999999999999999
    # on an exact match, and an == 1.0 test would fail on float noise alone
    check("a perfect prediction scores 0 MAE and rho 1",
          cs["mae"] == 0.0 and cs["macro_mae"] == 0.0
          and abs(cs["rho"] - 1.0) < 1e-9, f"rho {cs['rho']:.12f}")
    cs = clean_scores(np.array([3., 3., 3., 3.]), np.array([1, 1, 1, 5]))
    check("macro MAE refuses to ignore a rare level",
          abs(cs["macro_mae"] - 2.0) < 1e-12 and abs(cs["mae"] - 2.0) < 1e-12,
          f"macro {cs['macro_mae']:.3f} mae {cs['mae']:.3f}")

    print("selection cannot see an attack response")
    try:
        _assert_clean({"mae": 0.5, "mean_gap": 0.1})
        check("an attack key reaching selection raises", False)
    except ValueError:
        check("an attack key reaching selection raises", True)
    stripped = clean_only({"mae": 0.5, "macro_mae": 0.9, "rho": 0.4,
                           "epoch": 40, "mean_gap": 0.1, "signed": {}})
    check("clean_only keeps only clean keys",
          set(stripped) == {"mae", "macro_mae", "rho", "epoch"},
          str(sorted(stripped)))

    print("architecture selection")
    fake = {}
    for a, macro, rho in (("mean_linear", 0.90, 0.40),
                          ("proj_mean", 0.85, 0.45)):
        for i, sd in enumerate(SEEDS):
            fake[(a, sd)] = {"val": {"mae": macro,
                                     "macro_mae": macro + i * 0.01,
                                     "rho": rho, "epoch": 40,
                                     "sep": {"gap": 1.0, "auc": 0.7}},
                             "cal": {}, "epochs": 40, "early_stopped": True,
                             "params": 100}
    sel = select(fake)
    check("the better macro MAE wins", sel["arch"] == "proj_mean", sel["arch"])
    check("the median seed ships, not the best", sel["seed"] == SEEDS[1],
          f"seed {sel['seed']} of order {sel['seed_order']}")
    check("pick='best' ships the best seed",
          select(fake, pick="best")["seed"] == SEEDS[0])
    check("the summary carries an across-seed sd",
          sel["summary"]["proj_mean"]["macro_mae_sd"] > 0)

    tie = {}
    for a, rho in (("mean_linear", 0.30), ("proj_mean", 0.50)):
        for sd in SEEDS:
            tie[(a, sd)] = {"val": {"mae": 0.8, "macro_mae": 0.8, "rho": rho,
                                    "epoch": 1,
                                    "sep": {"gap": 1.0, "auc": 0.7}},
                            "cal": {}, "epochs": 1, "early_stopped": True,
                            "params": 1}
    check("Spearman breaks a macro-MAE tie",
          select(tie)["arch"] == "proj_mean")

    print("-" * 68)
    print("selftest OK" if ok else "selftest FAILED")
    return ok


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(
        description="select one reference probe on clean validation, then "
                    "freeze it with its calibration")
    ap.add_argument("--name", default="probe_locked")
    ap.add_argument("--archs", nargs="+", default=list(SELECT_ARCHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--pick", choices=["median", "best"], default="median")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    lock(name=a.name, archs=tuple(a.archs), seeds=tuple(a.seeds),
         device=a.device, pick=a.pick, push_to_s3=not a.no_push)
