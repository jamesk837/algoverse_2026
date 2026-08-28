"""Run the LOCKED probe over the benchmark corpus and write dV.

This is the bridge between phase 2 and phase 1. The probe was built on
videophy2_train (train/val/cal); the judges were scored on the benchmark
corpus (videophy2_test + ImplausiBench). Step 12's headline number is

    d = dJ - dV

so every clip x variant the judges saw needs the reference's answer too.
Nothing here touches a GPU-bound encoder: the embeddings are already cached,
so this is a 266K-parameter model doing ~7.5k forward passes. Minutes.

Two rules the file enforces rather than documents:

  * only a LOCKED probe is accepted. An unlocked checkpoint has no calibration
    and no manifest, so a dV computed from one is not the project's reference
    and could not be reproduced later.
  * `identity` is its own kind, never pooled into the invariance half. It is a
    codec control, not an attack -- averaging it into "did the score stay
    still" would corrupt exactly the number Step 12 reports.

Colab:
    !wget -q -O train_probe.py  <raw>/train_probe.py
    !wget -q -O lock_probe.py   <raw>/lock_probe.py
    !wget -q -O score_corpus.py <raw>/score_corpus.py
    import score_corpus
    r = score_corpus.run()

EC2:
    python score_corpus.py --probe probe_locked
    python score_corpus.py --selftest
"""

import argparse
import io
import json
import os
import time
from collections import defaultdict

import numpy as np

# lock_probe adopts everything from train_probe and raises loudly if it
# cannot, so depending on it is one dependency instead of two.
import lock_probe as L

BUCKET = L.BUCKET
REFERENCE_PREFIX = "reference"

BENCHMARK_DATASETS = ("videophy2_test", "implausibench_real",
                      "implausibench_implausible")

# Not an attack. See judge_harness.CONTROL_FILES / attack_suite.attack_identity.
CONTROL_VARIANTS = ("identity",)

# VideoPhy-2's human PC is 1-5; the doc's pre-specified normalisation is
# (x-1)/4 so every judge and the reference land on a common 0-1 scale.
SCALE_SPAN = 4.0

N_BOOT = 2000


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def kind_of(variant):
    """The 2x2 taxonomy plus the control, which belongs to neither half.

    train_probe.variant_kind() only knows temporal vs superficial, so calling
    it on `identity` would silently file the codec control under
    expected-invariance and fold it into mean_gap.
    """
    if variant in CONTROL_VARIANTS:
        return "control"
    return L.variant_kind(variant)


def normalise(pc):
    """1-5 -> 0-1, the doc's pre-specified monotonic endpoint mapping."""
    return (np.asarray(pc, dtype=np.float64) - 1.0) / SCALE_SPAN


# ------------------------------------------------------------- scoring

def score_dataset(model, calibration, dataset, device, pack=None):
    """-> per-clip calibrated PC and the within-clip delta for every variant.

    Deltas are paired within a clip by construction: build_eval keeps every
    variant row aligned to its own clean row and masks the ones that were
    never rendered, so a missing variant is skipped rather than compared
    against someone else's clean clip.
    """
    if pack is None:
        pack = L.load_pack(dataset)
    clean, y, variants, stems = L.build_eval(pack)

    raw_clean = L._predict(model, clean, device)
    cal_clean = L.isotonic_apply(calibration, raw_clean)

    clips = {}
    for j, stem in enumerate(stems):
        clips[stem] = {
            # -1 is embed_vjepa's "no human label" (ImplausiBench has none)
            "pc_human": None if int(y[j]) < 0 else int(y[j]),
            "clean": {"raw": float(raw_clean[j]),
                      "pc": float(cal_clean[j]),
                      "norm": float(normalise(cal_clean[j]))},
            "variants": {},
        }

    per_variant = {}
    for name in sorted(variants):
        X, mask = variants[name]
        if not mask.any():
            continue
        raw_v = L._predict(model, X, device)
        cal_v = L.isotonic_apply(calibration, raw_v)
        d = cal_v - cal_clean

        for j, stem in enumerate(stems):
            if not mask[j]:
                continue
            clips[stem]["variants"][name] = {
                "raw": float(raw_v[j]), "pc": float(cal_v[j]),
                "norm": float(normalise(cal_v[j])),
                "d_pc": float(d[j]), "d_norm": float(d[j] / SCALE_SPAN),
            }

        dm = d[mask]
        lo, hi = L._boot_ci(dm, N_BOOT)
        per_variant[name] = {
            "kind": kind_of(name), "n": int(mask.sum()),
            "d_pc": float(dm.mean()), "d_pc_lo": float(lo), "d_pc_hi": float(hi),
            "d_norm": float(dm.mean() / SCALE_SPAN),
            "d_norm_lo": float(lo / SCALE_SPAN),
            "d_norm_hi": float(hi / SCALE_SPAN),
            "abs_d_pc": float(np.abs(dm).mean()),
        }

    return {"dataset": dataset, "n_clips": len(stems),
            "clean_mean_pc": float(cal_clean.mean()),
            "clips": clips, "per_variant": per_variant}


def print_table(res, indent="  "):
    """Signed delta first: it is the whole taxonomy. A temporal perturbation
    should push the score DOWN, a superficial one should not move it, and the
    control says how much of either is just the codec."""
    pv = res["per_variant"]
    if not pv:
        print(f"{indent}no variants scored")
        return
    w = 34
    print(f"{indent}{'variant':<{w}} {'n':>5} {'d PC':>9} "
          f"{'95% CI':>19} {'|d| PC':>9} {'d norm':>9}")

    order = {"temporal": 0, "superficial": 1, "control": 2}
    groups = defaultdict(list)
    for name, r in pv.items():
        groups[r["kind"]].append((name, r))

    for kind in sorted(groups, key=lambda k: order.get(k, 9)):
        note = {"temporal": "expected-sensitivity  (should go DOWN)",
                "superficial": "expected-invariance   (should not move)",
                "control": "codec control         (subtract this)"}[kind]
        print(f"{indent}-- {note}")
        for name, r in sorted(groups[kind]):
            flag = ""
            if kind == "temporal" and r["d_pc_lo"] > 0:
                flag = "  <-- ROSE on a temporal attack"
            elif kind == "superficial" and r["d_pc_lo"] > 0:
                flag = "  <-- INFLATED by a superficial cue"
            print(f"{indent}{name:<{w}} {r['n']:>5} {r['d_pc']:>+9.4f} "
                  f"[{r['d_pc_lo']:>+8.4f},{r['d_pc_hi']:>+8.4f}] "
                  f"{r['abs_d_pc']:>9.4f} {r['d_norm']:>+9.4f}{flag}")

    if "identity" not in pv:
        print(f"{indent}NOTE no `identity` rows -- the codec control is not "
              "embedded for this corpus,")
        print(f"{indent}     so dV here still mixes the attack with one "
              "libx264 pass. Embed it with")
        print(f"{indent}     embed_vjepa and rerun to correct dV the way "
              "dJ gets corrected.")


def compact(results):
    """{variant: [d_norm, lo, hi]} pooled over datasets, weighted by n.

    The shape annotate.compare_to_vjepa() accepts, so the human study can be
    compared against the reference without anyone re-deriving a delta.
    """
    acc = defaultdict(lambda: {"n": 0, "s": 0.0, "lo": 0.0, "hi": 0.0})
    for res in results.values():
        for name, r in res["per_variant"].items():
            a = acc[name]
            a["n"] += r["n"]
            a["s"] += r["d_norm"] * r["n"]
            a["lo"] += r["d_norm_lo"] * r["n"]
            a["hi"] += r["d_norm_hi"] * r["n"]
    return {k: [v["s"] / v["n"], v["lo"] / v["n"], v["hi"] / v["n"]]
            for k, v in acc.items() if v["n"]}


# ------------------------------------------------------------------ run

def run(name="probe_locked", datasets=BENCHMARK_DATASETS, device=None,
        push_to_s3=True):
    model, calibration, manifest = L.load_locked(name, device=device)
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    probe = manifest["probe"]
    print(f"\nlocked probe: {probe['arch']} seed {probe['seed']}, "
          f"{probe['params']} params")
    print(f"  calibration fit on {manifest['calibration']['fit_on']}, "
          f"{manifest['calibration']['n']} clips, "
          f"{len(manifest['calibration']['x'])} knots")
    print(f"  frozen {manifest['created']}"
          + (f", git {manifest['git_sha'][:7]}" if manifest.get("git_sha")
             else ""))

    results, failed = {}, []
    for ds in datasets:
        print(f"\n{'=' * 70}\n{ds}\n{'=' * 70}")
        try:
            res = score_dataset(model, calibration, ds, device)
        except Exception as e:
            print(f"DATASET FAILED {ds}: {type(e).__name__}: {e}")
            failed.append((ds, f"{type(e).__name__}: {e}"))
            continue
        results[ds] = res
        print(f"  {res['n_clips']} clips, clean mean calibrated PC "
              f"{res['clean_mean_pc']:.3f}")
        print()
        print_table(res)

    if not results:
        raise RuntimeError("no dataset scored; nothing to write")

    payload = {
        "probe": name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": manifest,
        "normalisation": "calibrated PC in 1-5; norm = (pc - 1) / 4",
        "control_variants": list(CONTROL_VARIANTS),
        "datasets": results,
        "failed": failed,
    }
    summary = compact(results)

    print(f"\n{'=' * 70}\npooled dV across {len(results)} corpora "
          f"(normalised 0-1 units)")
    for v, (d, lo, hi) in sorted(summary.items(),
                                 key=lambda kv: kind_of(kv[0])):
        print(f"  {v:<34} {kind_of(v):<12} {d:>+9.4f}  "
              f"[{lo:>+8.4f}, {hi:>+8.4f}]")

    if push_to_s3:
        s3 = L._ensure_s3()
        base = f"{REFERENCE_PREFIX}/{name}"
        s3.put_object(Bucket=BUCKET, Key=f"{base}/dv.json",
                      Body=json.dumps(payload).encode())
        s3.put_object(Bucket=BUCKET, Key=f"{base}/vjepa_deltas.json",
                      Body=json.dumps(summary, indent=2).encode())
        print(f"\nuploaded -> s3://{BUCKET}/{base}/dv.json")
        print(f"uploaded -> s3://{BUCKET}/{base}/vjepa_deltas.json")

    with open("vjepa_deltas.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("saved -> ./vjepa_deltas.json  "
          "(annotate.py --report --compare vjepa_deltas.json)")

    if failed:
        print("\na clean exit is not a clean run:")
        for ds, err in failed:
            print(f"  FAILED {ds}: {err}")
    return {"payload": payload, "summary": summary, "results": results}


def load_dv(name="probe_locked"):
    """Read back what run() wrote -- the entry point for Step 12."""
    key = f"{REFERENCE_PREFIX}/{name}/dv.json"
    body = L._ensure_s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return json.loads(body)


# -------------------------------------------------------------- selftest

def selftest():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}"
              + (f"   {detail}" if detail else ""))

    print("taxonomy")
    check("shuffle is expected-sensitivity", kind_of("shuffle") == "temporal")
    check("an overlay is expected-invariance",
          kind_of("caption_echo_rubric_vocab") == "superficial")
    check("identity is a control, not an attack",
          kind_of("identity") == "control", kind_of("identity"))
    check("the control is NOT filed under invariance",
          kind_of("identity") != L.variant_kind("identity"),
          f"train_probe would say {L.variant_kind('identity')!r}")

    print("normalisation")
    check("PC 1 maps to 0.0", normalise(1.0) == 0.0)
    check("PC 5 maps to 1.0", normalise(5.0) == 1.0)
    check("a one-point PC delta is 0.25 normalised",
          abs((normalise(4.0) - normalise(3.0)) - 0.25) < 1e-12)

    print("pooling")
    fake = {
        "a": {"per_variant": {"shuffle": {"n": 100, "d_norm": -0.10,
                                          "d_norm_lo": -0.15,
                                          "d_norm_hi": -0.05, "kind": "temporal"}}},
        "b": {"per_variant": {"shuffle": {"n": 300, "d_norm": -0.20,
                                          "d_norm_lo": -0.25,
                                          "d_norm_hi": -0.15, "kind": "temporal"}}},
    }
    got = compact(fake)["shuffle"][0]
    check("pooling is weighted by clip count, not a mean of means",
          abs(got - (-0.175)) < 1e-12, f"{got:.4f} (unweighted would be -0.15)")
    check("a variant present in only one corpus survives pooling",
          "shuffle" in compact({"a": fake["a"]}))
    check("an empty result pools to nothing", compact({}) == {})

    print("-" * 68)
    print("selftest OK" if ok else "selftest FAILED")
    return ok


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(
        description="run the locked probe over the benchmark corpus -> dV")
    ap.add_argument("--probe", default="probe_locked")
    ap.add_argument("--datasets", nargs="+", default=list(BENCHMARK_DATASETS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    run(name=a.probe, datasets=tuple(a.datasets), device=a.device,
        push_to_s3=not a.no_push)
