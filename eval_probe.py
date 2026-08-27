

import argparse
import io
import os
from pathlib import Path

import numpy as np

# On EC2 these import from the sibling file. In Colab train_probe.py is pasted
# into an earlier cell, so the names are already global and the import fails
# harmlessly.
try:
    from train_probe import (BUCKET, PROBE_PREFIX, PC_LEVELS, N_THRESH,
                             TEMPORAL_ARCHS, TEMPORAL_VARIANTS,
                             attention_stats, build_eval, compression,
                             concat_packs, consistency_loss, expected_pc,
                             load_pack, make_model, ordinal_targets,
                             print_compression_table, print_variant_table,
                             threshold_pos_weight, variant_kind, _boot_ci,
                             _cos_features, _ensure_s3, _rankdata, _spearman)
except ImportError:
    pass

# duplicated from train_probe rather than imported, the way diagnose_probe
# already does it: that import is best-effort (it fails by design in Colab,
# where the names are already global) and a default argument is bound at def
# time, so it cannot depend on the import having worked.
HELD_OUT = ("val", "cal")

CONFORMAL_ALPHA = 0.10
N_BOOT = 2000


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def spearman(a, b):
    """Tie-corrected rank correlation, without scipy.

    Delegates to train_probe._spearman so this file and the training loop can
    never report two different rhos. It used to break ties by argsort order,
    which is not a rank: the target is an integer PC in 1..5, so the ties are
    groups of hundreds of clips.
    """
    return _spearman(a, b)


# ------------------------------------------------------------- selftest

def _raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


def selftest():
    """Check the ordinal machinery on synthetic data, before trusting any
    number computed with it."""
    import torch

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}" +
              (f": {detail}" if detail else ""))
        ok = ok and bool(cond)

    print("-" * 68)

    pc = torch.tensor([1., 2., 3., 4., 5.])
    t = ordinal_targets(pc)
    want = torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.], [1., 1., 0., 0.],
                         [1., 1., 1., 0.], [1., 1., 1., 1.]])
    check("ordinal_targets maps PC 1-5 to cumulative thresholds",
          torch.equal(t, want))

    big = torch.full((1, N_THRESH), 20.0)
    check("expected_pc saturates at 5", abs(expected_pc(big).item() - 5.0) < 1e-3,
          f"{expected_pc(big).item():.4f}")
    check("expected_pc floors at 1", abs(expected_pc(-big).item() - 1.0) < 1e-3,
          f"{expected_pc(-big).item():.4f}")
    check("expected_pc is 3 at zero logits",
          abs(expected_pc(torch.zeros(1, N_THRESH)).item() - 3.0) < 1e-6)

    steps = torch.linspace(-8, 8, 50).unsqueeze(1).repeat(1, N_THRESH)
    e = expected_pc(steps)
    check("expected_pc is monotonic in the logits", bool((e.diff() > 0).all()))

    # a PC distribution where only 1 clip in 10 is PC=1
    y = torch.tensor([1] + [4] * 9)
    pw = threshold_pos_weight(y)
    check("pos_weight upweights the rare side of each threshold",
          abs(pw[0].item() - 1 / 9) < 1e-6 and abs(pw[3].item() - 10.0) < 1e-6,
          f"{[round(v, 3) for v in pw.tolist()]}")

    a = torch.randn(16, N_THRESH)
    for kind in ("mse", "kl"):
        z = consistency_loss(a, a.clone(), kind).item()
        d = consistency_loss(a, torch.randn(16, N_THRESH), kind).item()
        sym = abs(consistency_loss(a, a + 1, kind).item()
                  - consistency_loss(a + 1, a, kind).item())
        check(f"consistency[{kind}] is 0 on identical inputs", z < 1e-9,
              f"{z:.2e}")
        check(f"consistency[{kind}] is positive on different inputs", d > 0,
              f"{d:.4f}")
        check(f"consistency[{kind}] is symmetric", sym < 1e-6, f"{sym:.2e}")

    # --- tie-corrected ranks. The old argsort(argsort()) passed none of these.
    r = _rankdata([10., 10., 20., 5.])
    check("_rankdata gives ties their midrank",
          np.allclose(r, [2.5, 2.5, 4.0, 1.0]), f"{r.tolist()}")
    check("_rankdata is invariant to the input order of tied values",
          np.allclose(_rankdata([1., 1., 1.]), [2., 2., 2.]))
    a = np.array([1., 2., 2., 3., 4.])
    check("spearman is 1.0 on a monotone map with ties",
          abs(spearman(a, 2 * a + 1) - 1.0) < 1e-12,
          f"{spearman(a, 2 * a + 1):.6f}")
    check("spearman is -1.0 on a reversed map with ties",
          abs(spearman(a, -a) + 1.0) < 1e-12, f"{spearman(a, -a):.6f}")
    check("spearman is nan on a constant prediction",
          np.isnan(spearman(np.ones(8), np.arange(8.))))
    # A case where ignoring ties is provably wrong. The two PC groups have the
    # SAME predicted values, so there is no rank information at all and rho
    # must be 0. Breaking the ties by argsort order instead correlates each
    # group with its own index order and invents ~+0.29 out of nothing.
    yy = np.array([2., 2., 2., 2., 1., 1., 1., 1.])
    pp = np.array([1., 2., 3., 4., 1., 2., 3., 4.])
    old = float(np.corrcoef(np.argsort(np.argsort(pp)).astype(float),
                            np.argsort(np.argsort(yy)).astype(float))[0, 1])
    check("spearman is 0 where ties carry no information",
          abs(spearman(pp, yy)) < 1e-12 and abs(old) > 0.2,
          f"tie-corrected {spearman(pp, yy):+.4f} vs ordinal {old:+.4f}")

    # --- bootstrap CI
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 4000)
    lo, hi = _boot_ci(x, 1000)
    check("_boot_ci brackets the sample mean", lo < x.mean() < hi,
          f"[{lo:.4f}, {hi:.4f}] around {x.mean():.4f}")
    check("_boot_ci of a constant has zero width",
          abs(np.subtract(*_boot_ci(np.full(50, 3.0), 200))) < 1e-12)
    check("_boot_ci is nan for n<2", np.isnan(_boot_ci([1.0], 100)[0]))

    # --- cosine between clean and variant features
    f = rng.normal(size=(7, 32, 1024)).astype(np.float32)
    check("_cos_features is 1.0 on identical features",
          np.allclose(_cos_features(f, f), 1.0, atol=1e-5))
    check("_cos_features is -1.0 on negated features",
          np.allclose(_cos_features(f, -f), -1.0, atol=1e-5))
    check("_cos_features sees a time reversal (per-moment alignment)",
          float(_cos_features(f, f[:, ::-1]).mean()) < 0.2,
          f"{float(_cos_features(f, f[:, ::-1]).mean()):.4f}")
    check("_cos_features handles the rank-2 mean-pooled packs",
          _cos_features(f[:, 0], f[:, 0]).shape == (7,))

    # --- attention: init must not be flat, entropy must be reported
    torch.manual_seed(0)
    acfg = {"hidden": 256, "dropout": 0.0, "arch": "attn", "n_temporal": 32}
    amodel = make_model(acfg)
    ax = torch.randn(64, 32, 1024)
    with torch.no_grad():
        w = amodel.attention(ax)
        ent = amodel.attention_entropy(ax)
    uni = float(np.log(32))
    check("attention weights sum to 1",
          bool(torch.allclose(w.sum(-1), torch.ones(64), atol=1e-5)))
    check("attention entropy matches -sum p log p",
          bool(torch.allclose(ent, -(w * w.clamp_min(1e-9).log()).sum(-1),
                              atol=1e-5)))
    check("attention entropy is at or below the uniform ceiling",
          bool((ent <= uni + 1e-5).all()), f"max {ent.max():.4f} vs {uni:.4f}")
    # the whole point of scale=1.0 + std=0.2: at std=0.02 with an extra
    # 1/sqrt(d) the mean max weight was 0.0317 against a uniform 0.03125
    check("attention is NOT flat at init (scale=1.0, query std=0.2)",
          float(w.max(-1).values.mean()) > 0.05,
          f"mean max weight {float(w.max(-1).values.mean()):.4f} vs uniform "
          f"{1/32:.4f}, H/logT {float(ent.mean())/uni:.4f}")
    check("attention_stats reports entropy for the attentive probe",
          (attention_stats(amodel, ax.numpy(), "cpu") or {}).get("mean")
          is not None)
    check("attention_stats returns None for the mean-pooled mlp",
          attention_stats(make_model({"hidden": 32, "dropout": 0.0}),
                          np.random.randn(4, 1024).astype(np.float32),
                          "cpu") is None)

    # --- the ablation archs: which ones can see time at all is structural,
    # not something to discover empirically after a 40-minute training run
    torch.manual_seed(0)
    xt = torch.randn(8, 32, 1024)
    perm = xt[:, torch.randperm(32)]
    order_blind = {"mean_linear", "proj_mean"}
    for arch in TEMPORAL_ARCHS:
        torch.manual_seed(0)
        m = make_model({"hidden": 32, "dropout": 0.0, "arch": arch,
                        "n_temporal": 32}).eval()
        with torch.no_grad():
            same = torch.allclose(m(xt), m(perm), atol=1e-5)
        want_blind = arch in order_blind
        check(f"{arch} is {'order-INVARIANT' if want_blind else 'order-SENSITIVE'}"
              " by construction", same is want_blind,
              f"max |d| under a time permutation "
              f"{float((m(xt) - m(perm)).abs().max()):.2e}")
        check(f"{arch} rejects the rank-2 mean-pooled packs",
              _raises(lambda mm=m: mm(torch.randn(8, 1024)), RuntimeError))
        check(f"{arch} emits {N_THRESH} cumulative logits",
              tuple(m(xt).shape) == (8, N_THRESH))

    # --- compression: span and slope must mean what the table says they mean
    yv = np.array([1, 2, 3, 4, 5] * 4)
    c_id = compression(yv.astype(float), yv)
    check("compression is slope 1.0 / span 4.0 on a perfect prediction",
          abs(c_id["slope"] - 1.0) < 1e-9 and abs(c_id["span"] - 4.0) < 1e-9,
          f"slope {c_id['slope']:.4f}, span {c_id['span']:.4f}")
    c_flat = compression(np.full(len(yv), 3.7), yv)
    check("compression is slope 0 / span 0 on a constant prediction",
          abs(c_flat["slope"]) < 1e-9 and abs(c_flat["span"]) < 1e-9)
    c_comp = compression(3.0 + 0.25 * (yv - 3), yv)
    check("compression recovers a deliberately compressed slope",
          abs(c_comp["slope"] - 0.25) < 1e-9 and abs(c_comp["span"] - 1.0) < 1e-9,
          f"slope {c_comp['slope']:.4f}, span {c_comp['span']:.4f}")
    check("compression counts each level and keeps them separate",
          c_id["counts"] == {l: 4 for l in PC_LEVELS} and c_id["n"] == 20)

    # --- pooling val and cal into one reporting set must not merge clips
    def _pack(stems, variants, offset):
        rows = [(st, v) for st in stems for v in variants]
        return {"X": np.stack([np.full(8, offset + i, dtype=np.float32)
                               for i in range(len(rows))]),
                "stem": np.array([r[0] for r in rows], dtype="<U160"),
                "variant": np.array([r[1] for r in rows], dtype="<U64"),
                "pc": np.array([3] * len(rows), dtype=np.int8),
                "sa": np.array([4] * len(rows), dtype=np.int8),
                "joint": np.array([2] * len(rows), dtype=np.int8)}

    pv = _pack(["v0", "v1"], ["clean", "shuffle", "photometric"], 0)
    pc_ = _pack(["c0"], ["clean", "shuffle"], 100)
    pooled = concat_packs([pv, pc_])
    check("concat_packs keeps every row",
          len(pooled["X"]) == len(pv["X"]) + len(pc_["X"]),
          f"{len(pooled['X'])} rows")
    pc_clean, pc_y, pc_var, pc_stems = build_eval(pooled)
    check("pooled build_eval sees every clip and keeps stems distinct",
          len(pc_stems) == 3 and sorted(pc_stems) == ["c0", "v0", "v1"])
    check("a variant missing on one split is masked, not dropped",
          bool(pc_var["photometric"][1].tolist() ==
               [s in ("v0", "v1") for s in pc_stems]))
    check("pooled rows keep their own features",
          bool((pc_clean[pc_stems.index("c0")] == 100).all()))
    check("concat_packs rejects mismatched feature shapes",
          _raises(lambda: concat_packs(
              [pv, dict(pc_, X=np.zeros((len(pc_["X"]), 9), np.float32))])))

    # --- taxonomy: the invariance gap must not average in a variant that is
    # supposed to move
    check("variant_kind splits the 2x2 taxonomy",
          all(variant_kind(v) == "temporal" for v in TEMPORAL_VARIANTS)
          and variant_kind("caption_echo_rubric_vocab") == "superficial"
          and variant_kind("photometric") == "superficial")

    # can the architecture actually fit something? 64 separable examples.
    torch.manual_seed(0)
    cfg = {"hidden": 32, "dropout": 0.0}
    model = make_model(cfg)
    X = torch.randn(64, 1024)
    y = torch.randint(1, 6, (64,))
    X = X + y.unsqueeze(1).float() * 0.5      # make PC linearly recoverable
    tgt = ordinal_targets(y.float())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(300):
        loss = bce(model(X), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        mae = (expected_pc(model(X)) - y.float()).abs().mean().item()
    check("the head can overfit a separable toy set", mae < 0.25,
          f"train mae {mae:.4f}")

    print("-" * 68)
    print("selftest OK" if ok else "selftest FAILED")
    return ok


# --------------------------------------------------------------- report

def load_probe(name="probe_v1", local_first=True):
    import torch

    local = Path(f"./{name}.pt")
    if local_first and local.exists():
        print(f"loading {local}")
        return torch.load(local, map_location="cpu", weights_only=False)
    key = f"{PROBE_PREFIX}/{name}.pt"
    print(f"loading s3://{BUCKET}/{key}")
    body = _ensure_s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return torch.load(io.BytesIO(body), map_location="cpu", weights_only=False)


def predict(model, X, device, batch=1024):
    """E[PC] = 1 + sum_k P(PC > k), so the per-variant deltas keep a continuous
    scale."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            chunk = torch.as_tensor(X[i:i + batch], device=device)
            out.append(expected_pc(model(chunk)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def monotonicity(model, X, device, batch=1024):
    """Fraction of clips whose cumulative probabilities are not non-increasing.

    Nothing constrains P(PC>1) >= P(PC>2) >= ...; a high rate means E[PC] is
    still usable but the per-threshold probabilities should not be read as a
    distribution.
    """
    import torch
    bad = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            chunk = torch.as_tensor(X[i:i + batch], device=device)
            p = torch.sigmoid(model(chunk))
            bad += int((p.diff(dim=-1) > 1e-6).any(dim=-1).sum())
    return bad / max(len(X), 1)


def report(name="probe_v1", device=None, packs=None, result=None,
           splits=HELD_OUT):
    """result= evaluates a train() return value straight from memory, so a probe
    can be tested before it is ever written anywhere.

    splits= is the reporting set, pooled. It defaults to every held-out split
    because val and cal are 10% each, so a val-only table reports half the
    clips that exist. Selection still happened on val alone, which is why the
    per-split breakdown is printed too: val is mildly optimistic, cal is not.
    splits=("val",) reproduces the pre-2026-08-21 report."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if result is not None:
        ckpt = {"state_dict": result["best"]["state"], "cfg": result["cfg"],
                "val": result["best"]}
        print("evaluating the in-memory probe (nothing loaded from disk or s3)")
    else:
        ckpt = load_probe(name)
    model = make_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cfg = ckpt["cfg"]
    print(f"\nprobe trained with lambda_cons={cfg['lambda_cons']}, "
          f"alpha={cfg['alpha']}, consistency={cfg['consistency']}, "
          f"best epoch {ckpt['val']['epoch']}")

    wanted = tuple(dict.fromkeys(tuple(splits) + ("cal",)))
    if packs is None:
        print("\nloading packs ...")
        packs = {s: load_pack(s) for s in wanted}

    splits = tuple(s for s in splits if s in packs)
    if not splits:
        raise ValueError(f"none of the requested splits were loaded: {wanted}")
    label = "+".join(splits)

    clean, y, variants, _ = build_eval(
        concat_packs([packs[s] for s in splits]))
    pred = predict(model, clean, device)
    yf = y.astype(np.float64)

    print(f"\n=== clean performance ({label}, {len(y)} clips) ===")
    mae = np.abs(pred - yf).mean()
    rmse = np.sqrt(((pred - yf) ** 2).mean())
    const = np.abs(yf - yf.mean()).mean()
    per_level = [np.abs(pred[y == lev] - yf[y == lev]).mean()
                 for lev in PC_LEVELS if (y == lev).any()]
    macro_mae = float(np.mean(per_level)) if per_level else float("nan")
    rho = spearman(pred, yf)

    # all three side by side, whichever one the run happened to select on
    print(f"  {'MAE':<12} {'macro MAE':<12} {'Spearman':<12}   "
          f"(selected on {cfg.get('select', 'mae')})")
    print(f"  {mae:<12.4f} {macro_mae:<12.4f} {rho:<+12.4f}")
    print(f"  RMSE                 {rmse:.4f}")
    print(f"  MAE, constant pred   {const:.4f}   "
          f"({'better' if mae < const else 'WORSE'} than predicting the mean)")
    print("  macro MAE averages the five per-level MAEs, so the rare PC levels "
          "cannot be\n  ignored for free; plain MAE is dominated by PC 3-4.")
    print("  Spearman is tie-corrected (midranks) -- the earlier ordinal-rank "
          "version\n  broke the PC ties arbitrarily and is not comparable.")

    if len(splits) > 1:
        print(f"\n  per split, so the pooled number cannot hide one of them")
        print(f"  {'split':<8} {'n':>5} {'MAE':>9} {'macro MAE':>11} "
              f"{'Spearman':>10}")
        for split in splits:
            sc, sy, _, _ = build_eval(packs[split])
            sp = predict(model, sc, device)
            syf = sy.astype(np.float64)
            lv = [np.abs(sp[sy == l] - syf[sy == l]).mean()
                  for l in PC_LEVELS if (sy == l).any()]
            note = ("used to pick the epoch" if split == "val"
                    else "never used for selection")
            print(f"  {split:<8} {len(sy):>5} {np.abs(sp - syf).mean():>9.4f} "
                  f"{np.mean(lv):>11.4f} {spearman(sp, syf):>+10.4f}   {note}")

    comp = compression(pred, y)
    print(f"\n  {'pc':>4} {'n':>5} {'mean pred':>11} {'MAE':>8}")
    for level in PC_LEVELS:
        m = y == level
        if not m.any():
            continue
        print(f"  {level:>4} {int(m.sum()):>5} {pred[m].mean():>11.3f} "
              f"{np.abs(pred[m] - yf[m]).mean():>8.3f}")
    print(f"  span {comp['span']:.3f}   slope {comp['slope']:.3f}   "
          f"(the labels span 4.0, a calibrated probe has slope 1.0)")

    print(f"\n=== per-perturbation deltas ({label}, within-clip, "
          f"{N_BOOT} bootstrap resamples) ===")
    stats = {}
    for vname in sorted(variants):
        X, mask = variants[vname]
        if not mask.any():
            continue
        pv = predict(model, X, device)
        d = pv[mask] - pred[mask]
        cos = _cos_features(clean[mask], X[mask])
        stats[vname] = {
            "kind": variant_kind(vname), "n": int(mask.sum()),
            "compression": compression(pv[mask], y[mask]),
            "signed": float(d.mean()), "abs": float(np.abs(d).mean()),
            "median_abs": float(np.median(np.abs(d))),
            "p95_abs": float(np.percentile(np.abs(d), 95)),
            "frac_over_half": float((np.abs(d) > 0.5).mean()),
            "cos": float(cos.mean()), "cos_min": float(cos.min()),
            "signed_ci": _boot_ci(d, N_BOOT),
            "abs_ci": _boot_ci(np.abs(d), N_BOOT),
            "cos_ci": _boot_ci(cos, N_BOOT),
        }
    print_variant_table(stats)

    if stats:
        print(f"\n=== compression by perturbation ({label}) ===")
        print_compression_table(comp, stats)

        print(f"\n  {'perturbation':<40} {'med|d|':>8} {'p95|d|':>8} "
              f"{'>0.5':>6} {'mean|d| 95% CI':>20} {'min cos':>8}")
        for vname, r in sorted(stats.items(),
                               key=lambda kv: (kv[1]["kind"] != "temporal",
                                               kv[0])):
            lo, hi = r["abs_ci"]
            print(f"  {vname:<40} {r['median_abs']:>8.4f} {r['p95_abs']:>8.4f} "
                  f"{r['frac_over_half']:>6.1%} "
                  f"{f'[{lo:.4f}, {hi:.4f}]':>20} {r['cos_min']:>8.4f}")

        sup = [r["abs"] for r in stats.values() if r["kind"] == "superficial"]
        tmp = [r["signed"] for r in stats.values() if r["kind"] == "temporal"]
        if sup:
            print(f"\n  mean|d| over expected-invariance perturbations "
                  f"{np.mean(sup):.4f}")
            worst = max((r for r in stats.values()
                         if r["kind"] == "superficial"),
                        key=lambda r: r["signed"])
            name = next(k for k, v in stats.items() if v is worst)
            print(f"  most inflating superficial cue: {name} "
                  f"{worst['signed']:+.4f}")
            print("  a positive signed shift means the cue inflates the "
                  "predicted score;\n  that is the gameability failure, and it "
                  "is worse than an equal-sized\n  symmetric wobble.")
        if tmp:
            print(f"\n  mean signed d over expected-sensitivity perturbations "
                  f"{np.mean(tmp):+.4f}")
            print("  these SHOULD be negative: a temporal manipulation breaks "
                  "the physics, so a\n  correct judge lowers its score. Near "
                  "zero (or positive) is temporal blindness.")
        print("\n  'cos' is the mean cosine between the clean and variant "
              "V-JEPA features,\n  per moment then averaged. It bounds what "
              "the probe could possibly do: a\n  perturbation the encoder "
              "barely registers cannot move the head much.")

    print(f"\n=== diagnostics ===")
    print(f"  threshold monotonicity violations  "
          f"{monotonicity(model, clean, device):.1%}")
    att = attention_stats(model, clean, device)
    if att:
        print(f"  attention entropy                  {att['mean']:.4f} nats "
              f"of a uniform {att['uniform']:.4f}")
        print(f"  normalized (H / log T)             {att['normalized']:.4f}  "
              + ("<- 1.00 is mean pooling; the attentive pooling has collapsed"
                 if att["normalized"] > 0.995 else ""))
        print(f"  mean max attention weight          {att['max_weight']:.4f} "
              f"vs uniform {att['uniform_weight']:.4f}")

    # The calibration protocol was never specified by the project lead. Split
    # conformal on absolute residuals is the standard default; confirm before
    # reporting these numbers as the project's calibration result.
    if "cal" in packs:
        cal_clean, cal_y, _, _ = build_eval(packs["cal"])
        cal_pred = predict(model, cal_clean, device)
        res = np.sort(np.abs(cal_pred - cal_y.astype(np.float64)))
        n = len(res)
        k = int(np.ceil((n + 1) * (1 - CONFORMAL_ALPHA)))
        q = res[min(k, n) - 1]
        # coverage has to be measured somewhere cal is not, or the quantile is
        # being scored against the residuals that defined it
        cover_on = tuple(sp for sp in splits if sp != "cal") or None
        print(f"\n=== calibration (cal, {n} clean clips) ===")
        print(f"  {1-CONFORMAL_ALPHA:.0%} split-conformal half-width  +/-{q:.3f}")
        if cover_on:
            cc, cy, _, _ = build_eval(
                concat_packs([packs[sp] for sp in cover_on]))
            cp = predict(model, cc, device)
            cover = float((np.abs(cp - cy.astype(np.float64)) <= q).mean())
            print(f"  empirical coverage on {'+'.join(cover_on):<19} "
                  f"{cover:.1%}")
        else:
            print("  no coverage number: every reporting split is the "
                  "calibration split")
        if "cal" in splits:
            print("  NOTE cal is inside the reporting set above, so the "
                  "pooled metrics are\n  not independent of this "
                  "quantile. The per-split table is the clean read.")
        print("  ASSUMPTION: split conformal on |residual|. The calibration "
              "protocol\n  was not specified -- confirm before quoting this.")

    # Compact and rounded on purpose: a notebook displays whatever the last
    # expression returns, and raw numpy scalars drown the tables above.
    return {"splits": list(splits), "n": int(len(y)),
            "mae": round(float(mae), 4),
            "macro_mae": round(float(macro_mae), 4),
            "rho": round(float(rho), 4), "rmse": round(float(rmse), 4),
            "signed": {k: round(v["signed"], 4) for k, v in stats.items()},
            "cos": {k: round(v["cos"], 4) for k, v in stats.items()}}


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(description="test the trained PC probe")
    ap.add_argument("--probe", default="probe_v1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="unit-check the ordinal machinery and exit")
    ap.add_argument("--splits", nargs="+", default=list(HELD_OUT),
                    choices=list(HELD_OUT),
                    help="reporting set, pooled (default: every held-out "
                         "split; pass 'val' for the old val-only report)")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    report(name=a.probe, device=a.device, splits=tuple(a.splits))
