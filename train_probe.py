import argparse
import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np

BUCKET = "nickb-aarj"
HUB_MODEL = "vjepa2_1_vit_large_384"
PROBE_PREFIX = "probes"

EMBED_DIM = 1024
N_TEMPORAL = 32       # temporal positions kept by embed_vjepa.pool_tokens
N_THRESH = 4          # PC>1, PC>2, PC>3, PC>4
N_PERT = 2            # perturbations per training clip, fixed in the split
PC_LEVELS = (1, 2, 3, 4, 5)

# val-only; never trained on, evaluated by within-clip delta
TEMPORAL_VARIANTS = ("shuffle", "reverse", "freeze")


def variant_kind(name):
    return "temporal" if name in TEMPORAL_VARIANTS else "superficial"


# both caches are live: the mean-pooled [1024] vectors are the baseline the
# attentive probe gets compared against, on the same splits and the same code
PACK_PREFIXES = {
    "mean": f"embeddings/{HUB_MODEL}/packs",
    "t32": f"embeddings/{HUB_MODEL}/packs_t{N_TEMPORAL}",
}
PACK_KIND = "t32"     # module level so eval_probe/diagnose_probe follow along

DEFAULTS = dict(
    hidden=256,
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    epochs=400,         # a cap; early stopping normally ends the run
    patience=15,        # evals without an improvement before stopping
    alpha=1.0,          # weight on the perturbed PC loss
    lambda_cons=1.0,    # weight on the consistency penalty
    consistency="kl",   # kl | mse
    arch=None,          # mlp | attn; inferred from the pack rank when None
    head="linear",      # linear | coral | corn
    logit_adjust=False,
    class_weight="inverse",  # none | inverse | sqrt_inverse
    select="macro_mae",  # mae | macro_mae | rho
    n_boot=2000,        # bootstrap resamples for the per-variant CIs
    seed=0,
    eval_every=10,
)

try:
    from google.colab import userdata
    os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
except Exception:
    pass
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


s3 = None


def _ensure_s3():
    global s3
    if s3 is None:
        s3 = boto3.client("s3")
    return s3


# ------------------------------------------------------------------ data

def load_pack(split, kind=None):
    key = f"{PACK_PREFIXES[kind or PACK_KIND]}/{split}.npz"
    body = _ensure_s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    with np.load(io.BytesIO(body)) as z:
        pack = {k: z[k] for k in z.files}
    shape = " x ".join(str(d) for d in pack["X"].shape[1:])
    print(f"  {split}: {len(pack['X'])} rows x {shape}")
    return pack


def group_by_stem(pack):
    """-> {stem: {variant: row index}}"""
    out = defaultdict(dict)
    for i, (stem, variant) in enumerate(zip(pack["stem"], pack["variant"])):
        out[str(stem)][str(variant)] = i
    return out


def build_train(pack, n_pert=N_PERT):
    """-> X_clean (N,*F), X_pert (N,n_pert,*F), y (N,), stems, where *F is
    (1024,) for the mean-pooled packs and (32,1024) for the temporal ones."""
    grouped = group_by_stem(pack)
    clean_i, pert_i, y, stems, skipped = [], [], [], [], 0
    for stem in sorted(grouped):
        d = grouped[stem]
        perts = sorted(k for k in d if k != "clean")
        if "clean" not in d or len(perts) < n_pert:
            skipped += 1
            continue
        clean_i.append(d["clean"])
        pert_i.append([d[p] for p in perts[:n_pert]])
        y.append(int(pack["pc"][d["clean"]]))
        stems.append(stem)

    X = pack["X"].astype(np.float32)
    if skipped:
        print(f"  skipped {skipped} train clips missing clean or a perturbation")
    return (X[np.array(clean_i)],
            X[np.array(pert_i)],
            np.array(y, dtype=np.int64),
            stems)


def build_eval(pack):
    """-> X_clean, y, {variant: (X, mask)}, stems. Rows stay aligned to their
    clean clip so deltas are within-clip."""
    grouped = group_by_stem(pack)
    stems = sorted(s for s in grouped if "clean" in grouped[s])
    X = pack["X"].astype(np.float32)

    clean = X[np.array([grouped[s]["clean"] for s in stems])]
    y = np.array([int(pack["pc"][grouped[s]["clean"]]) for s in stems],
                 dtype=np.int64)

    names = sorted({v for s in stems for v in grouped[s] if v != "clean"})
    variants = {}
    for name in names:
        rows = np.zeros((len(stems),) + X.shape[1:], dtype=np.float32)
        mask = np.zeros(len(stems), dtype=bool)
        for j, s in enumerate(stems):
            if name in grouped[s]:
                rows[j] = X[grouped[s][name]]
                mask[j] = True
        variants[name] = (rows, mask)
    return clean, y, variants, stems


# ----------------------------------------------------------------- model

_ATTN_CLS = None


def _attentive_cls():
    """Built on first use so importing this module still costs no torch."""
    global _ATTN_CLS
    if _ATTN_CLS is not None:
        return _ATTN_CLS

    import torch
    import torch.nn as nn

    class OrdinalHead(nn.Module):
        """4 cumulative logits, P(PC>1)..P(PC>4).

        linear: unconstrained. coral: one score plus a per-threshold bias.
        corn: conditional logits P(PC>k | PC>k-1), cumulative via running product.
        """

        def __init__(self, d, kind="linear"):
            super().__init__()
            self.kind = kind
            if kind == "coral":
                self.score = nn.Linear(d, 1, bias=False)
                self.bias = nn.Parameter(torch.zeros(N_THRESH))
            else:
                self.fc = nn.Linear(d, N_THRESH)

        def forward(self, h):
            if self.kind == "coral":
                return self.score(h) + self.bias
            return self.fc(h)

    class AttentiveProbe(nn.Module):
        """LayerNorm -> 1024->256 -> learned temporal positions -> single-query
        attentive pooling over the 32 moments -> ordinal head."""

        def __init__(self, cfg):
            super().__init__()
            d = cfg["hidden"]
            t = cfg.get("n_temporal") or N_TEMPORAL
            self.norm = nn.LayerNorm(EMBED_DIM)
            self.proj = nn.Linear(EMBED_DIM, d)
            self.act = nn.GELU()
            self.pos = nn.Parameter(torch.zeros(t, d))
            self.query = nn.Parameter(torch.zeros(d))
            self.drop = nn.Dropout(cfg["dropout"])
            self.head = OrdinalHead(d, cfg.get("head", "linear"))
            # a smaller scale/std flattens the softmax toward mean pooling
            self.scale = 1.0
            nn.init.normal_(self.pos, std=0.1)
            nn.init.normal_(self.query, std=0.2)
            self._adjust = None

        def set_logit_adjust(self, vec):
            """Subtract log(pos_weight) at prediction time. Eval only."""
            self._adjust = vec

        def _tokens(self, x):
            if x.ndim != 3:
                raise RuntimeError(f"expected (B, T, D) input, got "
                                   f"{tuple(x.shape)}; the attentive probe "
                                   f"needs the temporal packs")
            return self.act(self.proj(self.norm(x))) + self.pos

        def attention(self, x):
            """(B, T) softmax weights over the moments, for diagnostics."""
            return ((self._tokens(x) @ self.query) * self.scale).softmax(-1)

        def attention_entropy(self, x):
            """(B,) entropy in nats. Uniform is log(T) = 3.466 at T=32."""
            a = self.attention(x)
            return -(a * a.clamp_min(1e-9).log()).sum(-1)

        def raw_logits(self, x):
            h = self._tokens(x)
            attn = ((h @ self.query) * self.scale).softmax(-1)
            pooled = (attn.unsqueeze(-1) * h).sum(dim=1)
            return self.head(self.drop(pooled))

        def forward(self, x):
            z = self.raw_logits(x)
            if not self.training and self._adjust is not None:
                z = z - self._adjust.to(z.device)
            if self.head.kind == "corn":
                p = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6).cumprod(dim=-1)
                z = torch.log(p) - torch.log1p(-p)
            return z

    _ATTN_CLS = AttentiveProbe
    return _ATTN_CLS


def make_model(cfg):
    """Defaults to 'mlp' when arch is unset, so probes saved before the
    temporal rebuild still load."""
    import torch.nn as nn

    if (cfg.get("arch") or "mlp") == "attn":
        model = _attentive_cls()(cfg)
        adj = cfg.get("logit_adjust_values")
        if adj:
            import torch
            model.set_logit_adjust(torch.tensor(adj, dtype=torch.float32))
        return model

    if cfg.get("head") == "corn":
        raise ValueError("head='corn' needs the attentive probe; the "
                         "mean-pooled mlp path does not implement it")
    if cfg.get("logit_adjust_values"):
        raise ValueError("logit adjustment is implemented on the attentive "
                         "probe only")

    return nn.Sequential(
        nn.LayerNorm(EMBED_DIM),
        nn.Linear(EMBED_DIM, cfg["hidden"]),
        nn.GELU(),
        nn.Dropout(cfg["dropout"]),
        nn.Linear(cfg["hidden"], N_THRESH),
    )


def ordinal_targets(pc):
    """pc in 1..5 -> (N,4) with t_k = 1 iff pc > k. PC=3 -> [1,1,0,0]."""
    import torch
    ks = torch.arange(1, N_THRESH + 1, device=pc.device, dtype=pc.dtype)
    return (pc.unsqueeze(-1) > ks).float()


def expected_pc(logits):
    """E[PC] = 1 + sum_k P(PC > k)."""
    import torch
    return 1.0 + torch.sigmoid(logits).sum(-1)


def threshold_pos_weight(y):
    """neg/pos per threshold. PC=1 is 3.8% of the data, so PC>1 comes out
    ~25:1 positive and PC>4 ~5:1 negative."""
    import torch
    t = ordinal_targets(y.float())
    pos = t.sum(0)
    neg = t.shape[0] - pos
    return neg / pos.clamp(min=1.0)


def clip_weights(y, kind="none"):
    """Per-clip loss weight from PC frequency, normalised to mean 1. Attaches
    to the clip so its clean and perturbed rows share one weight."""
    import torch
    counts = torch.bincount(y, minlength=max(PC_LEVELS) + 1).float()
    per = counts[y].clamp(min=1.0)
    if kind == "inverse":
        w = 1.0 / per
    elif kind == "sqrt_inverse":
        w = 1.0 / per.sqrt()
    elif kind == "none":
        w = torch.ones_like(per)
    else:
        raise ValueError(f"unknown class_weight {kind!r}")
    return w / w.mean()


def corn_pos_weight(y):
    """neg/pos per threshold on corn's conditional subsets, which are far more
    balanced than the full set."""
    import torch
    t = ordinal_targets(y.float())
    out = []
    for k in range(t.shape[1]):
        mask = (torch.ones(len(t), dtype=torch.bool, device=t.device)
                if k == 0 else t[:, k - 1] > 0.5)
        sub = t[mask, k]
        pos = sub.sum()
        neg = mask.sum() - pos
        out.append((neg / pos.clamp(min=1.0)) if pos > 0 else
                   torch.tensor(1.0, device=t.device))
    return torch.stack([o if torch.is_tensor(o) else
                        torch.tensor(o, device=t.device) for o in out])


def corn_loss(logits, targets, pos_weight, w):
    """Threshold k is trained only on the clips that cleared k-1, which is what
    makes the running product a valid cumulative probability."""
    import torch
    import torch.nn.functional as F

    total = torch.zeros((), device=logits.device)
    denom = torch.zeros((), device=logits.device)
    for k in range(targets.shape[1]):
        mask = (torch.ones(len(targets), dtype=torch.bool,
                           device=targets.device)
                if k == 0 else targets[:, k - 1] > 0.5)
        if not bool(mask.any()):
            continue
        ww = w[mask]
        bce = F.binary_cross_entropy_with_logits(
            logits[mask, k], targets[mask, k],
            pos_weight=pos_weight[k], reduction="none")
        total = total + (bce * ww).sum()
        denom = denom + ww.sum()
    return total / denom.clamp(min=1e-8)


def weighted_bce(bce_none, logits, targets, w):
    return (bce_none(logits, targets).mean(dim=1) * w).sum() / w.sum().clamp(min=1e-8)


def set_dropout(model, active):
    """Toggle only the Dropout layers, leaving the rest in train mode. The
    consistency term needs dropout off or it measures the mask difference."""
    import torch.nn as nn
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train(active)


def consistency_loss(logits_a, logits_b, kind="mse"):
    """Distance between the clean and perturbed threshold probabilities.
    Symmetric and not detached, so neither side chases the other."""
    import torch
    pa, pb = torch.sigmoid(logits_a), torch.sigmoid(logits_b)
    if kind == "mse":
        return ((pa - pb) ** 2).mean()
    eps = 1e-6
    pa = pa.clamp(eps, 1 - eps)
    pb = pb.clamp(eps, 1 - eps)

    def kl(p, q):
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))

    return 0.5 * (kl(pa, pb) + kl(pb, pa)).mean()


# ----------------------------------------------------------------- eval

def _rankdata(a):
    """Midranks, so tied values share a rank. PC is an integer 1..5, so the
    ties are large and argsort(argsort(x)) would break them arbitrarily."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # 1-based midrank
        i = j + 1
    return ranks


def _spearman(a, b):
    """Tie-corrected rho. Underscored so eval_probe.spearman cannot shadow it
    when both files are pasted into one notebook."""
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")   # a constant prediction has no rank correlation
    return float(np.corrcoef(ra, rb)[0, 1])


def _boot_ci(x, n_boot=2000, alpha=0.05, seed=0, stat=np.mean):
    """Percentile bootstrap CI, resampling clips. Deltas are within-clip paired
    differences, so the clip is the independent unit."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(int(n_boot), len(x)))
    draws = stat(x[idx], axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _cos_features(a, b):
    """Per-clip cosine between clean and variant features. For the [32,1024]
    packs this is the mean of the 32 per-moment cosines: moment t against
    moment t, so a reorder in time shows up."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 2:            # (N, D) mean-pooled packs
        a, b = a[:, None, :], b[:, None, :]
    num = (a * b).sum(-1)
    den = (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)) + 1e-8
    return (num / den).mean(-1)


def attention_stats(model, X, device, batch=1024):
    """-> {mean, uniform, normalized, max_weight} or None for the mlp probe.
    normalized is entropy / log(T); 1.00 is mean pooling."""
    import torch
    if not hasattr(model, "attention_entropy"):
        return None
    ents, maxes = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            chunk = torch.as_tensor(X[i:i + batch], device=device)
            ents.append(model.attention_entropy(chunk).cpu().numpy())
            maxes.append(model.attention(chunk).max(dim=-1).values.cpu().numpy())
    ent = float(np.concatenate(ents).mean())
    t = int(X.shape[1])
    uniform = float(np.log(t))
    return {"mean": ent, "uniform": uniform, "normalized": ent / uniform,
            "max_weight": float(np.concatenate(maxes).mean()),
            "uniform_weight": 1.0 / t}


def evaluate(model, clean, y, variants, device, batch=1024, n_boot=0, seed=0,
             attn=True):
    """-> dict with clean MAE / macro MAE / rho and per-variant delta stats.
    Signed and absolute deltas both; n_boot>0 adds percentile CIs."""
    import torch

    model.eval()
    with torch.no_grad():
        def score(x):
            out = []
            for i in range(0, len(x), batch):
                chunk = torch.as_tensor(x[i:i + batch], device=device)
                out.append(expected_pc(model(chunk)))
            return torch.cat(out)

        pred_clean = score(clean)
        yt = torch.as_tensor(y, device=device, dtype=torch.float32)
        mae = (pred_clean - yt).abs().mean().item()
        rmse = ((pred_clean - yt) ** 2).mean().sqrt().item()

        # plain MAE is dominated by PC 3-4, which is 65% of the clips, so
        # macro_mae averages the per-level MAEs instead
        per_level = [(pred_clean - yt).abs()[yt == lev].mean().item()
                     for lev in PC_LEVELS if bool((yt == lev).any())]
        macro_mae = float(np.mean(per_level)) if per_level else float("nan")
        rho = _spearman(pred_clean.cpu().numpy(), np.asarray(y, dtype=float))

        base = pred_clean.cpu().numpy()
        gaps, signed, stats = {}, {}, {}
        for name, (rows, mask) in variants.items():
            if not mask.any():
                continue
            d = (score(rows).cpu().numpy() - base)[mask]
            cos = _cos_features(clean[mask], rows[mask])
            gaps[name] = float(np.abs(d).mean())
            signed[name] = float(d.mean())
            row = {"kind": variant_kind(name), "n": int(mask.sum()),
                   "signed": signed[name], "abs": gaps[name],
                   "median_abs": float(np.median(np.abs(d))),
                   "p95_abs": float(np.percentile(np.abs(d), 95)),
                   "frac_over_half": float((np.abs(d) > 0.5).mean()),
                   "cos": float(cos.mean()), "cos_min": float(cos.min())}
            if n_boot:
                row["signed_ci"] = _boot_ci(d, n_boot, seed=seed)
                row["abs_ci"] = _boot_ci(np.abs(d), n_boot, seed=seed)
                row["cos_ci"] = _boot_ci(cos, n_boot, seed=seed)
            stats[name] = row

        att = attention_stats(model, clean, device, batch) if attn else None

    model.train()

    # mean_gap is the invariance measure, so temporal variants stay out of it
    sup = [g for n, g in gaps.items() if variant_kind(n) == "superficial"]
    tmp = [signed[n] for n in gaps if variant_kind(n) == "temporal"]
    return {"mae": mae, "rmse": rmse, "macro_mae": macro_mae, "rho": rho,
            "gaps": gaps, "signed": signed, "variant_stats": stats,
            "attention": att,
            "mean_gap": float(np.mean(sup)) if sup else 0.0,
            "mean_temporal_signed": float(np.mean(tmp)) if tmp else float("nan"),
            "pred_clean": base}


def print_variant_table(stats, indent="  "):
    """Per-variant report grouped by taxonomy half. Shared by train() and
    eval_probe.report()."""
    if not stats:
        return
    order = {"temporal": 0, "superficial": 1}
    groups = defaultdict(list)
    for name, r in stats.items():
        groups[r["kind"]].append((name, r))

    print(f"{indent}{'perturbation':<40} {'n':>5} {'signed d':>9} "
          f"{'95% CI':>18} {'mean|d|':>8} {'cos':>7}")
    for kind in sorted(groups, key=lambda k: order.get(k, 9)):
        want = ("scores should DROP: signed d < 0" if kind == "temporal"
                else "scores should NOT MOVE: signed d ~ 0, and > 0 is inflation")
        label = "sensitivity" if kind == "temporal" else "invariance"
        print(f"{indent}-- expected-{label} ({want})")
        for name, r in sorted(groups[kind]):
            ci = r.get("signed_ci")
            ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
            print(f"{indent}{name:<40} {r['n']:>5} {r['signed']:>+9.4f} "
                  f"{ci_s:>18} {r['abs']:>8.4f} {r['cos']:>7.4f}")
        if kind == "temporal":
            n_ok = sum(1 for _, r in groups[kind]
                       if r.get("signed_ci") and r["signed_ci"][1] < 0)
            if any(r.get("signed_ci") for _, r in groups[kind]):
                print(f"{indent}   {n_ok}/{len(groups[kind])} temporal "
                      f"perturbations have a CI entirely below 0")
        else:
            bad = [n for n, r in groups[kind]
                   if r.get("signed_ci") and r["signed_ci"][0] > 0]
            if bad:
                print(f"{indent}   INFLATION: CI entirely above 0 for "
                      + ", ".join(sorted(bad)))


def print_headline(m, indent="  ", label=""):
    """MAE, macro MAE and rho side by side, whichever one is selecting."""
    print(f"{indent}{label}mae {m['mae']:.4f}   macro_mae {m['macro_mae']:.4f} "
          f"  rho {m['rho']:+.4f}")


# ---------------------------------------------------------------- train

def train(device=None, verbose=True, packs=None, **overrides):
    import torch

    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if packs is None:
        print("loading packs ...")
        packs = {s: load_pack(s) for s in ("train", "val")}

    Xc, Xp, y, _ = build_train(packs["train"])
    ev_clean, ev_y, ev_variants, _ = build_eval(packs["val"])

    # the pack is self-describing: rank 2 is mean-pooled, rank 3 temporal
    if cfg["arch"] is None:
        cfg["arch"] = "attn" if Xc.ndim == 3 else "mlp"
    if cfg["arch"] == "attn":
        if Xc.ndim != 3:
            raise ValueError("arch='attn' needs the [32,1024] packs; got "
                             f"features of shape {Xc.shape[1:]}")
        cfg["n_temporal"] = int(Xc.shape[1])
    elif Xc.ndim != 2:
        raise ValueError("arch='mlp' needs the mean-pooled packs; got features "
                         f"of shape {Xc.shape[1:]}")

    if verbose:
        counts = {p: int((y == p).sum()) for p in PC_LEVELS}
        print(f"  train {len(y)} clips {counts}")
        print(f"  val   {len(ev_y)} clips, {len(ev_variants)} perturbation types")
        kinds = defaultdict(list)
        for v in sorted(ev_variants):
            kinds[variant_kind(v)].append(v)
        for k in ("temporal", "superficial"):
            if kinds[k]:
                label = "sensitivity" if k == "temporal" else "invariance"
                print(f"    expected-{label:<11} {len(kinds[k])}: "
                      f"{', '.join(kinds[k])}")

    Xc = torch.as_tensor(Xc, device=device)
    Xp = torch.as_tensor(Xp, device=device)
    yt = torch.as_tensor(y, device=device)

    model = make_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    corn = cfg["head"] == "corn"
    pw = (corn_pos_weight(yt) if corn else threshold_pos_weight(yt)).to(device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    targets = ordinal_targets(yt.float())

    if cfg["logit_adjust"]:
        cfg["logit_adjust_values"] = torch.log(pw).tolist()
        model.set_logit_adjust(torch.log(pw).to(device))

    def pc_loss(x, tgt, weights):
        if corn:
            return corn_loss(model.raw_logits(x), tgt, pw, weights)
        return weighted_bce(bce, model(x), tgt, weights)
    cw = clip_weights(yt, cfg["class_weight"]).to(device)

    if verbose:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"  arch={cfg['arch']} head={cfg['head']} "
              f"hidden={cfg['hidden']} params={n_par/1e3:.0f}K")
        print(f"  pos_weight per threshold"
              f"{' (conditional)' if corn else ''}: "
              f"{', '.join(f'{v:.3f}' for v in pw.tolist())}")
        if cfg["logit_adjust"]:
            print(f"  logit adjustment (eval only): "
                  f"{', '.join(f'{v:+.3f}' for v in cfg['logit_adjust_values'])}")
        if cfg["class_weight"] != "none":
            per = {p: round(float(cw[yt == p][0]), 3)
                   for p in PC_LEVELS if (yt == p).any()}
            print(f"  clip weight ({cfg['class_weight']}) by pc: {per}")
        print(f"  device={device}  lambda_cons={cfg['lambda_cons']}  "
              f"alpha={cfg['alpha']}  consistency={cfg['consistency']}")
        print(f"  early stopping on val {cfg['select']}: patience "
              f"{cfg['patience']} evals "
              f"({cfg['patience'] * cfg['eval_every']} epochs), "
              f"cap {cfg['epochs']}")

    if cfg["select"] not in ("mae", "macro_mae", "rho"):
        raise ValueError(f"unknown select {cfg['select']!r}")

    n = len(yt)
    best = {"score": float("inf")}
    history = []
    stale = 0
    stopped = None
    t0 = time.perf_counter()

    for epoch in range(1, cfg["epochs"] + 1):
        perm = torch.randperm(n, device=device)
        totals = np.zeros(3)
        nb = 0
        for i in range(0, n, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            tgt = targets[idx]
            w = cw[idx]

            loss_c = pc_loss(Xc[idx], tgt, w)
            loss_p = torch.zeros((), device=device)
            for j in range(Xp.shape[1]):
                loss_p = loss_p + pc_loss(Xp[idx, j], tgt, w)
            loss_p = loss_p / Xp.shape[1]

            # dropout off here so this measures the perturbation rather than
            # two different dropout draws
            loss_k = torch.zeros((), device=device)
            set_dropout(model, False)
            clean_k = model(Xc[idx])
            for j in range(Xp.shape[1]):
                loss_k = loss_k + consistency_loss(clean_k, model(Xp[idx, j]),
                                                   cfg["consistency"])
            loss_k = loss_k / Xp.shape[1]
            set_dropout(model, True)

            loss = loss_c + cfg["alpha"] * loss_p + cfg["lambda_cons"] * loss_k
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            totals += [loss_c.item(), loss_p.item(), loss_k.item()]
            nb += 1

        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            m = evaluate(model, ev_clean, ev_y, ev_variants, device)
            # train MAE separates underfitting from a weak signal: if it tracks
            # val, more capacity can help; if it is far below, it cannot
            tr = evaluate(model, Xc, y, {}, device, attn=False)
            spread = float(m["pred_clean"].max() - m["pred_clean"].min())
            att = m["attention"]
            row = {"epoch": epoch, "loss_clean": totals[0] / nb,
                   "loss_pert": totals[1] / nb, "loss_cons": totals[2] / nb,
                   "train_mae": tr["mae"], "val_mae": m["mae"],
                   "val_macro_mae": m["macro_mae"], "val_rho": m["rho"],
                   "val_gap": m["mean_gap"],
                   "val_temporal_signed": m["mean_temporal_signed"],
                   "val_spread": spread,
                   "attn_entropy": att["mean"] if att else None,
                   "attn_entropy_norm": att["normalized"] if att else None}
            history.append(row)
            if verbose:
                ent = (f"  H {att['mean']:.3f}/{att['uniform']:.3f} "
                       f"({att['normalized']:.3f})" if att else "")
                print(f"  epoch {epoch:>4}  clean {row['loss_clean']:.4f}  "
                      f"cons {row['loss_cons']:.5f}   train mae "
                      f"{tr['mae']:.4f}  val mae {m['mae']:.4f}  "
                      f"macro {m['macro_mae']:.4f}  rho {m['rho']:+.3f}  "
                      f"gap {m['mean_gap']:.4f}  spread {spread:.2f}{ent}")
            # rho negated so one "lower is better" comparison covers all three
            score = -m["rho"] if cfg["select"] == "rho" else m[cfg["select"]]
            if score < best["score"]:
                best = {"score": score, "mae": m["mae"], "rmse": m["rmse"],
                        "macro_mae": m["macro_mae"], "rho": m["rho"],
                        "mean_gap": m["mean_gap"], "gaps": m["gaps"],
                        "signed": m["signed"],
                        "mean_temporal_signed": m["mean_temporal_signed"],
                        "attention": att, "epoch": epoch,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    stopped = epoch
                    if verbose:
                        print(f"  early stop at epoch {epoch}: no val "
                              f"{cfg['select']} improvement in {stale} evals "
                              f"(best epoch {best['epoch']})")
                    break

    secs = time.perf_counter() - t0
    ran = stopped or cfg["epochs"]

    # rho is NaN on constant predictions and never beats inf, so best can be unset
    unselected = "state" not in best
    if unselected:
        print(f"  WARNING no eval ever improved on the initial score (val "
              f"{cfg['select']} was NaN, or the run was too short); keeping "
              f"the last epoch's weights, which are NOT early-stopped")
        best = {"score": float("nan"), "epoch": ran,
                "state": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}}

    # CIs once, on the selected checkpoint
    model.load_state_dict(best["state"])
    final = evaluate(model, ev_clean, ev_y, ev_variants, device,
                     n_boot=cfg["n_boot"], seed=cfg["seed"])
    best["variant_stats"] = final["variant_stats"]
    if unselected:
        best.update({k: final[k] for k in
                     ("mae", "rmse", "macro_mae", "rho", "mean_gap", "gaps",
                      "signed", "mean_temporal_signed", "attention")})

    if verbose:
        print(f"\n  {ran} epochs in {secs:.1f}s ({secs/ran*1000:.0f} ms/epoch)"
              + ("" if stopped else "  [hit the epoch cap, not early stopping]"))
        print(f"  best epoch {best['epoch']} (selected on val {cfg['select']})")
        print_headline(best, indent="    ", label="val ")
        print(f"    mean invariance gap (superficial only) "
              f"{best['mean_gap']:.4f}")
        att = best.get("attention")
        if att:
            print(f"    attention entropy {att['mean']:.3f} nats of a uniform "
                  f"{att['uniform']:.3f} ({att['normalized']:.3f}); mean max "
                  f"weight {att['max_weight']:.4f} vs uniform "
                  f"{att['uniform_weight']:.4f}")
        print()
        print_variant_table(best["variant_stats"], indent="    ")

    return {"cfg": cfg, "best": best, "history": history, "epochs_run": ran,
            "early_stopped": stopped is not None,
            "seconds": secs, "device": device}


def save_probe(result, name="probe_v1", push_to_s3=True):
    import torch

    payload = {
        "state_dict": result["best"]["state"],
        "cfg": result["cfg"],
        "val": {k: result["best"][k] for k in
                ("mae", "rmse", "macro_mae", "rho", "mean_gap", "gaps",
                 "signed", "mean_temporal_signed", "variant_stats",
                 "attention", "epoch") if k in result["best"]},
        "history": result["history"],
        "embed_dim": EMBED_DIM,
        "n_thresh": N_THRESH,
    }
    buf = io.BytesIO()
    torch.save(payload, buf)

    local = Path(f"./{name}.pt")
    local.write_bytes(buf.getvalue())
    print(f"saved -> {local}")

    if push_to_s3:
        key = f"{PROBE_PREFIX}/{name}.pt"
        _ensure_s3().put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        print(f"uploaded -> s3://{BUCKET}/{key}")
    return payload


def push_probe(name="probe_v1"):
    """Upload a probe already saved locally."""
    local = Path(f"./{name}.pt")
    if not local.exists():
        raise FileNotFoundError(f"{local} not found; run save_probe first")
    key = f"{PROBE_PREFIX}/{name}.pt"
    _ensure_s3().put_object(Bucket=BUCKET, Key=key, Body=local.read_bytes())
    print(f"uploaded {local} -> s3://{BUCKET}/{key}")
    return key


def sweep(lambdas=(0.0, 10.0, 100.0, 1000.0), device=None, **overrides):
    """One training per lambda on shared packs."""
    print("loading packs ...")
    packs = {s: load_pack(s) for s in ("train", "val")}

    runs = []
    for lam in lambdas:
        print(f"\n=== lambda_cons={lam} ===")
        r = train(device=device, verbose=False, packs=packs,
                  lambda_cons=lam, **overrides)
        runs.append(r)
        att = r["best"].get("attention")
        print(f"  val mae {r['best']['mae']:.4f}   "
              f"macro {r['best']['macro_mae']:.4f}   "
              f"rho {r['best']['rho']:+.3f}   "
              f"mean gap {r['best']['mean_gap']:.4f}   "
              + (f"H {att['normalized']:.3f}   " if att else "")
              + f"({r['seconds']:.1f}s)")

    print(f"\n  {'lambda':>8} {'val mae':>9} {'macro':>9} {'rho':>7} "
          f"{'sup gap':>9} {'temporal d':>11} {'H/logT':>7} {'epoch':>6}")
    for lam, r in zip(lambdas, runs):
        b = r["best"]
        att = b.get("attention")
        print(f"  {lam:>8} {b['mae']:>9.4f} {b['macro_mae']:>9.4f} "
              f"{b['rho']:>+7.3f} {b['mean_gap']:>9.4f} "
              f"{b.get('mean_temporal_signed', float('nan')):>+11.4f} "
              + (f"{att['normalized']:>7.3f} " if att else f"{'-':>7} ")
              + f"{b['epoch']:>6}")
    return runs


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(description="train the ordinal PC probe")
    for k in ("hidden", "batch_size", "epochs", "patience", "seed",
              "eval_every", "n_boot"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    for k in ("dropout", "lr", "weight_decay", "alpha", "lambda_cons"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=None)
    ap.add_argument("--consistency", choices=["mse", "kl"], default=None)
    ap.add_argument("--arch", choices=["mlp", "attn"], default=None,
                    help="default: inferred from the pack rank")
    ap.add_argument("--head", choices=["linear", "coral", "corn"],
                    default=None)
    ap.add_argument("--logit-adjust", action="store_true", default=None,
                    help="undo the pos_weight shift at prediction time")
    ap.add_argument("--select", choices=["mae", "macro_mae", "rho"],
                    default=None, help="validation metric early stopping uses")
    ap.add_argument("--class-weight",
                    choices=["none", "inverse", "sqrt_inverse"], default=None,
                    help="per-clip PC-frequency weighting")
    ap.add_argument("--packs", choices=sorted(PACK_PREFIXES), default=PACK_KIND,
                    help="t32 = the [32,1024] cache, mean = the old baseline")
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="probe_v1")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--sweep", nargs="+", type=float, default=None,
                    help="lambda values to try instead of a single run")
    a = vars(ap.parse_args())

    lams, name, no_push = a.pop("sweep"), a.pop("name"), a.pop("no_push")
    dev = a.pop("device")
    PACK_KIND = a.pop("packs")
    if lams:
        sweep(lams, device=dev, **a)
    else:
        save_probe(train(device=dev, **a), name=name, push_to_s3=not no_push)
