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

# held-out only; never trained on, evaluated by within-clip delta
TEMPORAL_VARIANTS = ("shuffle", "reverse", "freeze")

# The held-out splits carry the full variant set. Selection stays on val alone
# -- cal must not touch early stopping -- but the reported numbers pool both,
# because val is 10% and cal is 10%, so a val-only table uses half the clips
# that exist.
HELD_OUT_SPLITS = ("val", "cal")


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
    arch=None,          # see ARCHS; inferred from the pack rank when None
    conv_kernel=3,      # diff_conv only
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


def concat_packs(packs):
    """Row-concatenate split packs so one eval can span them. Stems are unique
    across splits by construction (a stem lives in exactly one split), so
    group_by_stem still separates clips correctly."""
    packs = [p for p in packs if p is not None and len(p["X"])]
    if not packs:
        raise ValueError("no packs to concatenate")
    if len(packs) == 1:
        return packs[0]
    keys = set(packs[0])
    for p in packs[1:]:
        if set(p) != keys:
            raise ValueError("packs carry different keys: "
                             f"{sorted(keys)} vs {sorted(p)}")
        if p["X"].shape[1:] != packs[0]["X"].shape[1:]:
            raise ValueError("packs carry different feature shapes: "
                             f"{packs[0]['X'].shape[1:]} vs {p['X'].shape[1:]}")
    return {k: np.concatenate([p[k] for p in packs]) for k in keys}


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

_PROBE_CLASSES = None

# every arch below consumes the [32,1024] packs; "mlp" is the rank-2 baseline
TEMPORAL_ARCHS = ("attn", "mean_linear", "proj_mean", "diff_conv")
ARCHS = ("mlp",) + TEMPORAL_ARCHS


def _probe_classes():
    """arch -> class. Built on first use so importing this module stays
    torch-free."""
    global _PROBE_CLASSES
    if _PROBE_CLASSES is not None:
        return _PROBE_CLASSES

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

    class _HeadMixin:
        """Everything the four temporal probes share downstream of pooling:
        the eval-time logit adjustment and the corn conversion. Holds no
        parameters, so mixing it into AttentiveProbe leaves its state_dict
        keys -- and the mentor's init -- exactly as they were."""

        def set_logit_adjust(self, vec):
            """Subtract log(pos_weight) at prediction time. Eval only."""
            self._adjust = vec

        def _need_temporal(self, x):
            if x.ndim != 3:
                raise RuntimeError(f"expected (B, T, D) input, got "
                                   f"{tuple(x.shape)}; this probe needs the "
                                   f"temporal packs")
            return x

        def forward(self, x):
            z = self.raw_logits(x)
            if not self.training and self._adjust is not None:
                z = z - self._adjust.to(z.device)
            if self.head.kind == "corn":
                p = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6).cumprod(dim=-1)
                z = torch.log(p) - torch.log1p(-p)
            return z

    class AttentiveProbe(_HeadMixin, nn.Module):
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

    class MeanLinearProbe(_HeadMixin, nn.Module):
        """LayerNorm -> mean over the 32 moments -> linear ordinal head.

        No projection and no nonlinearity: the floor of the ablation. Time is
        averaged away before anything looks at it, so this probe is exactly
        order-blind by construction -- its temporal deltas are the null the
        other three are measured against."""

        def __init__(self, cfg):
            super().__init__()
            self.norm = nn.LayerNorm(EMBED_DIM)
            self.drop = nn.Dropout(cfg["dropout"])
            self.head = OrdinalHead(EMBED_DIM, cfg.get("head", "linear"))
            self._adjust = None

        def raw_logits(self, x):
            h = self.norm(self._need_temporal(x)).mean(dim=1)
            return self.head(self.drop(h))

    class ProjMeanProbe(_HeadMixin, nn.Module):
        """LayerNorm -> 1024->hidden -> GELU -> mean over the moments -> head.

        Identical to AttentiveProbe except the pooling is an unweighted mean,
        so the pair isolates exactly what the attentive pooling buys. No
        positional embedding: a mean is order-invariant, so adding pos before
        it would only contribute a constant bias."""

        def __init__(self, cfg):
            super().__init__()
            d = cfg["hidden"]
            self.norm = nn.LayerNorm(EMBED_DIM)
            self.proj = nn.Linear(EMBED_DIM, d)
            self.act = nn.GELU()
            self.drop = nn.Dropout(cfg["dropout"])
            self.head = OrdinalHead(d, cfg.get("head", "linear"))
            self._adjust = None

        def raw_logits(self, x):
            h = self.act(self.proj(self.norm(self._need_temporal(x))))
            return self.head(self.drop(h.mean(dim=1)))

    class DiffConvProbe(_HeadMixin, nn.Module):
        """LayerNorm -> 1024->hidden -> first difference along time -> Conv1D
        -> GELU -> mean over the 31 differences -> ordinal head.

        The first difference is the point: it cancels the static appearance and
        leaves only moment-to-moment change, so this is the one probe that can
        ONLY see temporal structure. The projection is linear and the
        nonlinearity comes after the conv, so the difference is taken in a
        space where diff and projection commute. If shuffle/reverse fail to
        move even this probe, the encoder is not carrying the order."""

        def __init__(self, cfg):
            super().__init__()
            d = cfg["hidden"]
            k = int(cfg.get("conv_kernel") or 3)
            self.norm = nn.LayerNorm(EMBED_DIM)
            self.proj = nn.Linear(EMBED_DIM, d)
            self.conv = nn.Conv1d(d, d, kernel_size=k, padding=k // 2)
            self.act = nn.GELU()
            self.drop = nn.Dropout(cfg["dropout"])
            self.head = OrdinalHead(d, cfg.get("head", "linear"))
            self._adjust = None

        def raw_logits(self, x):
            h = self.proj(self.norm(self._need_temporal(x)))   # (B, T, d)
            d1 = h[:, 1:] - h[:, :-1]                          # (B, T-1, d)
            c = self.act(self.conv(d1.transpose(1, 2)))        # (B, d, T-1)
            return self.head(self.drop(c.mean(dim=-1)))

    _PROBE_CLASSES = {"attn": AttentiveProbe,
                      "mean_linear": MeanLinearProbe,
                      "proj_mean": ProjMeanProbe,
                      "diff_conv": DiffConvProbe}
    return _PROBE_CLASSES


def _attentive_cls():
    """Kept as a name because CLAUDE.md documents it."""
    return _probe_classes()["attn"]


def make_model(cfg):
    """Defaults to 'mlp' when arch is unset, so probes saved before the
    temporal rebuild still load."""
    import torch.nn as nn

    arch = cfg.get("arch") or "mlp"
    classes = _probe_classes()
    if arch in classes:
        model = classes[arch](cfg)
        adj = cfg.get("logit_adjust_values")
        if adj:
            import torch
            model.set_logit_adjust(torch.tensor(adj, dtype=torch.float32))
        return model
    if arch != "mlp":
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHS}")

    if cfg.get("head") == "corn":
        raise ValueError("head='corn' needs one of the temporal probes; the "
                         "mean-pooled mlp path does not implement it")
    if cfg.get("logit_adjust_values"):
        raise ValueError("logit adjustment is implemented on the temporal "
                         "probes only")

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


def compression(pred, y):
    """Mean prediction at each human PC level, the span of those means, and the
    regression slope of the level means on the level.

    `slope` is fit on the five level means, NOT on the per-clip predictions, so
    it stays comparable with every compression number reported before
    2026-08-21. The two differ because the PC distribution is lopsided -- pc4
    has ~10x the clips of pc1 -- so a per-clip fit weights the crowded levels
    and a level-means fit treats all five alike, which is the right reading
    next to `span`. `slope_clip` carries the per-clip fit for anyone who wants
    it; they agree exactly whenever the level means are collinear.

    For a perturbed variant `y` is the CLEAN clip's PC -- the variant's own true
    score is unknown, and for a temporal one it is supposed to be lower. So a
    variant row is read against the clean row, never as accuracy. Nothing here
    is an MAE, which is exactly why it is computable on those rows at all."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    means, counts = {}, {}
    for level in PC_LEVELS:
        m = y == level
        if m.any():
            means[level] = float(pred[m].mean())
            counts[level] = int(m.sum())

    span, slope = float("nan"), float("nan")
    if len(means) > 1:
        span = max(means.values()) - min(means.values())
        lv = np.array(sorted(means), dtype=np.float64)
        mu = np.array([means[k] for k in sorted(means)], dtype=np.float64)
        slope = float(((lv - lv.mean()) * (mu - mu.mean())).sum()
                      / ((lv - lv.mean()) ** 2).sum())

    var = y.var()
    slope_clip = float("nan")
    if var > 0:
        slope_clip = float(((y - y.mean()) * (pred - pred.mean())).mean() / var)

    return {"means": means, "counts": counts, "span": span, "slope": slope,
            "slope_clip": slope_clip, "n": int(len(y))}


def print_compression_table(clean, stats, indent="  "):
    """The clean compression row, then the same columns for every perturbation,
    grouped by taxonomy half.

    The point of the table is the comparison down a column: a temporal
    perturbation the probe actually notices should pull the level means down
    and flatten the slope, and a superficial one should leave both alone.
    """
    if not clean:
        return
    levels = sorted(clean["means"])
    w = 36

    def row(name, c, ref=None):
        line = f"{indent}{name:<{w}} {c['n']:>5}"
        for lv in levels:
            v = c["means"].get(lv)
            line += f" {v:>7.3f}" if v is not None else f" {'-':>7}"
        d = "--" if ref is None else f"{c['span'] - ref['span']:+.3f}"
        print(line + f" {c['span']:>7.3f} {c['slope']:>7.3f} {d:>8}")

    head = f"{indent}{'variant':<{w}} {'n':>5}"
    for lv in levels:
        head += f" {'pc' + str(lv):>7}"
    print(head + f" {'span':>7} {'slope':>7} {'d span':>8}")
    print(f"{indent}{'(n per level, clean)':<{w}} {'':>5}"
          + "".join(f" {clean['counts'].get(lv, 0):>7}" for lv in levels))
    row("clean", clean)

    groups = defaultdict(list)
    for name, r in stats.items():
        if r.get("compression"):
            groups[r["kind"]].append((name, r["compression"]))
    for kind in sorted(groups, key=lambda k: {"temporal": 0}.get(k, 1)):
        want = ("should SAG and FLATTEN" if kind == "temporal"
                else "should sit on top of clean")
        label = "sensitivity" if kind == "temporal" else "invariance"
        print(f"{indent}-- expected-{label} ({want})")
        for name, c in sorted(groups[kind]):
            row(name, c, clean)

    print(f"{indent}   level means are grouped by the CLEAN clip's human PC, "
          "so a variant row is a shift")
    print(f"{indent}   of the clean row, not an accuracy -- the true score of "
          "a scrambled clip is")
    print(f"{indent}   unknown. slope is fit on the five level means (not "
          "per clip); 1.0 is calibrated.")


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
            pv = score(rows).cpu().numpy()
            d = (pv - base)[mask]
            cos = _cos_features(clean[mask], rows[mask])
            gaps[name] = float(np.abs(d).mean())
            signed[name] = float(d.mean())
            row = {"kind": variant_kind(name), "n": int(mask.sum()),
                   "signed": signed[name], "abs": gaps[name],
                   "median_abs": float(np.median(np.abs(d))),
                   "p95_abs": float(np.percentile(np.abs(d), 95)),
                   "frac_over_half": float((np.abs(d) > 0.5).mean()),
                   "cos": float(cos.mean()), "cos_min": float(cos.min()),
                   "compression": compression(pv[mask], np.asarray(y)[mask])}
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
            "compression": compression(base, y),
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


def per_split_headline(model, packs, splits, device):
    """Clean metrics for each held-out split on its own, so the pooled number
    can never hide one split doing the work. val was used to pick the epoch,
    so it is mildly optimistic; cal is untouched."""
    out = {}
    for split in splits:
        clean, y, _, _ = build_eval(packs[split])
        m = evaluate(model, clean, y, {}, device, attn=False)
        out[split] = {"n": int(len(y)), "mae": m["mae"],
                      "macro_mae": m["macro_mae"], "rho": m["rho"]}
    return out


def print_heldout(h, indent="  "):
    """The pooled held-out block: per-split headline, then the variant table."""
    label = "+".join(h["splits"])
    print(f"{indent}=== held-out ({label}, {h['n']} clips) ===")
    for split, r in h["per_split"].items():
        note = " (used for early stopping)" if split == "val" else " (untouched)"
        print(f"{indent}  {split:<5} n {r['n']:>4}  mae {r['mae']:.4f}   "
              f"macro_mae {r['macro_mae']:.4f}   rho {r['rho']:+.4f}{note}")
    print_headline(h, indent=indent + "  ", label="pooled ")
    print(f"{indent}  mean invariance gap (superficial only) "
          f"{h['mean_gap']:.4f}")
    print()
    print_variant_table(h["variant_stats"], indent=indent + "  ")
    print()
    print(f"{indent}  -- compression, {label} --")
    print_compression_table(h.get("compression"), h["variant_stats"],
                            indent=indent + "  ")


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
        packs = {s: load_pack(s) for s in ("train",) + HELD_OUT_SPLITS}

    Xc, Xp, y, _ = build_train(packs["train"])
    # selection is on val alone -- cal must never touch early stopping --
    # but the final tables pool both held-out splits
    ev_clean, ev_y, ev_variants, _ = build_eval(packs["val"])
    ho_splits = tuple(s for s in HELD_OUT_SPLITS if s in packs)
    ho = (build_eval(concat_packs([packs[s] for s in ho_splits]))
          if len(ho_splits) > 1 else None)

    # the pack is self-describing: rank 2 is mean-pooled, rank 3 temporal.
    # An explicit --arch still has to agree with the pack it was handed.
    if cfg["arch"] is None:
        cfg["arch"] = "attn" if Xc.ndim == 3 else "mlp"
    if cfg["arch"] in TEMPORAL_ARCHS:
        if Xc.ndim != 3:
            raise ValueError(f"arch={cfg['arch']!r} needs the [32,1024] packs; "
                             f"got features of shape {Xc.shape[1:]}")
        cfg["n_temporal"] = int(Xc.shape[1])
    elif cfg["arch"] == "mlp":
        if Xc.ndim != 2:
            raise ValueError("arch='mlp' needs the mean-pooled packs; got "
                             f"features of shape {Xc.shape[1:]}")
    else:
        raise ValueError(f"unknown arch {cfg['arch']!r}; expected one of {ARCHS}")

    if verbose:
        counts = {p: int((y == p).sum()) for p in PC_LEVELS}
        print(f"  train {len(y)} clips {counts}")
        print(f"  val   {len(ev_y)} clips, {len(ev_variants)} perturbation "
              f"types  (selection set)")
        if ho:
            print(f"  held-out {len(ho[1])} clips over {'+'.join(ho_splits)}, "
                  f"{len(ho[2])} perturbation types  (reporting set)")
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

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  arch={cfg['arch']} head={cfg['head']} "
              f"hidden={cfg['hidden']} params={n_params/1e3:.0f}K")
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
    best["compression"] = final["compression"]
    if unselected:
        best.update({k: final[k] for k in
                     ("mae", "rmse", "macro_mae", "rho", "mean_gap", "gaps",
                      "signed", "mean_temporal_signed", "attention")})

    # the same checkpoint on every held-out clip, not just the selection half
    if ho:
        ho_clean, ho_y, ho_variants, _ = ho
        h = evaluate(model, ho_clean, ho_y, ho_variants, device,
                     n_boot=cfg["n_boot"], seed=cfg["seed"])
        h.pop("pred_clean")
        best["heldout"] = dict(h, splits=list(ho_splits), n=int(len(ho_y)),
                               per_split=per_split_headline(
                                   model, packs, ho_splits, device))

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
        print()
        print("    === compression (val) ===")
        print_compression_table(best["compression"], best["variant_stats"],
                                indent="    ")
        if best.get("heldout"):
            print()
            print_heldout(best["heldout"], indent="    ")

    return {"cfg": cfg, "best": best, "history": history, "epochs_run": ran,
            "early_stopped": stopped is not None, "params": n_params,
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
        "heldout": result["best"].get("heldout"),
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


def ablation(archs=TEMPORAL_ARCHS, device=None, packs=None, **overrides):
    """The mentor's 2026-08-21 ablation: one probe per architecture, same
    embeddings, same split, same objective and hyperparameters, so the only
    thing that varies is how the 32 moments get pooled.

    His framing is that adding shuffle/reverse/freeze to *training* would
    engineer the result -- there is no human PC label for a scrambled clip --
    so the temporal question gets answered by varying the architecture instead
    and reading the within-clip deltas, which need no label.

    Selection is clean-val macro MAE for every arch (the mentor named it);
    Spearman is printed beside it as secondary evidence, never as the
    selector."""
    if packs is None:
        print("loading packs ...")
        packs = {s: load_pack(s) for s in ("train",) + HELD_OUT_SPLITS}

    runs = {}
    for arch in archs:
        print(f"\n=== arch={arch} ===")
        r = train(device=device, verbose=False, packs=packs, arch=arch,
                  **overrides)
        runs[arch] = r
        b = r["best"]
        print(f"  best epoch {b['epoch']}  macro {b['macro_mae']:.4f}  "
              f"mae {b['mae']:.4f}  rho {b['rho']:+.3f}  "
              f"({r['seconds']:.1f}s, {r['epochs_run']} epochs)")

    cfg = next(iter(runs.values()))["cfg"]
    print(f"\n  identical for every row: select={cfg['select']} "
          f"lr={cfg['lr']} wd={cfg['weight_decay']} bs={cfg['batch_size']} "
          f"hidden={cfg['hidden']} dropout={cfg['dropout']} "
          f"alpha={cfg['alpha']} lambda_cons={cfg['lambda_cons']} "
          f"consistency={cfg['consistency']} class_weight={cfg['class_weight']} "
          f"head={cfg['head']} seed={cfg['seed']}")

    def block(key, title):
        print(f"\n  -- {title} --")
        print(f"  {'arch':<13} {'params':>8} {'epoch':>6} {'macro':>8} "
              f"{'mae':>8} {'rho':>8} {'nuisance |d|':>13} {'nuisance d':>11} "
              f"{'temporal d':>11} {'H/logT':>7}")
        for arch in archs:
            b = runs[arch]["best"]
            m = (b.get(key) or b) if key else b
            att = b.get("attention")
            sup = [v for k, v in m["signed"].items()
                   if variant_kind(k) == "superficial"]
            print(f"  {arch:<13} {runs[arch]['params']/1e3:>7.0f}K "
                  f"{b['epoch']:>6} {m['macro_mae']:>8.4f} {m['mae']:>8.4f} "
                  f"{m['rho']:>+8.3f} {m['mean_gap']:>13.4f} "
                  f"{np.mean(sup) if sup else float('nan'):>+11.4f} "
                  f"{m.get('mean_temporal_signed', float('nan')):>+11.4f} "
                  + (f"{att['normalized']:>7.3f}" if att else f"{'-':>7}"))

    block(None, "val (the selection split)")
    if next(iter(runs.values()))["best"].get("heldout"):
        block("heldout", "held-out (val+cal, the reporting set)")

    print(f"\n  -- signed delta per temporal perturbation "
          "(these SHOULD be negative) --")
    tmp = sorted(TEMPORAL_VARIANTS)
    print(f"  {'arch':<13}" + "".join(f" {t:>10}" for t in tmp))
    for arch in archs:
        b = runs[arch]["best"]
        m = b.get("heldout") or b
        print(f"  {arch:<13}"
              + "".join(f" {m['signed'].get(t, float('nan')):>+10.4f}"
                        for t in tmp))
    print("  mean_linear averages time away before the head sees it, so its "
          "row is the")
    print("  order-blind null -- read every other row against it, not "
          "against zero.")
    return runs


def sweep(lambdas=(0.0, 10.0, 100.0, 1000.0), device=None, **overrides):
    """One training per lambda on shared packs."""
    print("loading packs ...")
    packs = {s: load_pack(s) for s in ("train",) + HELD_OUT_SPLITS}

    runs = []
    for lam in lambdas:
        print(f"\n=== lambda_cons={lam} ===")
        r = train(device=device, verbose=False, packs=packs,
                  lambda_cons=lam, **overrides)
        runs.append(r)
        att = r["best"].get("attention")
        m = r["best"].get("heldout") or r["best"]
        tag = "held-out" if r["best"].get("heldout") else "val"
        print(f"  {tag} mae {m['mae']:.4f}   "
              f"macro {m['macro_mae']:.4f}   "
              f"rho {m['rho']:+.3f}   "
              f"mean gap {m['mean_gap']:.4f}   "
              + (f"H {att['normalized']:.3f}   " if att else "")
              + f"({r['seconds']:.1f}s)")

    print(f"\n  {'lambda':>8} {'ho mae':>9} {'macro':>9} {'rho':>7} "
          f"{'sup gap':>9} {'temporal d':>11} {'H/logT':>7} {'epoch':>6}")
    print("  metrics are the pooled held-out set where available, else val")
    for lam, r in zip(lambdas, runs):
        b = r["best"]
        m = b.get("heldout") or b
        att = b.get("attention")
        print(f"  {lam:>8} {m['mae']:>9.4f} {m['macro_mae']:>9.4f} "
              f"{m['rho']:>+7.3f} {m['mean_gap']:>9.4f} "
              f"{m.get('mean_temporal_signed', float('nan')):>+11.4f} "
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
    ap.add_argument("--arch", choices=list(ARCHS), default=None,
                    help="default: inferred from the pack rank (attn on t32)")
    ap.add_argument("--conv-kernel", type=int, default=None,
                    help="diff_conv only")
    ap.add_argument("--ablation", action="store_true",
                    help="train every temporal arch on shared packs and "
                         "print the comparison table")
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
    abl = a.pop("ablation")
    if abl:
        a.pop("arch")   # the ablation is over arch; a fixed one makes no sense
        for arch, r in ablation(device=dev, **a).items():
            save_probe(r, name=f"{name}_{arch}", push_to_s3=not no_push)
    elif lams:
        sweep(lams, device=dev, **a)
    else:
        save_probe(train(device=dev, **a), name=name, push_to_s3=not no_push)
