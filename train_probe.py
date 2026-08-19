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
    select="mae",       # mae | macro_mae | rho
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
    """-> X_clean, y, {variant: (X, mask)}, stems.

    Perturbations stay aligned to their clean clip so the invariance gap is
    measured within-clip.
    """
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

    import math
    import torch
    import torch.nn as nn

    class OrdinalHead(nn.Module):
        """4 cumulative logits, P(PC>1)..P(PC>4).

        'linear' is unconstrained, so nothing forces P(PC>1) >= P(PC>2) >= ...
        'coral' shares one score and gives each threshold its own bias, so the
        ordering follows from the biases coming out sorted. 'corn' keeps a free
        Linear(d, 4) but reads it as conditional logits, P(PC>k | PC>k-1); the
        cumulative probabilities are their running product, which cannot
        increase.
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
        attentive pooling over the 32 moments -> ordinal head.

        A physics violation is local in time, and mean pooling dilutes one bad
        moment 32:1. The weights are inspectable through attention().
        """

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
            self.scale = 1.0 / math.sqrt(d)
            nn.init.normal_(self.pos, std=0.02)
            nn.init.normal_(self.query, std=0.02)
            self._adjust = None

        def set_logit_adjust(self, vec):
            """Undo the pos_weight shift at prediction time.

            Training with pos_weight w_k shifts the logit-space optimum by
            log(w_k), so the four sigmoids do not sum to a calibrated E[PC].
            Eval only: in training it would just be absorbed into the bias.
            """
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
    """Per-clip loss weight from PC frequency, normalised to mean 1.

    The weight attaches to the clip, so its clean row and its two perturbed
    rows share one; weighting rows would favour clips with more variants. This
    stacks with threshold_pos_weight, which corrects a different imbalance.
    """
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
    """Toggle only the Dropout layers, leaving the rest in train mode.

    The consistency term needs dropout off: with it on the clean and perturbed
    passes draw different masks and the penalty measures that noise, which was
    ~13x the real signal at dropout=0.2.
    """
    import torch.nn as nn
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train(active)


def consistency_loss(logits_a, logits_b, kind="mse"):
    """Distance between the clean and perturbed threshold probabilities.
    Symmetric and not detached: the goal is that the two agree, not that one
    chases the other."""
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

def _spearman(a, b):
    """Underscored so eval_probe.spearman cannot shadow it when both files are
    pasted into one notebook."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(model, clean, y, variants, device, batch=1024):
    """-> dict with clean MAE and the per-variant invariance gap."""
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

        gaps = {}
        for name, (rows, mask) in variants.items():
            if not mask.any():
                continue
            pred = score(rows)
            m = torch.as_tensor(mask, device=device)
            gaps[name] = (pred - pred_clean).abs()[m].mean().item()

    model.train()
    return {"mae": mae, "rmse": rmse, "macro_mae": macro_mae, "rho": rho,
            "gaps": gaps,
            "mean_gap": float(np.mean(list(gaps.values()))) if gaps else 0.0,
            "pred_clean": pred_clean.cpu().numpy()}


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
            tr = evaluate(model, Xc, y, {}, device)
            spread = float(m["pred_clean"].max() - m["pred_clean"].min())
            row = {"epoch": epoch, "loss_clean": totals[0] / nb,
                   "loss_pert": totals[1] / nb, "loss_cons": totals[2] / nb,
                   "train_mae": tr["mae"], "val_mae": m["mae"],
                   "val_macro_mae": m["macro_mae"], "val_rho": m["rho"],
                   "val_gap": m["mean_gap"], "val_spread": spread}
            history.append(row)
            if verbose:
                print(f"  epoch {epoch:>4}  clean {row['loss_clean']:.4f}  "
                      f"cons {row['loss_cons']:.5f}   train mae "
                      f"{tr['mae']:.4f}  val mae {m['mae']:.4f}  "
                      f"macro {m['macro_mae']:.4f}  rho {m['rho']:+.3f}  "
                      f"gap {m['mean_gap']:.4f}  spread {spread:.2f}")
            # rho negated so one "lower is better" comparison covers all three
            score = -m["rho"] if cfg["select"] == "rho" else m[cfg["select"]]
            if score < best["score"]:
                best = {"score": score, "mae": m["mae"], "rmse": m["rmse"],
                        "macro_mae": m["macro_mae"], "rho": m["rho"],
                        "mean_gap": m["mean_gap"], "gaps": m["gaps"],
                        "epoch": epoch,
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
    if verbose:
        print(f"\n  {ran} epochs in {secs:.1f}s ({secs/ran*1000:.0f} ms/epoch)"
              + ("" if stopped else "  [hit the epoch cap, not early stopping]"))
        print(f"  best epoch {best['epoch']}: val mae {best['mae']:.4f}, "
              f"macro mae {best['macro_mae']:.4f}, rho {best['rho']:+.3f}, "
              f"mean invariance gap {best['mean_gap']:.4f}")
        for name, g in sorted(best["gaps"].items()):
            print(f"    {name:42s} {g:.4f}")

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
                 "epoch")},
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
    """Upload a probe already saved locally, so the normal flow can be train ->
    save -> test -> push only if it is good."""
    local = Path(f"./{name}.pt")
    if not local.exists():
        raise FileNotFoundError(f"{local} not found; run save_probe first")
    key = f"{PROBE_PREFIX}/{name}.pt"
    _ensure_s3().put_object(Bucket=BUCKET, Key=key, Body=local.read_bytes())
    print(f"uploaded {local} -> s3://{BUCKET}/{key}")
    return key


def sweep(lambdas=(0.0, 10.0, 100.0, 1000.0), device=None, **overrides):
    """One training per lambda on shared packs. Too low leaves the probe
    gameable, too high collapses it toward a constant, so read the mae column
    and the gap column together."""
    print("loading packs ...")
    packs = {s: load_pack(s) for s in ("train", "val")}

    runs = []
    for lam in lambdas:
        print(f"\n=== lambda_cons={lam} ===")
        r = train(device=device, verbose=False, packs=packs,
                  lambda_cons=lam, **overrides)
        runs.append(r)
        print(f"  val mae {r['best']['mae']:.4f}   "
              f"macro {r['best']['macro_mae']:.4f}   "
              f"rho {r['best']['rho']:+.3f}   "
              f"mean gap {r['best']['mean_gap']:.4f}   "
              f"({r['seconds']:.1f}s)")

    print(f"\n  {'lambda':>8} {'val mae':>9} {'macro':>9} {'rho':>7} {'mean gap':>9} {'epoch':>6}")
    for lam, r in zip(lambdas, runs):
        print(f"  {lam:>8} {r['best']['mae']:>9.4f} "
              f"{r['best']['macro_mae']:>9.4f} {r['best']['rho']:>+7.3f} "
              f"{r['best']['mean_gap']:>9.4f} {r['best']['epoch']:>6}")
    print("\n  pick the largest lambda that has not started to cost mae")
    return runs


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(description="train the ordinal PC probe")
    for k in ("hidden", "batch_size", "epochs", "patience", "seed", "eval_every"):
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
