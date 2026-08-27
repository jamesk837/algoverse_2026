

import argparse
import os

import numpy as np

try:
    from train_probe import PC_LEVELS, group_by_stem, load_pack, make_model
    from eval_probe import load_probe, predict
except ImportError:
    pass

HELD_OUT = ("val", "cal")

SA_GOOD = 4           # "high SA"
PC_BAD, PC_GOOD = 2, 4
MISSING = -1          # build_split writes -1 for a label the CSV did not have


def in_notebook():
    try:
        __IPYTHON__          # noqa: F821
        return True
    except NameError:
        return False


def clean_rows(pack):
    """-> (stems, X, pc, sa) for the clean variant of every clip in the pack."""
    grouped = group_by_stem(pack)
    stems = [s for s in sorted(grouped) if "clean" in grouped[s]]
    idx = np.array([grouped[s]["clean"] for s in stems])
    return (stems,
            pack["X"].astype(np.float32)[idx],
            pack["pc"][idx].astype(np.int64),
            pack["sa"][idx].astype(np.int64))


def level_means(pred, y, stems=None, sa=None, examples=3):
    """Question 1: mean predicted PC for each human PC label, plus a few
    individual clips at each label -- "evaluate the probe on a few clips for
    each different human pc label 1-5"."""
    print(f"\n=== mean predicted PC per human PC label ({len(y)} clips) ===")
    print(f"  {'human pc':>8} {'n':>5} {'mean pred':>10}")

    means = {}
    for level in PC_LEVELS:
        m = y == level
        if not m.any():
            continue
        means[level] = float(pred[m].mean())
        print(f"  {level:>8} {int(m.sum()):>5} {means[level]:>10.3f}")

    if len(means) >= 2:
        lo, hi = min(means.values()), max(means.values())
        ideal = max(means) - min(means)
        print(f"\n  level means run {lo:.3f} -> {hi:.3f}, "
              f"a span of {hi - lo:.3f} where the labels span {ideal:.1f}")

    if examples and stems is not None:
        print()
        print(f"  -- {examples} example clips per label --")
        print(f"  {'pc':>3} {'sa':>3} {'pred':>7}  clip")
        stems = np.asarray(stems)
        for level in PC_LEVELS:
            for i in np.flatnonzero(y == level)[:examples]:
                sa_i = "-" if sa is None or sa[i] == MISSING else str(sa[i])
                print(f"  {level:>3} {sa_i:>3} {pred[i]:>7.3f}  "
                      f"{str(stems[i])[:52]}")
    return means


def sa_controlled(pred, y, sa):
    """Question 2: bad physics vs good physics, both with good semantics."""
    if (sa == MISSING).all():
        print("\nno SA labels in this split; skipping the SA comparison")
        return {}

    bad = (sa >= SA_GOOD) & (y <= PC_BAD)
    good = (sa >= SA_GOOD) & (y >= PC_GOOD)

    print(f"\n=== physics under good semantics (SA >= {SA_GOOD}) ===")
    print(f"  {'group':<34} {'n':>5} {'mean pred':>10}")
    print(f"  {f'PC <= {PC_BAD}  (bad physics)':<34} {int(bad.sum()):>5} "
          f"{pred[bad].mean() if bad.any() else float('nan'):>10.3f}")
    print(f"  {f'PC >= {PC_GOOD}  (good physics)':<34} {int(good.sum()):>5} "
          f"{pred[good].mean() if good.any() else float('nan'):>10.3f}")

    out = {"n_bad": int(bad.sum()), "n_good": int(good.sum())}
    if bad.any() and good.any():
        out["mean_bad"] = float(pred[bad].mean())
        out["mean_good"] = float(pred[good].mean())
        out["gap"] = out["mean_good"] - out["mean_bad"]
        print(f"\n  good - bad = {out['gap']:+.3f}   "
              f"(the human labels differ by at least "
              f"{PC_GOOD - PC_BAD:.0f} points)")
    return out


def diagnose(name="probe_v1", device=None, packs=None, splits=HELD_OUT,
             examples=3, return_arrays=False):
    """Prints the two tables. Returns only the summary numbers -- a notebook
    auto-displays the return value of the last expression, and handing back the
    669-element prediction array buried the tables under a wall of floats.
    Pass return_arrays=True if you actually want them."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = tuple(s for s in splits if s in HELD_OUT)
    if not splits:
        raise ValueError(f"held-out only; splits must be within {HELD_OUT}")

    model, ckpt = load_model(name, device)
    val = ckpt.get("val", {})
    nan = float("nan")
    print(f"probe '{name}': best epoch {val.get('epoch')} "
          f"(selected on {ckpt.get('cfg', {}).get('select', 'mae')})")
    print(f"  val mae {val.get('mae', nan):.4f}   "
          f"macro_mae {val.get('macro_mae', nan):.4f}   "
          f"rho {val.get('rho', nan):+.4f}")

    if packs is None:
        print("loading packs ...")
        packs = {s: load_pack(s) for s in splits}

    preds, ys, sas, stems = [], [], [], []
    for s in splits:
        if s not in packs:
            continue
        st, X, pc, sa_i = clean_rows(packs[s])
        preds.append(predict(model, X, device))
        ys.append(pc)
        sas.append(sa_i)
        stems.extend(st)

    pred = np.concatenate(preds)
    y = np.concatenate(ys)
    sa = np.concatenate(sas)
    print(f"evaluating on {'+'.join(splits)}, clean clips only")

    means = level_means(pred, y, stems=stems, sa=sa, examples=examples)
    out = sa_controlled(pred, y, sa)

    summary = {"level_means": {k: round(v, 3) for k, v in means.items()},
               "sa": {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in out.items()}}
    if return_arrays:
        summary |= {"pred": pred, "pc": y, "sa_label": sa}
    return summary


def load_model(name, device):
    import torch
    ckpt = load_probe(name)
    model = make_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(description="the project lead's two probe questions")
    ap.add_argument("--probe", default="probe_v1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--splits", nargs="+", default=list(HELD_OUT),
                    choices=list(HELD_OUT))
    ap.add_argument("--examples", type=int, default=3,
                    help="example clips printed per PC label (0 for none)")
    a = ap.parse_args()
    diagnose(name=a.probe, device=a.device, splits=tuple(a.splits),
             examples=a.examples)
