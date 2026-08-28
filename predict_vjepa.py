"""Step 10 -- V-JEPA predictor / latent surprise.

The PC probe reads V-JEPA's ENCODER: "describe this clip". This file uses the
other half, the PREDICTOR: "given what you have seen, what happens next?" It
was trained to be punished for guessing wrong, so its prediction error is an
opinion about normal dynamics that needs no labels, no training, and -- most
importantly -- no probe.

That last part is the point. The locked reference is `proj_mean`, which means
away the time axis and is provably order-invariant, so its temporal response
cannot distinguish "V-JEPA does not represent order" from "the readout cannot
recover it". The predictor answers that question without a readout.

What it does per clip: encode all 64 frames once, then for each moment t ask
the predictor to produce moment t's tokens from the preceding `context`
moments, and record how wrong it was. That yields a length-(32 - context)
error sequence, which is the doc's "raw latent prediction-error sequence".

Corpus, per the doc: clean train clips (which define the spike threshold and
nothing else), plus held-out clean clips and ALL nine of their variants.

    python predict_vjepa.py --verify          # DO THIS FIRST, ~2 min
    python predict_vjepa.py --dry-run
    python predict_vjepa.py --limit 20
    python predict_vjepa.py                   # full run, resumable
    python predict_vjepa.py --report

Colab: wget embed_vjepa.py and this file, then
    import predict_vjepa; predict_vjepa.verify(); predict_vjepa.run()
"""

import argparse
import io
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# embed_vjepa owns the S3 layout, the split doc, video download and decode, and
# the frame sampling. Reusing it is not just convenience: the predictor's clips
# have to be preprocessed EXACTLY as the probe's were, or the two analyses are
# not describing the same videos.
try:
    import embed_vjepa as E
except ImportError as e:
    raise ImportError(
        f"could not import embed_vjepa ({type(e).__name__}: {e}).\n"
        "  - put embed_vjepa.py in the working directory (it owns the split, "
        "the S3 layout, decode and frame sampling)\n"
        "  - it needs boto3, opencv-python-headless and numpy") from e

BUCKET = E.BUCKET
HUB_MODEL = E.HUB_MODEL
N_TEMPORAL = E.N_TEMPORAL          # 32 moments
N_SPATIAL = E.N_SPATIAL            # 576 spatial tokens per moment
EMBED_DIM = E.EMBED_DIM

ERR_PREFIX = f"predictor/{HUB_MODEL}/errors"
PACK_KEY = f"predictor/{HUB_MODEL}/pack.npz"
REPORT_KEY = f"predictor/{HUB_MODEL}/report.json"

TMP_DIR = Path("./tmp_predict")

TEMPORAL_VARIANTS = E.TEMPORAL_VARIANTS
HELD_OUT_SPLITS = E.HELD_OUT_SPLITS

# ---- the two config choices that are genuinely arbitrary --------------------
#
# CONTEXT: how many preceding moments the predictor is given. 1 is the
# strictest reading of "prediction-error sequence" and keeps every row the same
# shape so one batched call does the whole clip. It is also further from
# V-JEPA's training distribution (multi-block masks over most of the clip) than
# a longer window would be. That inflates the ABSOLUTE error, which is why
# nothing here reads absolute error: every number reported is a within-clip
# clean->attack difference, where a constant OOD offset cancels.
#
# MASK_INDEX: the predictor carries 10 learned mask tokens, chosen during
# training by mask type. There is no principled inference-time choice, so this
# pins upstream's own default. What matters is that it is CONSTANT across clean
# and attacked -- varying it would put the two sides on different queries.
CONTEXT = 1
MASK_INDEX = 1

# CONTEXT_MODE: how the context tokens are produced.
#
#   "masked" -- encoder(x, [mask]) per row, so the context is encoded from the
#     patches it is allowed to see and nothing else. This is what V-JEPA does
#     in training: vision_transformer.forward applies the mask right after
#     patch_embed, BEFORE the transformer blocks, so the hidden patches never
#     enter attention. Costs a second encoder pass (~2x).
#
#   "full" -- encode the whole clip once and gather the context tokens out of
#     it. Cheaper, but those tokens attended to the future before being
#     gathered, so a bit of the answer leaks into the question and the
#     predictor's job is easier than it should be.
#
# "masked" is the default because it is the faithful one. "full" is kept as the
# fallback and as a robustness column: if the two disagree, the leak is doing
# real work and that belongs in the writeup.
CONTEXT_MODE = "masked"

SPIKE_PERCENTILE = 95.0     # threshold from clean TRAIN only, per the doc
N_BOOT = 2000


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


# --------------------------------------------------------------- selection

def has_train_split(doc):
    """A split doc carries train/val/cal; a benchmark doc built by
    embed_vjepa.load_doc() labels every clip with the dataset name instead."""
    return any(e["split"] == "train" for e in doc["clips"].values())


def variants_for(entry, doc, has_train=None):
    """train contributes CLEAN ONLY, everything else contributes clean plus all
    nine variants.

    Deliberately not embed_vjepa.variants_needed(), which gives a train clip
    its two assigned superficial perturbations. Those exist to train the probe;
    here train clips have exactly one job -- defining the spike threshold from
    a pile of normal clips nothing else in the analysis touches.

    On a benchmark corpus there is no train/val/cal, so every clip carries the
    full set. Without that branch a benchmark doc silently yields nothing --
    it would run to completion having computed zero errors.
    """
    if has_train is None:
        has_train = has_train_split(doc)
    full = (["clean"] + list(doc["superficial_variants"])
            + list(E.temporal_variants(doc)))
    if not has_train:
        return full
    if entry["split"] == "train":
        return ["clean"]
    if entry["split"] in E.held_out_splits(doc):
        return full
    return []


def plan(doc, limit=None, workers=E.CACHE_READ_WORKERS):
    """-> [(stem, [missing variants])], resumable per (clip, variant)."""
    cached = cached_stems()
    todo, have, n_want = [], 0, 0
    has_train = has_train_split(doc)
    stems = sorted(doc["clips"])
    for stem in stems:
        want = variants_for(doc["clips"][stem], doc, has_train)
        if not want:
            continue
        n_want += len(want)
        done = cached.get(stem, set())
        missing = [v for v in want if v not in done]
        have += len(want) - len(missing)
        if missing:
            todo.append((stem, missing))
    if limit:
        todo = todo[:limit]
    print(f"  {n_want} (clip, variant) pairs wanted, {have} already cached, "
          f"{sum(len(m) for _, m in todo)} to compute over {len(todo)} clips")
    return todo


def err_key(stem):
    return f"{ERR_PREFIX}/{E.safe_local_name(stem)}.npz"


def cached_stems():
    """{stem: {variant}} from one LIST plus parallel GETs of the headers."""
    keys = E.list_keys(ERR_PREFIX + "/")
    if not keys:
        return {}
    out = {}
    stems = [Path(k).stem for k in keys]

    def read(stem):
        try:
            body = E._ensure_s3().get_object(
                Bucket=BUCKET, Key=f"{ERR_PREFIX}/{stem}.npz")["Body"].read()
            with np.load(io.BytesIO(body), allow_pickle=False) as z:
                return stem, {k[4:] for k in z.files if k.startswith("err_")}
        except Exception:
            return stem, set()

    with ThreadPoolExecutor(max_workers=E.CACHE_READ_WORKERS) as pool:
        for stem, variants in pool.map(read, stems):
            out[stem] = variants
    # keys are safe_local_name'd; map back by matching on the stored real stem
    return {k: v for k, v in out.items()}


# ------------------------------------------------------------------ model

def load_models(device="cuda"):
    """Encoder AND predictor. embed_vjepa deletes the predictor on line 416;
    this is the same checkpoint, keeping both halves."""
    import torch

    print(f"loading {HUB_MODEL} (encoder + predictor) from torch.hub ...")
    processor = torch.hub.load(E.HUB_REPO, E.HUB_PREPROCESSOR,
                               crop_size=E.CROP_SIZE, trust_repo=True)
    E.patch_hub_base_url()
    encoder, predictor = torch.hub.load(E.HUB_REPO, HUB_MODEL, trust_repo=True)

    for m in (encoder, predictor):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False

    ne = sum(p.numel() for p in encoder.parameters())
    npd = sum(p.numel() for p in predictor.parameters())
    print(f"  encoder {ne/1e6:.0f}M, predictor {npd/1e6:.0f}M, both frozen "
          f"on {device}")

    stats = check_predictor_weights(predictor)
    if stats is None:
        print("  WARNING could not locate the checkpoint to verify the "
              "predictor weights; proceeding")
    else:
        print(f"  predictor weights: {stats['matched']}/{stats['in_model']} "
              f"tensors matched the checkpoint, max abs diff "
              f"{stats['max_diff']:.2e}")
        if stats["matched"] < 0.8 * stats["in_model"]:
            raise RuntimeError(
                f"only {stats['matched']} of {stats['in_model']} predictor "
                "tensors came from the checkpoint. upstream loads the "
                "predictor with strict=False, so a key mismatch leaves it at "
                "RANDOM INIT and every number below would be noise.")
    return processor, encoder, predictor


def check_predictor_weights(predictor):
    """Did the predictor actually load, or is it still at init?

    `_make_vjepa2_model` does `predictor.load_state_dict(..., strict=False)`.
    That is correct for the pos_embed keys RoPE makes unused, but it also means
    a renamed or reshaped tensor is silently skipped and those weights stay
    random -- the same class of silent corruption as the meta-device bug on
    phyjudge, and just as invisible in the output. Best effort: a missing
    checkpoint file prints rather than fails.
    """
    import torch
    ckpts = sorted(Path(torch.hub.get_dir()).glob("checkpoints/*.pt"))
    ckpts = [c for c in ckpts if "vitl" in c.name.lower()] or ckpts
    if not ckpts:
        return None
    try:
        blob = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        sd = blob["predictor"]
    except Exception:
        return None
    sd = {k.replace("module.", "").replace("backbone.", ""): v
          for k, v in sd.items()}
    live = predictor.state_dict()
    matched = [k for k in sd
               if k in live and tuple(live[k].shape) == tuple(sd[k].shape)]
    diff = max(((live[k].float().cpu() - sd[k].float()).abs().max().item()
                for k in matched), default=float("nan"))
    return {"in_ckpt": len(sd), "in_model": len(live),
            "matched": len(matched), "max_diff": diff}


def causal_masks(context=CONTEXT, t=N_TEMPORAL, s=N_SPATIAL, device="cpu"):
    """-> (masks_x, masks_y) index tensors, one row per predicted moment.

    Moment i owns token indices [i*s, (i+1)*s) -- the same temporal-major
    assumption pool_tokens() makes and verify_temporal_axis() checks. If that
    ever flips to spatial-major these masks address spatial groups instead of
    moments and the whole file measures nothing, silently.

    Batching over rows rather than over the predictor's mask-list argument is
    deliberate: forward() computes B = len(x) // len(masks_x), so a list of M
    masks against a batch of 1 gives B = 0.
    """
    import torch
    idx = torch.arange(t * s, device=device).reshape(t, s)
    xs = torch.stack([idx[i - context:i].reshape(-1) for i in range(context, t)])
    ys = torch.stack([idx[i] for i in range(context, t)])
    return xs, ys


def _prep(processor, frames, device):
    """Decode -> RGB -> preprocessor -> (1, C, T, H, W). Identical to
    embed_vjepa.embed()'s front half, so the predictor and the probe are
    looking at byte-identical model input."""
    import torch
    import cv2
    idx = E.frame_indices(len(frames))
    buf = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx])
    clip = processor(buf)
    if isinstance(clip, (list, tuple)):
        clip = clip[0]
    if clip.ndim != 4:
        raise RuntimeError(f"preprocessor returned ndim={clip.ndim}, expected 4")
    if clip.shape[0] != 3 and clip.shape[1] == 3:
        clip = clip.permute(1, 0, 2, 3)
    return clip.unsqueeze(0).to(device)


def _encode_context(encoder, x, mx, mode):
    """Context tokens for every row: (rows, context*S, D).

    In "masked" mode the mask list is one entry per row, and apply_masks
    concatenates along the batch dim -- so a single encoder call with 31 masks
    against a batch of 1 returns 31 independently-masked encodings.
    """
    if mode == "masked":
        return encoder(x, [mx[i:i + 1] for i in range(len(mx))])
    if mode == "full":
        tokens = encoder(x)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        return tokens[0][mx]
    raise ValueError(f"context_mode must be 'masked' or 'full', got {mode!r}")


def predict_errors(processor, encoder, predictor, frames, device="cuda",
                   context=CONTEXT, mask_index=MASK_INDEX,
                   context_mode=CONTEXT_MODE):
    """-> (err[T-context], cos[T-context], err_tok[T-context, S])

    err is the L1 gap between predicted and actual latents, averaged over the
    576 spatial tokens of the predicted moment. L1 because that is V-JEPA's own
    training objective (app/vjepa/train.py: torch.mean(torch.abs(z - h))), so
    it is the loss the predictor was actually optimised against rather than one
    chosen here. Cosine is reported alongside as a scale-free second view.
    """
    import torch

    x = _prep(processor, frames, device)

    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            # the FULL-clip pass is the ground truth, and is meant to see
            # everything -- upstream's target encoder does exactly this
            tokens = encoder(x)
            if isinstance(tokens, (list, tuple)):
                tokens = tokens[-1]
            n = int(tokens.shape[1])
            if n != N_TEMPORAL * N_SPATIAL:
                raise RuntimeError(
                    f"encoder returned {n} tokens, expected "
                    f"{N_TEMPORAL * N_SPATIAL}; the causal masks assume a "
                    f"{N_TEMPORAL}x{N_SPATIAL} temporal-major grid")

            mx, my = causal_masks(context, device=device)
            ctx = _encode_context(encoder, x, mx, context_mode)
            if ctx.shape[:2] != (len(mx), mx.shape[1]):
                raise RuntimeError(
                    f"context encoding is {tuple(ctx.shape)}, expected "
                    f"({len(mx)}, {mx.shape[1]}, D). apply_masks did not "
                    "expand the batch the way this assumes.")
            pred = predictor(ctx, mx, my, mask_index=mask_index)

    if isinstance(pred, (list, tuple)):
        pred = pred[-1]
    # return_all_tokens is off upstream, so this is the target block; slice
    # defensively in case a future version returns context+target
    if pred.shape[1] != N_SPATIAL:
        if pred.shape[1] < N_SPATIAL:
            raise RuntimeError(f"predictor returned {tuple(pred.shape)}, "
                               f"fewer than the {N_SPATIAL} target tokens")
        pred = pred[:, -N_SPATIAL:]

    actual = tokens[0][my].float()
    pred = pred.float()
    if pred.shape[-1] != actual.shape[-1]:
        raise RuntimeError(f"predictor width {pred.shape[-1]} != encoder width "
                           f"{actual.shape[-1]}; out_embed_dim changed")

    err_tok = (pred - actual).abs().mean(dim=-1)              # (rows, S)
    cos = torch.nn.functional.cosine_similarity(pred, actual, dim=-1)
    return (err_tok.mean(dim=-1).cpu().numpy().astype(np.float32),
            cos.mean(dim=-1).cpu().numpy().astype(np.float32),
            err_tok.cpu().numpy().astype(np.float16))


def copy_baseline(processor, encoder, frames, device="cuda", context=CONTEXT):
    """Error of the trivial "next moment looks like the last one" predictor.

    The check that catches a predictor which loaded as noise: a trained one
    must beat copying. Uses the same encoder pass and the same masks, so the
    two numbers are directly comparable.
    """
    import torch
    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            tokens = encoder(_prep(processor, frames, device))
            if isinstance(tokens, (list, tuple)):
                tokens = tokens[-1]
    mx, my = causal_masks(context, device=device)
    last = tokens[0][mx.reshape(len(mx), context, N_SPATIAL)[:, -1]].float()
    actual = tokens[0][my].float()
    return (last - actual).abs().mean(dim=-1).mean(dim=-1).cpu().numpy()


# ------------------------------------------------------------------- cache

def read_cached(stem):
    try:
        body = E._ensure_s3().get_object(
            Bucket=BUCKET, Key=err_key(stem))["Body"].read()
    except Exception:
        return {}
    with np.load(io.BytesIO(body), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_cached(stem, data):
    buf = io.BytesIO()
    np.savez_compressed(buf, **data)
    E._ensure_s3().put_object(Bucket=BUCKET, Key=err_key(stem),
                              Body=buf.getvalue())


def summarise(err, threshold=None):
    """The doc's six statistics over one clip's error sequence."""
    err = np.asarray(err, dtype=np.float64)
    out = {"mean": float(err.mean()), "max": float(err.max()),
           "std": float(err.std(ddof=1)) if len(err) > 1 else 0.0,
           "p90": float(np.percentile(err, 90)),
           "p95": float(np.percentile(err, 95)), "n": int(len(err))}
    out["spike_rate"] = (float((err > threshold).mean())
                         if threshold is not None else float("nan"))
    return out


# --------------------------------------------------------------------- run

def run(limit=None, device=None, dry_run=False, context=CONTEXT,
        store_tokens=True, doc=None, models=None,
        context_mode=CONTEXT_MODE):
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    doc = doc or E.load_split()
    print("planning ...")
    todo = plan(doc, limit=limit)
    if dry_run or not todo:
        return {"todo": len(todo)}

    processor, encoder, predictor = models or load_models(device)

    t0 = time.perf_counter()
    done, failed = 0, 0
    total = sum(len(m) for _, m in todo)
    for i, (stem, missing) in enumerate(todo, 1):
        data = read_cached(stem)
        keys = {v: E.video_key(doc, stem, v) for v in missing}
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = dict(zip(missing, pool.map(
                    lambda v: E.download_video(keys[v]), missing)))
        except Exception as e:
            print(f"FAILED {stem}: download: {type(e).__name__}: {e}")
            failed += len(missing)
            continue

        for variant in missing:
            try:
                p = paths[variant]
                if isinstance(p, Exception):
                    raise p
                frames = E.read_clip(p)
                err, cos, tok = predict_errors(
                    processor, encoder, predictor, frames, device, context,
                    context_mode=context_mode)
                data[f"err_{variant}"] = err
                data[f"cos_{variant}"] = cos
                if store_tokens:
                    data[f"tok_{variant}"] = tok
                done += 1
            except Exception as e:
                print(f"FAILED {stem} {variant}: {type(e).__name__}: {e}")
                failed += 1
            finally:
                p = paths.get(variant)
                if isinstance(p, (str, Path)):
                    Path(p).unlink(missing_ok=True)

        data["__context__"] = np.array([context])
        data["__mode__"] = np.array([context_mode])
        write_cached(stem, data)

        if i % 25 == 0 or i == len(todo):
            el = time.perf_counter() - t0
            rate = el / max(done, 1)
            print(f"  [{i}/{len(todo)}] {done}/{total} variants, "
                  f"{rate:.2f}s each, elapsed {E.fmt_secs(el)}, "
                  f"eta {E.fmt_secs(rate * (total - done))}", flush=True)

    print(f"\ndone: {done} computed, {failed} failed")
    if failed:
        print("  a clean exit is not a clean run -- grep FAILED above")
    return {"computed": done, "failed": failed}


def run_all(datasets="train", device=None, dry_run=False, context=CONTEXT,
            store_tokens=True, limit=None, context_mode=CONTEXT_MODE):
    """Several corpora back to back with the models loaded once.

    `datasets` takes embed_vjepa's spelling: "train" is the probe's own split
    (the doc's Step 10 corpus), "all" adds the three benchmark corpora, or name
    them individually. A corpus that blows up is reported and the run
    continues, and everything stays resumable per (clip, variant).

    Worth being explicit about what the benchmark corpora are FOR: the doc
    scopes Step 10 to the split corpus, because that is where a predictor-vs-PC
    probe comparison is possible at all. Running the benchmark corpora collects
    data the doc does not ask for -- cheap to gather now, expensive to come
    back for -- but the doc's caveat still binds when it comes to using it:
    predictor statistics are "compared qualitatively and statistically with the
    PC probe but not by directly comparing their raw numerical scales", so
    these do not slot into dJ - dV.
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    names = E.expand_datasets(datasets)
    label = ", ".join(n or "videophy2_train" for n in names)
    print(f"predictor over {len(names)} corpora: {label}")

    models = None if dry_run else load_models(device)

    summary = []
    for i, name in enumerate(names, 1):
        title = name or "videophy2_train"
        print(f"\n{'#' * 70}\n# [{i}/{len(names)}] {title}\n{'#' * 70}")
        try:
            doc = E.load_split() if name is None else E.load_doc(name)
            got = run(limit=limit, device=device, dry_run=dry_run,
                      context=context, store_tokens=store_tokens, doc=doc,
                      models=models, context_mode=context_mode)
            summary.append((title, got.get("computed", 0),
                            got.get("failed", 0), None))
        except Exception as e:
            print(f"CORPUS FAILED {title}: {type(e).__name__}: {e}")
            summary.append((title, 0, 0, f"{type(e).__name__}: {e}"))

    print(f"\n{'=' * 70}\nsummary")
    print(f"  {'corpus':<30} {'computed':>9} {'failed':>7}")
    for title, done, failed, err in summary:
        print(f"  {title:<30} {done:>9} {failed:>7}"
              + (f"   ERROR {err}" if err else ""))
    if any(f for _, _, f, _ in summary) or any(e for *_, e in summary):
        print("  a clean exit is not a clean run: check the lines above")
    return summary


def verify(device=None, n_clips=3, context=CONTEXT):
    """Run BEFORE the full run. Three things, in the order they can break.

    1. do the predictor weights match the checkpoint (strict=False can leave
       them random);
    2. does the trained predictor beat copying the previous moment (a loaded-
       as-noise predictor will not);
    3. does a shuffled clip surprise it more than its clean original.

    (3) is a sanity check on the pipeline, NOT the result and NOT a selection
    criterion -- the config is pinned in this file before any of it runs.
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    doc = E.load_split()
    processor, encoder, predictor = load_models(device)

    ho = E.held_out_splits(doc)
    stems = [s for s in sorted(doc["clips"])
             if doc["clips"][s]["split"] in ho][:n_clips]
    print(f"\nchecking {len(stems)} held-out clips (context={context})")
    print("  both context modes are run so the faithful one can be shown to "
          "work before a\n  long run commits to it, and so the leak in 'full' "
          "is a measured number.")
    print(f"\n  {'clip':<24} {'mode':<7} {'pred':>9} {'copy':>9} {'ratio':>7} "
          f"{'shuffle':>9} {'d':>9}")

    ok, usable = True, {}
    for stem in stems:
        try:
            clean = E.read_clip(E.download_video(E.video_key(doc, stem, "clean")))
            shuf = E.read_clip(E.download_video(
                E.video_key(doc, stem, "shuffle")))
            base = copy_baseline(processor, encoder, clean, device, context)
        except Exception as e:
            print(f"  {stem[:24]:<24} FAILED loading: {type(e).__name__}: {e}")
            ok = False
            continue

        for mode in ("masked", "full"):
            try:
                e_clean, _, _ = predict_errors(processor, encoder, predictor,
                                               clean, device, context,
                                               context_mode=mode)
                e_shuf, _, _ = predict_errors(processor, encoder, predictor,
                                              shuf, device, context,
                                              context_mode=mode)
            except Exception as e:
                print(f"  {stem[:24]:<24} {mode:<7} FAILED "
                      f"{type(e).__name__}: {e}")
                usable.setdefault(mode, []).append(False)
                continue
            r = float(e_clean.mean() / base.mean())
            usable.setdefault(mode, []).append(r < 1.0)
            print(f"  {stem[:24]:<24} {mode:<7} {e_clean.mean():>9.4f} "
                  f"{base.mean():>9.4f} {r:>7.3f} {e_shuf.mean():>9.4f} "
                  f"{float(e_shuf.mean() - e_clean.mean()):>+9.4f}")

    print()
    for mode in ("masked", "full"):
        got = usable.get(mode, [])
        state = ("beats the copy baseline on all "
                 f"{len(got)}" if got and all(got) else "DID NOT WORK")
        print(f"  {mode:<7} {state}")
    good = bool(usable.get(CONTEXT_MODE)) and all(usable.get(CONTEXT_MODE, []))
    ok = ok and good

    if not good:
        other = "full" if CONTEXT_MODE == "masked" else "masked"
        print(f"\n  PROBLEM: the default mode ({CONTEXT_MODE}) is not usable.")
        if usable.get(other) and all(usable[other]):
            print(f"  {other} works -- rerun with --context-mode {other} and "
                  "record the deviation.")
        else:
            print("  Neither mode beats copying the previous moment. Either "
                  "the predictor weights\n  did not load, or the causal masks "
                  "are not addressing moments -- check\n  embed_vjepa.py "
                  "--verify-axis before anything else.")

    print("\n  the shuffle column is a pipeline sanity check, NOT the result "
          "and not a\n  selection criterion: the config is pinned at the top "
          "of this file before any\n  of it runs.")
    return ok


# ------------------------------------------------------------------ report

def _boot_ci(x, n_boot=N_BOOT, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return tuple(np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def _auc(pos, neg):
    """Tie-corrected, same midrank convention as everywhere else here."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa, i = a[order], 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def collect(doc=None, stat="mean"):
    """-> ({split: {stem: {variant: value}}}, threshold). One GET per clip."""
    doc = doc or E.load_split()
    ho = E.held_out_splits(doc)
    stems = [s for s in sorted(doc["clips"])
             if doc["clips"][s]["split"] in ("train",) + tuple(ho)]

    raw = {}
    with ThreadPoolExecutor(max_workers=E.CACHE_READ_WORKERS) as pool:
        for stem, data in zip(stems, pool.map(read_cached, stems)):
            if data:
                raw[stem] = data

    train_err = np.concatenate(
        [raw[s]["err_clean"] for s in raw
         if doc["clips"][s]["split"] == "train" and "err_clean" in raw[s]]
        or [np.zeros(0)])
    if not len(train_err):
        raise RuntimeError("no clean TRAIN error sequences cached; the spike "
                           "threshold is defined only from those")
    threshold = float(np.percentile(train_err, SPIKE_PERCENTILE))
    print(f"spike threshold = P{SPIKE_PERCENTILE:.0f} of "
          f"{len(train_err)} clean TRAIN moments = {threshold:.4f}")

    out = defaultdict(dict)
    for stem, data in raw.items():
        split = doc["clips"][stem]["split"]
        per = {}
        for k in data:
            if not k.startswith("err_"):
                continue
            per[k[4:]] = summarise(data[k], threshold)[stat]
        if per:
            out[split][stem] = per
    return dict(out), threshold


def report(doc=None, stat="mean", push_to_s3=True):
    doc = doc or E.load_split()
    per_split, threshold = collect(doc, stat)
    ho = [s for s in E.held_out_splits(doc) if s in per_split]
    clips = {}
    for s in ho:
        clips.update(per_split[s])
    if not clips:
        raise RuntimeError("no held-out clips cached; nothing to compare")

    print(f"\n=== latent surprise ({stat} of the error sequence), "
          f"{len(clips)} held-out clips ===")
    names = sorted({v for c in clips.values() for v in c if v != "clean"})
    rows = {}
    for name in names:
        pairs = [(c["clean"], c[name]) for c in clips.values()
                 if "clean" in c and name in c]
        if not pairs:
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        d = b - a
        lo, hi = _boot_ci(d)
        rows[name] = {
            "kind": "temporal" if name in TEMPORAL_VARIANTS else "superficial",
            "n": len(d), "clean": float(a.mean()), "attacked": float(b.mean()),
            "d": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "auc": _auc(b, a),
        }

    w = 34
    print(f"  {'variant':<{w}} {'n':>5} {'clean':>9} {'attacked':>9} "
          f"{'d':>9} {'95% CI':>21} {'AUC':>7}")
    for kind in ("temporal", "superficial"):
        want = ("should RISE -- the predictor is surprised" if kind == "temporal"
                else "should not move -- dynamics are untouched")
        print(f"  -- expected-{'sensitivity' if kind == 'temporal' else 'invariance'}"
              f"  ({want})")
        for name in sorted(n for n in rows if rows[n]["kind"] == kind):
            r = rows[name]
            flag = ""
            if kind == "temporal" and r["lo"] > 0:
                flag = "  <-- DETECTED"
            elif kind == "superficial" and r["lo"] > 0:
                flag = "  <-- moved on a superficial cue"
            print(f"  {name:<{w}} {r['n']:>5} {r['clean']:>9.4f} "
                  f"{r['attacked']:>9.4f} {r['d']:>+9.4f} "
                  f"[{r['lo']:>+9.4f},{r['hi']:>+9.4f}] {r['auc']:>7.3f}{flag}")

    print("\n  AUC separates clean from attacked on this statistic alone; 0.5 "
          "is chance.")
    print("  Read the temporal rows against the superficial ones, not against "
          "zero -- an\n  absolute error under a context window this short is "
          "off-distribution, and only\n  the within-clip difference cancels "
          "that offset.")
    print("  These are NOT on the PC probe's scale and must not be differenced "
          "against it.")

    payload = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "stat": stat, "context": CONTEXT, "mask_index": MASK_INDEX,
               "spike_percentile": SPIKE_PERCENTILE, "threshold": threshold,
               "n_clips": len(clips), "variants": rows}
    if push_to_s3:
        E._ensure_s3().put_object(Bucket=BUCKET, Key=REPORT_KEY,
                                  Body=json.dumps(payload, indent=2).encode())
        print(f"\nuploaded -> s3://{BUCKET}/{REPORT_KEY}")
    return payload


# ---------------------------------------------------------------- selftest

def selftest():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}"
              + (f"   {detail}" if detail else ""))

    print("causal masks")
    import torch
    mx, my = causal_masks(context=1, t=4, s=3)
    check("one row per predicted moment", mx.shape == (3, 3) and my.shape == (3, 3),
          f"{tuple(mx.shape)} {tuple(my.shape)}")
    check("row 0 predicts moment 1 from moment 0",
          mx[0].tolist() == [0, 1, 2] and my[0].tolist() == [3, 4, 5])
    check("context always precedes its target",
          bool((mx.max(dim=1).values < my.min(dim=1).values).all()))
    check("no index appears in both context and target of a row",
          all(not (set(mx[i].tolist()) & set(my[i].tolist()))
              for i in range(len(mx))))
    mx2, _ = causal_masks(context=2, t=4, s=3)
    check("a wider context widens the rows and drops the first moment",
          mx2.shape == (2, 6), f"{tuple(mx2.shape)}")
    check("moment i owns indices [i*s, (i+1)*s)",
          causal_masks(1, 4, 3)[1][2].tolist() == [9, 10, 11])

    print("summary statistics")
    s = summarise(np.array([1.0, 2.0, 3.0, 4.0]), threshold=2.5)
    check("mean/max are the sequence's", s["mean"] == 2.5 and s["max"] == 4.0)
    check("spike rate counts moments above the threshold",
          s["spike_rate"] == 0.5, str(s["spike_rate"]))
    check("no threshold gives NaN, not a silent zero",
          summarise(np.array([1.0, 2.0]))["spike_rate"] != summarise(
              np.array([1.0, 2.0]))["spike_rate"])
    check("a flat sequence has zero spread",
          summarise(np.array([2.0, 2.0, 2.0]))["std"] == 0.0)

    print("AUC")
    check("cleanly separated is 1.0", _auc([5, 6, 7], [1, 2, 3]) == 1.0)
    check("all tied is 0.5", _auc([1, 1], [1, 1]) == 0.5)
    check("reversed is 0.0", _auc([1, 2], [5, 6]) == 0.0)

    print("bootstrap")
    lo, hi = _boot_ci(np.full(50, 0.3))
    check("a constant sample has a degenerate CI at its value",
          abs(lo - 0.3) < 1e-9 and abs(hi - 0.3) < 1e-9)
    lo, hi = _boot_ci(np.random.default_rng(0).normal(1.0, 0.1, 400))
    check("a positive effect gives a CI above zero", lo > 0, f"[{lo:.3f},{hi:.3f}]")

    print("context mode")
    for bad in ("", "none", "Masked"):
        try:
            _encode_context(None, None, None, bad)
            check(f"an unknown context_mode {bad!r} raises", False)
        except ValueError:
            check(f"an unknown context_mode {bad!r} raises", True)
        except Exception as e:
            check(f"an unknown context_mode {bad!r} raises",
                  False, type(e).__name__)
    check("the default is the faithful mode", CONTEXT_MODE == "masked",
          CONTEXT_MODE)

    print("corpus selection")
    doc = {"superficial_variants": ["photometric", "caption_echo_rubric_vocab"],
           "held_out_splits": ["val", "cal"], "val_temporal_variants":
           ["shuffle", "reverse", "freeze"]}
    check("train contributes clean only",
          variants_for({"split": "train"}, doc, has_train=True) == ["clean"])
    v = variants_for({"split": "val"}, doc, has_train=True)
    check("held-out contributes clean plus every variant", len(v) == 6, str(len(v)))
    check("held-out includes the temporal variants",
          set(TEMPORAL_VARIANTS) <= set(v))
    check("an unknown split of a SPLIT doc contributes nothing",
          variants_for({"split": "other"}, doc, has_train=True) == [])
    # the branch that would otherwise make a benchmark run silently compute
    # nothing: those clips carry split == the dataset name
    check("a benchmark clip gets the full variant set",
          len(variants_for({"split": "videophy2_test"}, doc,
                           has_train=False)) == 6)
    check("has_train_split distinguishes the two doc shapes",
          has_train_split({"clips": {"a": {"split": "train"}}})
          and not has_train_split({"clips": {"a": {"split": "videophy2_test"}}}))

    print("-" * 68)
    print("selftest OK" if ok else "selftest FAILED")
    return ok


if __name__ == "__main__" and not in_notebook():
    ap = argparse.ArgumentParser(description="V-JEPA predictor latent surprise")
    ap.add_argument("--verify", action="store_true",
                    help="weights + copy baseline + a shuffled clip; run first")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--context", type=int, default=CONTEXT)
    ap.add_argument("--context-mode", choices=["masked", "full"],
                    default=CONTEXT_MODE,
                    help="masked: encode the context from only the patches "
                         "it may see (faithful). full: gather from a "
                         "whole-clip pass (cheaper, leaks the future)")
    ap.add_argument("--stat", default="mean",
                    choices=["mean", "max", "std", "p90", "p95", "spike_rate"])
    ap.add_argument("--no-tokens", action="store_true",
                    help="skip the per-token errors (~35 KB/clip)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--datasets", nargs="+", default=["train"],
                    help="train (the doc's Step 10 corpus) | all | a "
                         "benchmark corpus name")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if a.verify:
        raise SystemExit(0 if verify(device=a.device, context=a.context) else 1)
    if a.report:
        report(stat=a.stat, push_to_s3=not a.no_push)
    else:
        run_all(datasets=a.datasets, device=a.device, dry_run=a.dry_run,
                context=a.context, store_tokens=not a.no_tokens,
                limit=a.limit, context_mode=a.context_mode)
