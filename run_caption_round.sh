#!/usr/bin/env bash
# One caption-search round, unattended: render the clips, then score them.
#
#   ./run_caption_round.sh <round>      # e.g. 0, 1, 2
#   ./run_caption_round.sh final        # the winners, on the eval subset
#   ./run_caption_round.sh ladder       # overlay experiment A -- MECHANISM
#   ./run_caption_round.sh robust "<winning caption>"   # experiment B
#
# `ladder` and `robust` are different experiments (A: blank box / random chars
# / georgian / nonsense; B: one caption crossed over placement x opacity x
# size) but both are the same render-then-judge shape on the same pass-2
# subset and the same venv preflight, so they live here rather than in a
# second script with a second copy of the health checks.
#
# `robust` takes the winning caption as its second argument the FIRST time and
# pins it to overlay_robust_active.json; later runs re-read the pin and the
# argument becomes optional. Passing a caption that differs from the pinned
# one is an error rather than a silent repin: renders are keyed by name and
# are never re-made, so a grid half-rendered under two captions cannot be
# untangled afterwards.
#
# SMOKE=<n> does the whole thing on n clips per dataset (default 2) -- the
# same code path end to end, minutes instead of hours.
#
# The two halves need different environments -- rendering wants ffmpeg and
# opencv, scoring wants one of three mutually-pinned judge venvs -- which is
# why they were two commands. This chains them: render (CPU, ~5 min for round
# 0's 600 clips), then hand the rendered variant names straight to
# run_shard.sh (GPU, hours). Nobody retypes a content hash.
#
# Both halves are resumable. Rendering skips any S3 key that exists and
# scoring checkpoints per (clip, variant, call), so re-running after an
# interruption picks up where it stopped and costs nothing for what is done.
#
# env overrides (anything not listed is passed through to run_shard.sh):
#   RENDER_VENV=$HOME/venvs/render   built automatically if missing
#   RENDER_WORKERS=8
#   PYTHON=python3.12   interpreter the venvs are built from; the judge pins
#                       need 3.10-3.12, so this is required wherever python3
#                       is newer (the DLAMI on Ubuntu 26.04 ships 3.14)
#   JUDGES="..."        subset, in order. Default all three.
#   SKIP_RENDER=1       score what is already rendered
#   SKIP_JUDGE=1        render only, print what would be scored
#   JUDGES="..."        subset of judges, in order. Default all three.
#   LOGS=$HOME/logs
set -uo pipefail
cd "$(dirname "$0")"

ROUND="${1:-}"
CAPTION_ARG="${2:-${OVERLAY_CAPTION:-}}"
[ -n "$ROUND" ] \
  || { echo "usage: $0 <round|final|ladder|robust> [caption]" >&2; exit 1; }
case "$ROUND" in
  final|ladder|robust|[0-9]|[0-9][0-9]) ;;
  *) echo "ERROR: target must be an integer, 'final', 'ladder' or 'robust', got '$ROUND'" >&2
     exit 1 ;;
esac

# SMOKE=<n> renders and scores only the first n clips per dataset (SMOKE=1
# means 2, which is the useful floor -- one clip cannot show a spread). The
# judge half needs no matching cap: naming the variants routes through
# clips_with_variants, which requires each one to be RENDERED, so it can only
# ever see the clips the smoke render produced.
SMOKE_N="${SMOKE:-}"
[ "$SMOKE_N" = "1" ] && SMOKE_N=2
case "$SMOKE_N" in
  ''|*[!0-9]*) [ -z "$SMOKE_N" ] || { echo "SMOKE must be a number" >&2; exit 1; } ;;
esac
export SMOKE_N
[ -z "$SMOKE_N" ] || echo "SMOKE MODE: $SMOKE_N clip(s) per dataset"

RENDER_VENV="${RENDER_VENV:-$HOME/venvs/render}"
RENDER_WORKERS="${RENDER_WORKERS:-8}"
LOGS="${LOGS:-$HOME/logs}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$LOGS"

die() { echo "ERROR: $*" >&2; exit 1; }

case "$ROUND" in
  ladder|robust) ;;   # the overlay experiments do not read the search pool
  *) [ -f caption_pool.json ] \
       || die "caption_pool.json missing -- run: python caption_search.py --init" ;;
esac

# ---- render venv -----------------------------------------------------------
# Deliberately its own venv rather than borrowing a judge's. Only vila installs
# opencv, and building vila to render a text overlay would pull torch and take
# ten minutes; this is boto3 + opencv-headless + numpy and takes about thirty
# seconds. It also keeps ffmpeg work off the venvs whose pins are load-bearing.
#
# A venv DIRECTORY existing is not a venv working -- that has bitten this repo
# twice, once on half-built venvs and once on venvs created against Python
# 3.14. Check the interpreter version and that the imports actually resolve.
render_venv_healthy() {
  [ -x "$RENDER_VENV/bin/python" ] || return 1
  "$RENDER_VENV/bin/python" - <<'HEALTH' >/dev/null 2>&1 || return 1
import sys
assert (3, 10) <= sys.version_info[:2] <= (3, 12), sys.version
import boto3, cv2, numpy
HEALTH
}

if render_venv_healthy; then
  echo "render venv ok: $RENDER_VENV"
else
  [ -z "${SKIP_RENDER:-}" ] || die "SKIP_RENDER is set but $RENDER_VENV is unusable"
  if [ -d "$RENDER_VENV" ]; then
    echo "render venv at $RENDER_VENV is unusable -- rebuilding"
    rm -rf "$RENDER_VENV"
  fi
  echo "building render venv ($PYTHON) ..."
  "$PYTHON" -c "
import sys
assert (3,10) <= sys.version_info[:2] <= (3,12), sys.version" \
    || die "$PYTHON is out of range; the judge venvs need 3.10-3.12 too. Try: PYTHON=\$(~/.local/bin/uv python find 3.12) $0 $ROUND"
  "$PYTHON" -m venv "$RENDER_VENV" || die "venv creation failed"
  "$RENDER_VENV/bin/pip" install -q --upgrade pip
  "$RENDER_VENV/bin/pip" install -q boto3 opencv-python-headless numpy \
    || die "render venv pip install failed"
  render_venv_healthy || die "render venv built but still unusable"
fi

command -v ffmpeg >/dev/null || die "ffmpeg not on PATH (apt install ffmpeg)"
[ -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ] \
  || echo "  WARN DejaVuSans.ttf not at the pinned path -- drawtext will fail"

"$RENDER_VENV/bin/python" -c "
import boto3
boto3.client('sts', region_name='us-east-1').get_caller_identity()" >/dev/null \
  || die "no AWS credentials -- is the IAM instance profile attached?"

# ---- pin the experiment-B caption ------------------------------------------
# Before the render, so a missing or contradictory caption costs nothing. The
# pin FILE is what the render workers read: a module global would not survive a
# ProcessPoolExecutor worker that spawns rather than forks.
if [ "$ROUND" = "robust" ]; then
  OVERLAY_CAPTION="$CAPTION_ARG" FORCE_PIN="${FORCE_PIN:-}" \
    "$RENDER_VENV/bin/python" -u -c '
import json, os
import attack_suite as A

text = os.environ.get("OVERLAY_CAPTION") or ""
path = A.OVERLAY_ROBUST_PATH
if os.path.exists(path) and not os.environ.get("FORCE_PIN"):
    with open(path, encoding="utf-8") as fh:
        cur = json.load(fh)
    if text and cur["text"] != text:
        raise SystemExit(
            "ERROR: %s already pins %r as %s.\n"
            "Re-run with no caption argument to use it, or set FORCE_PIN=1 "
            "ONLY if nothing has been rendered under that label yet."
            % (path, cur["text"], cur["label"]))
    print("pinned already: %r  (%s, %d cells)"
          % (cur["text"], cur["label"], len(cur["variants"])))
elif text:
    grid = A.set_robustness_winner(text, force=True, verbose=False)
    print("pinned: %r  (%d cells) -> %s" % (text, len(grid), path))
else:
    raise SystemExit(
        "ERROR: no caption pinned and none given. Pass the selection loop "
        "winner as the second argument.")' \
    || die "could not pin the experiment-B caption"
fi

# ---- render ----------------------------------------------------------------
RLOG="$LOGS/caption_render_${ROUND}.log"
if [ -n "${SKIP_RENDER:-}" ]; then
  echo "SKIP_RENDER set -- not rendering"
else
  echo "======================================================================"
  echo "RENDER  round $ROUND  ->  $RLOG"
  echo "======================================================================"
  t0=$(date +%s)
  if [ "$ROUND" = "ladder" ]; then
    # the mechanism arms, on the pass-2 subset. check_font first: a script the
    # font cannot draw renders as tofu boxes and uploads happily, which would
    # silently collapse the georgian rung onto the blank rung next to it.
    "$RENDER_VENV/bin/python" -u -c "
from attack_suite import run_suite, pass2_stems, check_font
bad = [n for n, ok in check_font().items() if ok is False]
if bad:
    raise SystemExit('font cannot draw: %s' % ', '.join(bad))
for ds in ('test', 'implausibench_implausible', 'implausibench_real'):
    run_suite(dataset=ds, limit_clips=None, num_workers=$RENDER_WORKERS,
              attacks='overlay', only_stems=pass2_stems(ds))" 2>&1 | tee "$RLOG"
  elif [ "$ROUND" = "robust" ]; then
    # experiment B, on the same pass-2 subset as the ladder. check_font covers
    # the pinned grid as well as the static arms, so a caption the font cannot
    # draw is caught here instead of discovered as 2040 boxes.
    "$RENDER_VENV/bin/python" -u -c "
import os
from attack_suite import run_suite, pass2_stems, check_font
bad = [n for n, ok in check_font(verbose=False).items() if ok is False]
if bad:
    raise SystemExit('font cannot draw: %s' % ', '.join(bad))
n = int(os.environ.get('SMOKE_N') or 0)
for ds in ('test', 'implausibench_implausible', 'implausibench_real'):
    stems = sorted(pass2_stems(ds))
    if n:
        stems = stems[:n]
    run_suite(dataset=ds, limit_clips=None, num_workers=$RENDER_WORKERS,
              attacks='overlay_robust', only_stems=stems)" 2>&1 | tee "$RLOG"
  elif [ "$ROUND" = "final" ]; then
    "$RENDER_VENV/bin/python" -u caption_search.py --finalize \
      --workers "$RENDER_WORKERS" 2>&1 | tee "$RLOG"
  else
    "$RENDER_VENV/bin/python" -u caption_search.py --render "$ROUND" \
      --workers "$RENDER_WORKERS" 2>&1 | tee "$RLOG"
  fi
  rc="${PIPESTATUS[0]}"
  printf -- 'render finished in %dm%02ds (exit %s)\n' \
    $(( ($(date +%s) - t0) / 60 )) $(( ($(date +%s) - t0) % 60 )) "$rc"
  [ "$rc" -eq 0 ] || die "render failed -- not starting the judges"
  # run_suite catches per-attack and per-clip failures and keeps going, so a
  # zero exit does not mean everything rendered. A clip missing its variant is
  # then dropped by require_attacks and silently shrinks the experiment.
  n_failed="$(grep -c '^FAILED\|FAILED ' "$RLOG" || true)"
  if [ "$n_failed" -ne 0 ]; then
    echo "  WARN $n_failed FAILED line(s) in the render log; those clips will be"
    echo "       dropped by require_attacks. grep FAILED $RLOG"
  fi
fi

# ---- hand off --------------------------------------------------------------
# The render step tells the judge step what it produced. Content hashes are not
# something to retype, and N has to be the PER-DATASET count because run_shard
# loops datasets and passes num_clips to each separately.
if [ "$ROUND" = "ladder" ]; then
  # Derived from attack_suite rather than spelled out here: the arm names are
  # already mirrored in two places (attack_suite and judge_harness) and a
  # third hand-maintained copy is how they drift apart.
  ENVOUT="$("$RENDER_VENV/bin/python" -c "
import attack_suite as A
names = sorted(A.attack_filenames(A.attack_set('overlay')))
dss = ('test', 'implausibench_real', 'implausibench_implausible')
stems = {d: A.pass2_stems(d) for d in dss}
print('CAPTION_LABEL=\"overlay mechanism ladder\"')
print('CAPTION_VARIANTS=\"%s\"' % ' '.join(['clean'] + names))
print('CAPTION_N=%d' % max(len(v) for v in stems.values()))
print('CAPTION_NVARIANTS=%d' % len(names))
print('CAPTION_TOTAL_CLIPS=%d' % sum(len(v) for v in stems.values()))")"
elif [ "$ROUND" = "robust" ]; then
  # Derived from the pin file rather than spelled out here: the cell names
  # embed a hash of the caption, which is exactly what should never be retyped.
  ENVOUT="$("$RENDER_VENV/bin/python" -c "
import os
import attack_suite as A
names = sorted(A.robustness_pool(reload=True))
if not names:
    raise SystemExit(1)
dss = ('test', 'implausibench_real', 'implausibench_implausible')
n = int(os.environ.get('SMOKE_N') or 0)
stems = {d: (sorted(A.pass2_stems(d))[:n] if n else A.pass2_stems(d)) for d in dss}
print('CAPTION_LABEL=\"overlay presentation robustness\"')
print('CAPTION_VARIANTS=\"%s\"' % ' '.join(['clean'] + names))
print('CAPTION_N=%d' % max(len(v) for v in stems.values()))
print('CAPTION_NVARIANTS=%d' % len(names))
print('CAPTION_TOTAL_CLIPS=%d' % sum(len(v) for v in stems.values()))")"
else
  ENVOUT="$("$RENDER_VENV/bin/python" caption_search.py --emit-env "$ROUND")"
fi
[ -n "$ENVOUT" ] || die "could not work out what to score"
eval "$ENVOUT"
[ -n "${CAPTION_VARIANTS:-}" ] || die "--emit-env produced no variants"
[ "${CAPTION_NVARIANTS:-0}" -gt 0 ] \
  || die "no variants for '$ROUND' -- is that round in caption_pool.json?"

echo
echo "======================================================================"
echo "SCORE   $CAPTION_LABEL"
echo "  variants: $CAPTION_NVARIANTS + clean"
echo "  clips:    $CAPTION_TOTAL_CLIPS total, cap $CAPTION_N per dataset"
echo "  est:      $(( CAPTION_NVARIANTS * CAPTION_TOTAL_CLIPS * 26 )) generations"
echo "======================================================================"

if [ -n "${SKIP_JUDGE:-}" ]; then
  echo "SKIP_JUDGE set -- would have run:"
  echo "  VARIANTS=\"$CAPTION_VARIANTS\" N=$CAPTION_N ./run_shard.sh 0 1"
  exit 0
fi

# Pass 1 mode: no PARAPHRASE, no pass2. The searched captions are read against
# the clean and caption_echo rows already in results/pass1.
VARIANTS="$CAPTION_VARIANTS" N="$CAPTION_N" ./run_shard.sh 0 1
rc=$?

echo
if [ "$rc" -eq 0 ]; then
  echo "$ROUND done. Next:"
  if [ "$ROUND" = "final" ]; then
    echo "  python caption_search.py --report --push"
  elif [ "$ROUND" = "ladder" ] || [ "$ROUND" = "robust" ]; then
    echo "  python analysis/analyze.py     # prints the overlay ablation tables"
  else
    echo "  python caption_search.py --rank $ROUND"
    echo "  ...then add round $((ROUND + 1)) to caption_pool.json and re-run this."
  fi
else
  echo "run_shard.sh exited $rc -- check the logs under $LOGS before ranking."
fi
exit "$rc"
