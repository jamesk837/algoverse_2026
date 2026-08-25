#!/usr/bin/env bash
# Run every judge over this box's stripe of the clip list.
#
#   ./run_shard.sh <shard_index> [shard_count]
#
# One box per shard index, 0 .. count-1, every box given the same count. The
# stripes are disjoint and checkpointing is per (clip, variant, call) in S3, so
# the boxes need no coordination and a rerun resumes wherever it stopped.
#
# Judges run one after another, never at once: each needs its own venv (their
# upstream pins are mutually exclusive) and its own process (a second model load
# in a live interpreter is what produces the silent meta-device corruption).
#
# env overrides:
#   N=590            clip count (default: ask S3)
#   DATASET=test     test | implausibench_real | implausibench_implausible
#   JUDGES="..."     space-separated subset, in order
#   VENVS=$HOME/venvs
#   LOGS=$HOME/logs
#   SKIP_SETUP=1     assume the venvs exist instead of building missing ones
#   PYTHON=python3.12  interpreter the venvs are built from; the pins need
#                    3.10-3.12, so this is required wherever python3 is newer
#
# Smoke test -- one clip through all three judges, written to S3 so it can be
# inspected from Colab with check_results.py:
#   N=1 ./run_shard.sh 0 1
set -uo pipefail

cd "$(dirname "$0")"

IDX="${1:-}"
COUNT="${2:-8}"
DATASET="${DATASET:-test}"
JUDGES="${JUDGES:-phyjudge_9b vila_ewm videophy2_auto}"
VENVS="${VENVS:-$HOME/venvs}"
LOGS="${LOGS:-$HOME/logs}"

# venv per judge; the names match the setup_ec2_*.sh scripts
venv_for() {
  case "$1" in
    phyjudge_9b)    echo "$VENVS/phyjudge" ;;
    vila_ewm)       echo "$VENVS/vila" ;;
    videophy2_auto) echo "$VENVS/videophy2" ;;
    *) return 1 ;;
  esac
}

setup_for() {
  case "$1" in
    phyjudge_9b)    echo "./setup_ec2_phyjudge.sh" ;;
    vila_ewm)       echo "./setup_ec2_vila.sh" ;;
    videophy2_auto) echo "./setup_ec2_videophy2.sh" ;;
    *) return 1 ;;
  esac
}

die() { echo "ERROR: $*" >&2; exit 1; }

[ -n "$IDX" ] || die "usage: $0 <shard_index> [shard_count]"
case "$IDX$COUNT" in *[!0-9]*) die "shard index and count must be integers" ;; esac
[ "$IDX" -lt "$COUNT" ] || die "shard index $IDX must be < count $COUNT"

# ---- preflight -------------------------------------------------------------
# Every check here guards a failure that otherwise shows up hours in, or worse,
# not at all.
for judge in $JUDGES; do
  venv_for "$judge" >/dev/null || die "unknown judge: $judge"
done

# All venvs are built before any judge runs. Lazily building each one just
# before its judge would hide a broken vila setup behind phyjudge's ~40 hours;
# ten minutes up front buys fail-fast on all three.
# A venv DIRECTORY existing is not the same as a venv being usable, and
# treating it as such has now failed twice: once on venvs left half-built by a
# crashed setup, and once on venvs created against Python 3.14 before the
# interpreter problem was understood. Both had bin/python and neither could run
# a judge -- the second surfaced as a confusing "no AWS credentials" much later,
# because the preflight ran boto3 out of a venv that had never been populated.
# So: check the interpreter version, and check a package every setup script
# installs actually imports.
venv_healthy() {
  local venv="$1"
  [ -x "$venv/bin/python" ] || return 1
  "$venv/bin/python" - <<'HEALTH' >/dev/null 2>&1 || return 1
import sys
assert (3, 10) <= sys.version_info[:2] <= (3, 12), sys.version
import boto3   # every setup_ec2_*.sh installs it; absent => setup never finished
HEALTH
}

for judge in $JUDGES; do
  venv="$(venv_for "$judge")"
  if venv_healthy "$venv"; then
    echo "venv ok: $venv"
    continue
  fi
  [ -z "${SKIP_SETUP:-}" ] || die "venv at $venv is missing or unusable, and SKIP_SETUP is set"
  script="$(setup_for "$judge")"
  [ -x "$script" ] || die "$script missing or not executable"
  if [ -d "$venv" ]; then
    echo "venv at $venv is unusable (wrong Python, or setup never finished) -- rebuilding"
    rm -rf "$venv"
  fi
  echo "building venv for $judge ($script) ..."
  VENV="$venv" "$script" || die "$script failed -- fix it before running $judge"
  venv_healthy "$venv" || die "$script finished but $venv is still not usable"
done

case "$JUDGES" in
  *phyjudge_9b*)
    # phyjudge resolves its base model from the Hub, not S3
    [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN unset; phyjudge_9b cannot pull its base model"
    ;;
esac

FIRST_VENV="$(venv_for "${JUDGES%% *}")"
"$FIRST_VENV/bin/python" -c "
import boto3
boto3.client('sts', region_name='us-east-1').get_caller_identity()" \
  || die "no AWS credentials -- is the IAM instance profile attached?"

if [ -z "${N:-}" ]; then
  echo "counting clips in $DATASET ..."
  N="$("$FIRST_VENV/bin/python" -c "
from judge_harness import list_source_videos
print(len(list_source_videos('$DATASET')))")" || die "could not list source videos"
fi
case "$N" in ''|*[!0-9]*) die "bad clip count: '$N'" ;; esac

mkdir -p "$LOGS"
echo "======================================================================"
echo "shard $IDX/$COUNT   dataset=$DATASET   clips=$N"
echo "judges: $JUDGES"
echo "logs:   $LOGS"
echo "======================================================================"

# ---- run -------------------------------------------------------------------
run_start=$(date +%s)
status=0
total_failed=0

for judge in $JUDGES; do
  venv="$(venv_for "$judge")"
  log="$LOGS/${judge}_${IDX}.log"
  echo
  echo "---- $judge  ($(date '+%F %T'))  -> $log"
  t0=$(date +%s)

  # -u or a detached log stays empty until the buffer flushes
  "$venv/bin/python" -u -c "
from judge_harness import run_judges
run_judges(dataset='$DATASET', num_clips=$N, models=['$judge'], shard=($IDX, $COUNT))
" 2>&1 | tee "$log"
  rc="${PIPESTATUS[0]}"

  secs=$(( $(date +%s) - t0 ))
  printf -- '---- %s finished in %dh%02dm (exit %s)\n' \
    "$judge" $((secs/3600)) $(((secs%3600)/60)) "$rc"

  # A clean exit does not mean clean results: per-clip and per-call failures are
  # caught and printed, and the meta-device warning is not an error at all.
  if [ "$rc" -ne 0 ]; then
    echo "     WARN non-zero exit"
    status=1
  fi
  # run_judges catches a load failure, prints "[judge] SKIPPED: <reason>" and
  # returns normally -- so the exit code is 0, no FAILED line is emitted, and a
  # judge that scored nothing at all would otherwise be reported as fine. On an
  # unattended multi-day fleet run that is the worst possible silent failure.
  if grep -q "SKIPPED:" "$log"; then
    echo "     DID NOT RUN: the judge failed to load and scored nothing:"
    sed -n 's/.*SKIPPED: */       /p' "$log" | head -3
    status=1
  fi
  if grep -q "meta device" "$log"; then
    echo "     VOID: 'meta device' in the log -- the adapter did not load and"
    echo "           these scores came from the bare base model. Delete them:"
    echo "           aws s3 rm --recursive s3://nickb-aarj/results/pass1/$judge/$DATASET/"
    status=1
  fi
  # FAILED lines are tolerable -- a variant that was never rendered, a call
  # that raised -- and the run is resumable, so they are reported but do not
  # fail the script. They must never be silently folded into "no problems".
  n_failed="$(grep -c FAILED "$log" || true)"
  if [ "$n_failed" -ne 0 ]; then
    echo "     $n_failed FAILED lines -- grep the log"
    total_failed=$(( total_failed + n_failed ))
  fi
done

total=$(( $(date +%s) - run_start ))
echo
printf 'shard %s/%s done in %dh%02dm\n' "$IDX" "$COUNT" $((total/3600)) $(((total%3600)/60))
if [ "$status" -ne 0 ]; then
  echo "REVIEW THE WARNINGS ABOVE"
elif [ "$total_failed" -ne 0 ]; then
  echo "completed, but $total_failed FAILED lines across the logs -- review them,"
  echo "then rerun this same command to retry only what is still missing"
else
  echo "no problems detected"
fi
exit "$status"
