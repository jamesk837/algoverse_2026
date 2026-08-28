#!/usr/bin/env bash
# Stop a judging run on this box. Safe to interrupt at any moment.
#
#   ./stop_runs.sh              # graceful: TERM, wait, then KILL what is left
#   ./stop_runs.sh --dry-run    # list what would be stopped, kill nothing
#   ./stop_runs.sh --kill       # skip straight to KILL
#   GRACE=30 ./stop_runs.sh     # seconds to wait for a clean exit (default 15)
#
# WHY THE SHELL DIES FIRST
#
# run_shard.sh is a loop over judges x datasets that runs each one in the
# FOREGROUND. Killing the python child alone therefore does not stop the run --
# the shell simply reaps it and starts the next judge, which is worse than not
# stopping at all: you get a fresh model load on a box you meant to free. The
# wrapper is signalled first, then the python, then anything still breathing.
#
# WHY THIS CANNOT CORRUPT RESULTS
#
# Checkpointing is per (clip, variant, call): process_clip PUTs the merged
# record after each variant's calls, and an S3 PutObject is atomic -- a
# half-written object is not a state S3 can be left in. The most a kill can
# cost is the variant in flight, which the next run regenerates because its
# calls are simply absent from the record. Nothing needs cleaning up, and there
# is no such thing as a partial record that has to be deleted first.

set -uo pipefail

GRACE="${GRACE:-15}"
MODE="term"
for arg in "$@"; do
  case "$arg" in
    --dry-run) MODE="dry" ;;
    --kill)    MODE="kill" ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# The two patterns are deliberately separate so the wrapper can be signalled
# first. They match the full command line, which is how a `python -u -c`
# one-liner is identifiable at all -- there is no script name to match on.
WRAPPER_PAT='run_shard\.sh'
PYTHON_PAT='run_judges|judge_harness'

# /proc/<pid>/cmdline, not `ps`, and not pgrep. run_shard.sh launches python
# with a MULTI-LINE -c script:
#
#     "$venv/bin/python" -u -c "
#     from judge_harness import run_judges
#     run_judges(dataset='$ds', ...)
#     "
#
# so `ps -eo args=` prints a record spanning four lines, the pattern matches on
# a continuation line, and a naive `awk '{print $1}'` returns "from" and
# "run_judges(dataset='test'," instead of a pid -- while the real pid's own
# line, which ends at `-c`, matches nothing. cmdline is NUL-separated with no
# such ambiguity; flattening it gives one line per process, always.
PROC_ROOT="${PROC_ROOT:-/proc}"

list_pids() {
  local pat="$1" pid cmd
  for d in "$PROC_ROOT"/[0-9]*; do
    [ -r "$d/cmdline" ] || continue
    pid="${d##*/}"
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "${PPID:-0}" ] && continue
    cmd="$(tr '\0\n' '  ' < "$d/cmdline" 2>/dev/null)"
    [ -z "$cmd" ] && continue
    case "$cmd" in *stop_runs.sh*) continue ;; esac
    printf '%s\n' "$cmd" | grep -qE "$pat" && printf '%s\n' "$pid"
  done
  return 0
}

cmdline_of() {
  tr '\0\n' '  ' < "$PROC_ROOT/$1/cmdline" 2>/dev/null
}

show() {
  local label="$1" pids="$2" pid line
  [ -z "$pids" ] && return 0
  echo "  $label:"
  for pid in $pids; do
    line="$(cmdline_of "$pid")"
    echo "    pid $pid  ${line:0:140}"
  done
}

WRAPPERS="$(list_pids "$WRAPPER_PAT")"
PYTHONS="$(list_pids "$PYTHON_PAT")"

echo "=============================================================="
echo "judging processes on $(hostname)"
echo "=============================================================="
if [ -z "$WRAPPERS$PYTHONS" ]; then
  echo "  none running"
else
  show "run_shard.sh wrapper(s)" "$WRAPPERS"
  show "run_judges python process(es)" "$PYTHONS"
fi

if command -v tmux >/dev/null 2>&1; then
  sessions="$(tmux ls 2>/dev/null || true)"
  if [ -n "$sessions" ]; then
    echo
    echo "  tmux sessions (the processes above may live in one; this script"
    echo "  stops the processes, not the session):"
    echo "$sessions" | sed 's/^/    /'
  fi
fi

if [ -z "$WRAPPERS$PYTHONS" ]; then
  exit 0
fi

if [ "$MODE" = "dry" ]; then
  echo
  echo "--dry-run: nothing signalled."
  exit 0
fi

sig_all() {
  local sig="$1"
  # wrapper first: otherwise it reaps the child and launches the next judge
  for pids in "$WRAPPERS" "$PYTHONS"; do
    [ -z "$pids" ] && continue
    # shellcheck disable=SC2086
    kill "-$sig" $pids 2>/dev/null || true
  done
}

echo
if [ "$MODE" = "kill" ]; then
  echo "sending KILL ..."
  sig_all KILL
else
  echo "sending TERM (wrapper first), waiting up to ${GRACE}s ..."
  sig_all TERM
  for _ in $(seq "$GRACE"); do
    sleep 1
    left="$(list_pids "$WRAPPER_PAT")$(list_pids "$PYTHON_PAT")"
    [ -z "$left" ] && break
  done
fi

sleep 1
LEFT_W="$(list_pids "$WRAPPER_PAT")"
LEFT_P="$(list_pids "$PYTHON_PAT")"
if [ -n "$LEFT_W$LEFT_P" ]; then
  echo "still alive after ${GRACE}s -- sending KILL"
  WRAPPERS="$LEFT_W"; PYTHONS="$LEFT_P"
  sig_all KILL
  sleep 2
fi

STILL="$(list_pids "$WRAPPER_PAT")$(list_pids "$PYTHON_PAT")"
echo
if [ -n "$STILL" ]; then
  echo "WARNING processes survived KILL (uninterruptible sleep in a CUDA call?):"
  show "surviving" "$STILL"
  echo "  wait for the driver call to return, then re-run this script"
  exit 1
fi

echo "stopped."
echo
echo "Nothing to clean up: results are checkpointed per (clip, variant, call)"
echo "and every S3 PutObject is atomic, so at most the variant in flight was"
echo "lost. Re-running the same command later regenerates only what is absent."
if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "GPU after stopping:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
             --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  echo "  (memory still held means a process has not exited yet -- re-check)"
fi
