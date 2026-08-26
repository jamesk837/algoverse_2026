"""Reproduce WorldModelBench's own CoT invocation, so anyone can re-check it.

GROUND TRUTH -- WorldModelBench-Team/WorldModelBench, evaluation.py
  https://github.com/WorldModelBench-Team/WorldModelBench/blob/main/evaluation.py

    def evaluate_video(self, video: 'llava.Video', prompt: str, cot: bool = True) -> str
    parser.add_argument("--cot", action="store_true", help="Enable Chain-of-Thought output")
    pred = evaluator.evaluate_video(video, prompt, args.cot)

The SIGNATURE defaults cot=True, but the call is POSITIONAL with args.cot, and
--cot is store_true -- so running upstream's script gives CoT OFF. Reading the
signature alone gives the opposite answer, which is the whole reason this file
exists rather than a comment. judge_harness.WMB_COT = False reproduces
upstream's effective default; VilaEwmJudge(cot=True) reproduces `--cot`.

Everything else on our path is already identical to theirs:
generate_content([video, prompt]) with no generation_config, llava.Video(path),
and the same two-stem rewrite. The single deviation is parsing -- upstream
defaults an unparseable instruction score to 0, we store None so it surfaces in
record["unparsed"]. --run below applies UPSTREAM's parser, not ours, so a
reviewer is comparing like with like.

    python check_upstream_cot.py          # string checks only; no GPU, no model
    python check_upstream_cot.py --run    # + 8 real generations on one clip
    python check_upstream_cot.py --run --clip 3

MEASURED 2026-08-26, one L40S/A100, upstream's --cot path end to end: all 8
calls returned 2-8 characters ('Score: 1', 'No', 'Yes'). The same checkpoint
returns 600+ characters of prose to a bare "Describe the video", so the
capability is intact and the collapse is conditioned on the WMB template. This
file is how that claim is audited, not an invitation to keep tuning prompts.
"""
from pathlib import Path

import judge_harness as J

COT_STEMS = {
    "instruction": "Let's analyze step-by-step and conclude with",
    "physical_laws": "Let's think step-by-step and conclude with",
    "common_sense": "Let's think step-by-step and conclude with",
}
INDENT = " " * 12


def template_key(call_id):
    return "instruction" if call_id == "instruction" else call_id.rsplit("_", 1)[0]


def raw_upstream(call_id, caption):
    """The template as shipped, interpolated the way judge_harness does."""
    key = template_key(call_id)
    if key == "instruction":
        return J.WMB_PROMPT_TEMPLATES[key].format(instruction=caption)
    idx = int(call_id.rsplit("_", 1)[1])
    question = J.WMB_QUESTION_POOL[key][idx]
    return J.WMB_PROMPT_TEMPLATES[key].format(**{key: question.lower()})


def verify(caption="a ball rolls off a table and falls to the floor"):
    """String-level audit. No GPU, no weights, no video."""
    problems = []
    cot = J.VilaEwmJudge(cot=True)
    off = J.VilaEwmJudge(cot=False)

    if J.WMB_COT is not False:
        problems.append("WMB_COT is %r; upstream's --cot is store_true, so the "
                        "effective default is False" % J.WMB_COT)

    for call_id in cot.call_ids():
        key = template_key(call_id)
        want = raw_upstream(call_id, caption)
        got = cot.build_prompt(call_id, caption)
        if got != want:
            problems.append("%s: cot=True prompt is NOT the raw template" % call_id)
        if COT_STEMS[key] not in got:
            problems.append("%s: cot stem %r missing" % (call_id, COT_STEMS[key]))
        if COT_STEMS[key] in off.build_prompt(call_id, caption):
            problems.append("%s: cot stem survived cot=False" % call_id)

        a = got.splitlines()
        b = off.build_prompt(call_id, caption).splitlines()
        if len(a) != len(b):
            problems.append("%s: the two modes differ in line count" % call_id)
        else:
            diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            if len(diff) != 1:
                problems.append("%s: modes differ on %d lines, expected exactly "
                                "the format instruction" % (call_id, len(diff)))
        # WorldModelBench's source indents continuation lines by 12 spaces and
        # _wmb_template asserts on it; a reflow would be a silent prompt change
        bad = [l for l in a[1:] if l.strip() and not l.startswith(INDENT)]
        if bad:
            problems.append("%s: %d continuation line(s) lost the 12-space "
                            "indent" % (call_id, len(bad)))

    print("checked %d calls against the raw templates" % len(cot.call_ids()))
    if problems:
        print("\n%d PROBLEM(S):" % len(problems))
        for p in problems:
            print("  - %s" % p)
    else:
        print("cot=True is byte-identical to upstream's templates; cot=False "
              "differs on exactly one line per call; indentation preserved")
    return problems


def upstream_parse(call_id, pred):
    """Upstream's own parsing, warts included -- note the 0 default, which
    judge_harness deliberately replaces with None."""
    if call_id == "instruction":
        try:
            return float(pred.split(":")[-1].strip(" ."))
        except ValueError:
            return 0
    return "no" in pred.lower()


def run(dataset="test", clip=0):
    """8 real generations through upstream's call path and parser."""
    import boto3

    problems = verify()
    if problems:
        raise RuntimeError("string checks failed; fix those before generating")

    s3 = boto3.client("s3", region_name="us-east-1")
    key = J.list_source_videos(dataset)[clip]
    local = Path("/tmp/wmb_cot_%d.mp4" % clip)
    if not local.exists():
        s3.download_file(J.BUCKET, key, str(local))
    caption = J.lookup_caption(dataset, J.build_caption_manifest(dataset), key)

    judge = J.VilaEwmJudge(cot=True)
    judge.load()
    video = judge.llava.Video(str(local))
    print("\nclip: %s\ncaption: %s\n" % (Path(key).stem, caption[:90]))

    out = {}
    for call_id in judge.call_ids():
        prompt = judge.build_prompt(call_id, caption)
        assert COT_STEMS[template_key(call_id)] in prompt, call_id
        pred = str(judge.judge.generate_content([video, prompt]))
        out[call_id] = pred
        print("%-16s len %4d  upstream_parse=%r"
              % (call_id, len(pred), upstream_parse(call_id, pred)))
        print("    %r" % pred[:500])
    return out


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


if not in_notebook() and __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true",
                    help="also generate; needs the vila venv and a GPU")
    ap.add_argument("--dataset", default="test")
    ap.add_argument("--clip", type=int, default=0)
    args = ap.parse_args()
    if args.run:
        run(dataset=args.dataset, clip=args.clip)
    else:
        raise SystemExit(1 if verify() else 0)
