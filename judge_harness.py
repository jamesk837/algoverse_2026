import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

# Colab supplies credentials through userdata; everywhere else (EC2, local)
# boto3's ambient chain does, so the import must not be at module scope the way
# pushs3.py's is -- that is what made this file Colab-only. setdefault, not
# assignment, so an instance role's own region is not overwritten. The region
# has to land before the client below is built.
try:
    from google.colab import userdata
except ImportError:
    userdata = None

if userdata is not None:
    # The import succeeding does NOT mean userdata is usable: it needs a live
    # IPython kernel, so `!python foo.py` in Colab -- a subprocess with no
    # kernel -- raises AttributeError deep inside google.colab._message. That
    # is a normal way to run things in Colab, and an unguarded call here made
    # importing this module fail outright. Every secret is therefore optional:
    # on a miss we fall through to boto3's ambient chain, which is what the
    # non-Colab path uses anyway.
    for _var in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'HF_TOKEN'):
        try:
            _val = userdata.get(_var)
        except Exception:
            continue
        if _val:
            os.environ[_var] = _val
    if not os.environ.get('HF_TOKEN'):
        print("no HF_TOKEN secret; phyjudge_9b base model must be public or cached")
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')

BUCKET = "nickb-aarj"
s3 = boto3.client('s3')
MODEL_CACHE = Path("./models_cache")
TMP_DIR = Path("./tmp_judge")
RESULT_PREFIX = "results/pass1"

# Pass 2 writes to its own prefix. Checkpointing is per (clip, variant, call),
# so sharing results/pass1 would merge pass-2 output into the pass-1 records
# call by call and destroy both. Same layout underneath, so check_results /
# stats / monitor read it by pointing their own RESULT_PREFIX here.
PASS2_RESULT_PREFIX = "results/pass2"
PASS2 = False  # set by run_judges(pass2=...); read by result_key()
# Prompt-paraphrase run: same calls, same parsing, same decoding as pass 1,
# with each judge's prompt swapped for reworded index k. Results go to their
# own prefix -- checkpointing is per (clip, variant, call), so merging them
# into results/pass1 would fold paraphrased answers into pass-1 records.
PARAPHRASE_RESULT_PREFIX = "results/paraphrase"
PARAPHRASE = None  # set by run_judges(paraphrase=k); read by result_key()


def _paraphrases():
    """Lazy so this file still runs pasted into a bare cell without it."""
    import paraphrases
    return paraphrases


# Pass 2 asks for a SCORE RATIONALE, and deliberately not for chain-of-thought.
# The reasoning configuration is identical to pass 1 and identical across all
# three judges -- no thinking mode, no "let's think step-by-step", no "explain
# before you rate".
#
# Pass 2 EDITS each judge's format instruction rather than appending a sentence
# after it. Appending was tried first and does not work: every judge states its
# output contract emphatically and last, so a trailing request reads as noise.
# phyjudge is the clearest case -- its contract is "output ONLY a JSON object",
# and the appended sentence landed glued to the JSON example line, asking for
# an explanation immediately after forbidding one. All three judges ignored it
# and answered exactly as in pass 1 (measured 2026-08-26).
#
# So the modification is semantically identical across judges and surface-wise
# per judge, in each one's own idiom. The alternative -- one identical string --
# is identical text that lands as an instruction in one judge and as noise in
# another, which is uniform in the wrong place.
#
# Order is still the whole design: the answer comes FIRST and is then
# justified. Reasoning first is CoT, it moves the score, and moving the score
# is what pass 2 exists to measure and must not do to itself.
RATIONALE_CLAUSE = ", then a short explanation of why and how you arrived at it."

# (old, new) applied to a pass-1 prompt to produce the pass-2 prompt. Each
# targets that judge's final format instruction.
PASS2_REWRITES = {
    "vila_score": ("Answer with 'Score: [score]'.",
                   "Answer with 'Score: [score]'" + RATIONALE_CLAUSE),
    "vila_yesno": ('Answer with "Yes" or "No".',
                   'Answer with "Yes" or "No"' + RATIONALE_CLAUSE),
}
# phyjudge is not a phrase-format judge, it is a SCHEMA judge: its prompt ends
# with a worked example of the JSON it must emit. Measured 2026-08-26: editing
# only the sentence -- dropping ONLY and asking for an explanation -- changes
# nothing, because the example still shows ONE key and the model follows the
# demonstration over the instruction. The example is what has to change.
PHYJUDGE_P2_SENTENCE_OLD = "Then output ONLY a JSON object with exactly one key: %s."
PHYJUDGE_P2_SENTENCE_NEW = "Then output a JSON object with exactly two keys: %s and reason."
PHYJUDGE_P2_REASON = '"<why you gave this score>"'
# videophy2_auto is fine-tuned to emit a score digit and end the turn, so a
# rationale request inside the same turn is ignored (measured 2026-08-26: every
# reply is one character). Its prompt is already a Human:/AI: transcript, so the
# alternative is a real second turn:
#
#   turn 1  the pass-1 prompt, verbatim -> the score digit
#   turn 2  that transcript with the digit filled in after "AI: ", a new Human:
#           turn asking why, and a fresh "AI: " for the model to complete
#
# The turn-1 digit is NOT regenerated -- it is read from results/pass1, so the
# pass-2 score is the pass-1 score by construction. Consequence worth stating
# when reporting: for this judge the two passes cannot measure whether requiring
# a justification moves the score, because it cannot. What they compare is the
# rationale against a score that is held fixed.
#
# MEASURED, 3 clips x SA and PC: turn 2 returns ONE character, echoing the digit
# that was planted rather than answering the new Human turn. Kept in the tree
# because it is the mode that was asked for and the negative result is the
# finding. "suffix" reproduces the earlier single-turn behaviour, which fails
# differently -- the suffix is ignored and the reply is the bare digit.
VP2_PASS2_MODE = "two_turn"          # "two_turn" | "suffix"
VP2_TWO_TURN_ASK = (
    "Why did you give that rating? Explain how and why you arrived at it."
)
VP2_RATIONALE_SUFFIX = (
    " Give the rating first, then a short explanation of why and how you "
    "arrived at it."
)


def phyjudge_pass2_prompt(user_prompt, score_key):
    """Pass-2 form of a phyjudge prompt: two-key contract AND two-key example.

    Both edits are required and the example is the load-bearing one. Raises if
    either target is missing rather than returning a half-edited prompt --
    a pass-2 prefix full of records that are really pass 1 is indistinguishable
    from a pass 2 that simply had no effect, which is the thing being measured.
    """
    old = PHYJUDGE_P2_SENTENCE_OLD % score_key
    if old not in user_prompt:
        raise RuntimeError(
            "pass 2: phyjudge key sentence %r not found; subq+human.yaml "
            "changed" % old)
    out = user_prompt.replace(old, PHYJUDGE_P2_SENTENCE_NEW % score_key, 1)
    # the example is already rendered by infer.py, so single braces here
    ex = re.compile(r'\{"%s":\s*(\d+)\}' % re.escape(score_key))
    m = ex.search(out)
    if not m:
        raise RuntimeError(
            "pass 2: phyjudge JSON example for key %r not found; the model "
            "follows the example over the instruction, so a prompt without "
            "the two-key example is pass 1 wearing a pass-2 prefix" % score_key)
    two = '{"%s": %s, "reason": %s}' % (score_key, m.group(1), PHYJUDGE_P2_REASON)
    return out[:m.start()] + two + out[m.end():]


def _apply_rewrite(prompt, *keys):
    """Rewrite a pass-1 prompt into its pass-2 form. Raises if the format
    instruction is not where we expect -- silently returning the pass-1 prompt
    would produce a whole run of pass-2 records that are really pass 1."""
    for key in keys:
        old, new = PASS2_REWRITES[key]
        if old in prompt:
            return prompt.replace(old, new, 1)
    raise RuntimeError(
        "pass 2: no format instruction found to rewrite (looked for %s). The "
        "prompt changed shape; fix PASS2_REWRITES rather than shipping pass-1 "
        "prompts under a pass-2 prefix." % list(keys))

# Pass 1's parsers read the whole reply, so they cannot survive a rationale
# after the answer: vila's yes/no check is `"no" in pred.lower()`, and "no" is
# inside "not", "nothing" and "cannot", so almost any prose reads as a
# violation; videophy2's scans a dict of number WORDS in dict order rather than
# by position, so "there is one ball" wins over a leading "3". Pass 2 therefore
# parses POSITIONALLY -- first answer token, everything after it ignored.
_P2_SCORE_RE = re.compile(r"score\s*[:=]\s*(-?[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
# The prompt asks for "Yes" or "No", but a model writing an explanation often
# answers True/False instead, so both vocabularies are accepted. POLARITY: the
# question is "does it show any violation of X", and upstream stores True for
# "no violation found" (its parse is `"no" in pred.lower()`). So yes/true mean
# a violation IS present and store False; no/false store True. Getting this
# backwards would invert every law column without any visible symptom.
_P2_YESNO_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
_P2_VIOLATION_WORDS = {"yes", "true"}
# The two scales are NOT the same and must not share a pattern: WorldModelBench's
# instruction score is 0-3, VideoPhy-2's is 1-5. A shared [1-5] silently drops a
# legitimate instruction score of 0 and can pick up a stray 4 or 5 out of the
# explanation instead.
_P2_INSTR_RE = re.compile(r"\b([0-3])\b")   # vila instruction
# JSON first, prose after: upstream's parse_score never had to cope with
# trailing text, so pass 2 retries it on the first {...} block if it comes
# back empty.
_P2_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
MANIFEST_PREFIX = "manifests"

DATASET_PREFIXES = {
    "test": "datasets/videophy2_test/",
    "implausibench_real": "datasets/implausibench/ImplausiBench/real/",
    "implausibench_implausible": "datasets/implausibench/ImplausiBench/implausible/",
}
VIDEO_SUFFIXES = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

ATTACK_FILES = [
    "shuffle",
    "reverse",
    "freeze",
    "photometric",
    "caption_echo_rubric_vocab",
    "caption_echo_score_anchor_positive",
    "caption_echo_authoritative_claim",
    "caption_echo_score_anchor_negative",
    "caption_echo_control_irrelevant",
]

# Not attacks -- the codec control. Every attacked variant above was written
# out by ffmpeg and carries one extra libx264 CRF 23 pass that `clean` does
# not, since clean IS the source object. `identity` is that pass with nothing
# manipulated, so dJ(attack) - dJ(identity) is the codec-corrected effect.
# Mirrors attack_suite.CONTROL_ATTACKS; edit the two together.
CONTROL_FILES = ["identity"]

# VARIANTS is the experiment and stays the default run, so adding the control
# does not retroactively mark every completed clip incomplete -- audit_runs,
# check_results and monitor all measure against a list of this length.
# ALL_VARIANTS is only what `variants=` is allowed to name.
VARIANTS = ["clean"] + ATTACK_FILES
ALL_VARIANTS = VARIANTS + CONTROL_FILES

IMPLAUSIBENCH_FIXED_CAPTION = "the events in the video appear to follow physics principles"

DATASETS = {
    "test": {
        "caption_source": "metadata",
        "meta_key": "datasets/videophy2_test/_metadata/videophy2_test.csv",
        "caption_column": "caption",
        "id_column": "video_url",
    },
    "implausibench_real": {"caption_source": "fixed", "caption": IMPLAUSIBENCH_FIXED_CAPTION},
    "implausibench_implausible": {"caption_source": "fixed", "caption": IMPLAUSIBENCH_FIXED_CAPTION},
}

VIDEOPHY_CODE_PREFIX = "code/videophy/VIDEOPHY2/"

PROMPT_SA = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: Does this video match the description: \"{caption}\"? "
    "Please rate the video on a scale from 1 to 5, where 5 indicates a perfect match and 1 indicates no relevance.\n"
    "AI: "
)

PROMPT_PHYSICS = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: Does this video adhere to the physical laws? "
    "Rate the video on a scale from 1 to 5, where 5 means full compliance and 1 means significant violations.\n"
    "AI: "
)

VP2_GENERATE_KWARGS = {
    "do_sample": False,
    "top_k": 1,
    "temperature": 0.001,
    "max_length": 256,
}
# rationale replies are prose, and max_length counts prompt + completion
VP2_RATIONALE_MAX_LENGTH = 512
VP2_NUM_FRAMES = 32

VP2_NUM_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}

# Pass 2's videophy2 parser, built FROM VP2_NUM_MAP so its vocabulary can never
# drift from pass 1's. The only difference is that this one is positional: it
# takes the earliest token in the reply, where vp2_parse walks the map in dict
# order and so lets a "one" anywhere in the explanation beat a leading "3".
# Digits-only would have been a regression -- pass 1 accepts number words, so a
# reply of "Three. The water flows naturally." must keep parsing as 3.
_P2_VP2_RE = re.compile(
    r"\b(" + "|".join(sorted(VP2_NUM_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)


# ------------------------------------------------ vp2 rationale elicitation --
# videophy_2_auto is two fine-tunes downstream of VideoCon (HF model tree:
# videophy_2_auto <- videocon_physics <- VideoCon on mPLUG-Owl-video), and
# VideoCon's third training task was natural-language explanation:
#   "[V] What is the misalignment between this video and the description [C]?"
# videophy_2_auto's OWN SFT also included a per-rule 0/1/2 task the harness
# never wired in (upstream VIDEOPHY2/template.py, PROMPT_RULE, quoted verbatim
# below). Both are TRAINED interfaces -- vp2_rationale_probe tries those before
# any free-form ask, because format-matching a trained task is what makes a
# non-digit answer reachable at all on a model whose score template collapsed
# to digit->EOS.
VP2_PREAMBLE = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
)
VP2_PROMPT_NLE = (
    VP2_PREAMBLE +
    "Human: <|video|>\n"
    "Human: What is the misalignment between this video and the description: \"{caption}\"?\n"
    "AI: "
)
# verbatim from upstream VIDEOPHY2/template.py (including the stray space
# before the final newline)
VP2_PROMPT_RULE = (
    VP2_PREAMBLE +
    "Human: <|video|>\n"
    "Human: Does the video follow the physical rule: \"{rule}\"?\n"
    "Choose 0 if not, 1 if valid, or 2 if indeterminate. \n"
    "AI: "
)
# generic probes only; production should feed each clip's own candidate rules
# from the VideoPhy-2 metadata
VP2_PROBE_RULES = (
    "Objects fall under gravity unless supported",
    "Solid objects do not pass through each other",
    "Object motion is smooth and continuous over time",
)
VP2_EXPLAIN_TURN = "Explain why you gave that rating. What in the video did you rely on?"

WMB_COT = False

def _wmb_template(text):
    return text.strip().replace("\n\n", "\n" + " " * 12 + "\n")


WMB_PROMPT_TEMPLATES = {
    "instruction": _wmb_template("""
            Evaluate if this video follows the instruction: '{instruction}'.
            Use the following scoring criteria:

            - 0: The video does not follow the instruction at all.
            - 1: The video includes the correct object but performs the wrong action, or vice versa.
            - 2: The video follows the instruction and shows a tendency toward the intended goal.
            - 3: The video follows the instruction precisely and successfully achieves the goal.

            Let's analyze step-by-step and conclude with 'Score: [score]'.
        """),

    "physical_laws": _wmb_template("""
            Watch the video and determine if it shows any '{physical_laws}'
            Let's think step-by-step and conclude with "Yes" or "No".
        """),

    "common_sense": _wmb_template("""
            Does the video exhibit '{common_sense}'?
            Let's think step-by-step and conclude with "Yes" or "No".
        """),
}

assert WMB_PROMPT_TEMPLATES["instruction"].count("\n" + " " * 12 + "\n") == 2

WMB_QUESTION_POOL = {
    "instruction": None,
    "physical_laws": [
        "Violation of Newton's Law: Objects move without any external force.",
        "Violation of the Law of Conservation of Mass or Solid Constitutive Law: Objects deform irregularly.",
        "Violation of Fluid Constitutive Law: Liquids flow in an unnatural manner.",
        "Violation of Non-physical Penetration: Objects unnaturally pass through each other.",
        "Violation of Gravity: Objects behave inconsistently with gravity.",
    ],
    "common_sense": [
        "Poor Aesthetics: Visually unappealing or low-quality content.",
        "Temporal Inconsistency: Noticeable flickering or abrupt changes.",
    ],
}

PHYJUDGE_GENERAL_KEYS = ["SA", "PTV", "persistence"]
PHYJUDGE_LAWS = [
    "gravity", "inertia", "momentum", "impenetrability", "collision", "material",
    "buoyancy", "displacement", "flow_dynamics", "boundary_interaction",
    "fluid_continuity", "reflection", "shadow",
]
PHYJUDGE_FPS = 2.0
PHYJUDGE_MAX_PIXELS = 360 * 640
PHYJUDGE_MAX_NEW_TOKENS = 128
# A rationale is prose, not a JSON score, so it needs room. Greedy decoding
# makes the larger budget free: identical tokens either way, it only lifts the
# ceiling. Pass 1's budget is untouched.
PHYJUDGE_RATIONALE_MAX_NEW_TOKENS = 512
PHYJUDGE_PROMPT_YAML = "subq+human.yaml"

def key_exists(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def get_json(key):
    try:
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except Exception:
        return None
    return json.loads(body)


def put_json(key, obj):
    s3.put_object(
        Bucket=BUCKET, Key=key,
        Body=json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def list_keys(prefix, suffixes=None):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if suffixes and Path(k).suffix.lower() not in suffixes:
                continue
            keys.append((k, obj["Size"]))
    return keys


def list_source_videos(dataset, limit=None):
    keys = [k for k, _ in list_keys(DATASET_PREFIXES[dataset], VIDEO_SUFFIXES)]
    keys.sort()
    return keys[:limit] if limit else keys


def fmt_secs(s):
    s = int(s)
    if s >= 3600:
        return f"{s//3600}h{(s%3600)//60:02d}m"
    if s >= 60:
        return f"{s//60}m{s%60:02d}s"
    return f"{s}s"


def clips_with_attacks(dataset):
    """Stems that have a rendered variant directory under attacks/<dataset>/.

    A clip whose variants were never rendered still has a `clean` (it is the
    source object), so the harness would happily score it -- 16 phyjudge calls
    -- and then print `missing video` nine times. That is pure cost: every
    measurement in this project is a within-clip delta between clean and a
    variant, so a clip with no variants contributes nothing. It also leaves a
    permanently-incomplete record in results/pass1.

    Delimiter="/" makes this one cheap LIST returning a common prefix per clip,
    rather than an object listing of every rendered variant.
    """
    prefix = f"attacks/{dataset}/"
    stems, paginator = set(), s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            stem = cp["Prefix"][len(prefix):].strip("/")
            if stem:
                stems.add(stem)
    return stems


def clips_with_variants(dataset, variants):
    """Stems that have EVERY requested attacked variant actually rendered.

    clips_with_attacks() answers a cheaper question -- does the clip have a
    variant directory at all -- using Delimiter="/" so one LIST returns a
    common prefix per clip instead of an object per variant. That is the right
    trade when the run wants all nine, because a clip with a directory almost
    always has the full set and a stray gap costs one `missing video` line.

    It is the wrong trade when the run wants a NAMED subset: a clip whose
    directory holds eight variants but not `shuffle` passes that check, gets
    counted against num_clips, and then contributes a clean score with nothing
    to difference it against. Every measurement here is a within-clip delta, so
    that clip is not a partial result, it is no result. This does the full
    object LIST -- ~9 keys per clip, still one paginated call -- and requires
    each named variant to be present.
    """
    want = {v for v in variants if v != "clean"}   # clean IS the source object
    if not want:
        return None                                # nothing to require
    prefix = f"attacks/{dataset}/"
    have, paginator = {}, s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            rest = obj["Key"][len(prefix):]
            stem, _, name = rest.rpartition("/")
            if stem and name.endswith(".mp4"):
                have.setdefault(stem, set()).add(name[:-len(".mp4")])
    return {stem for stem, got in have.items() if want <= got}


def clips_with_pass1(dataset, model, call_ids, variants, stems, workers=8):
    """Stems whose results/pass1 record for `model` is complete over the
    requested variants -- every call present, on every variant.

    Existence of the result object is not the same question. Checkpointing is
    per (clip, variant, call), so a record appears at a clip's FIRST variant
    and an interrupted run leaves a partial one behind; a LIST would count
    those as scored. Pass 2 exists to be differenced against pass 1, so a clip
    whose pass-1 side is half-written yields a comparison with a hole in it.

    Reads only the candidate stems, in parallel -- the module client is
    thread-safe (it is fork-safety boto3 lacks) and botocore pools 10
    connections by default, so the worker count stays under that.
    """
    keys = {stem: f"{RESULT_PREFIX}/{model}/{dataset}/{stem}.json"
            for stem in stems}

    def fetch(stem):
        return stem, get_json(keys[stem])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = dict(pool.map(fetch, sorted(stems)))

    complete, absent, partial, unparsed = set(), [], [], []
    for stem, rec in records.items():
        if not rec:
            absent.append(stem)
            continue
        runs = rec.get("runs", {})
        gaps = [v for v in variants
                if any(c not in runs.get(v, {}).get("calls", {})
                       for c in call_ids)]
        if gaps:
            partial.append(stem)
            continue
        complete.add(stem)
        if any(out.get("parsed") is None
               for v in variants for out in runs[v]["calls"].values()):
            unparsed.append(stem)

    print(f"  pass-1 {model}: {len(complete)} of {len(stems)} clips complete "
          f"over {'+'.join(variants)} ({len(absent)} never scored, "
          f"{len(partial)} partial)")
    if unparsed:
        # kept, not dropped: an unparsed call is already visible in the
        # record's `unparsed` list, and silently shrinking n would be worse
        print(f"  NOTE {len(unparsed)} of those carry at least one unparsed "
              f"pass-1 call; they are kept -- check record['unparsed']")
    return complete


def shard_keys(keys, shard):
    """Take this worker's stripe of `keys`, as (index, count).

    Nothing in the harness parallelizes on its own: the loop is one generate()
    at a time, and device_map="auto" shards a model across GPUs rather than
    replicating it, so extra GPUs on one box add nothing by themselves. The work
    is embarrassingly parallel at the clip level instead -- checkpointing is per
    (clip, variant, call) in S3, so disjoint stripes never touch the same result
    key and need no coordination. One process per GPU (CUDA_VISIBLE_DEVICES=i)
    or one per instance both work.

    Striped (keys[i::n]) rather than contiguous blocks: clip cost varies with
    duration and consecutive keys sort together by filename, so blocks would
    hand one worker a run of long clips. list_source_videos sorts explicitly, so
    every worker stripes an identical list without talking to the others.
    """
    if shard is None:
        return keys
    index, count = shard
    if not (isinstance(index, int) and isinstance(count, int)):
        raise ValueError(f"shard must be (index, count) ints, got {shard!r}")
    if count < 1 or not (0 <= index < count):
        raise ValueError(f"shard index must satisfy 0 <= {index} < {count}")
    return keys[index::count]


def sync_prefix(prefix, dest):
    dest.mkdir(parents=True, exist_ok=True)
    objects = list_keys(prefix)
    if not objects:
        raise RuntimeError(f"nothing under s3://{BUCKET}/{prefix}")
    for key, size in objects:
        rel = key[len(prefix):].lstrip("/")
        if not rel or key.endswith("/"):
            continue
        local = dest / rel
        if local.exists() and local.stat().st_size == size:
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {rel} ({size / 1e6:.1f} MB)")
        s3.download_file(BUCKET, key, str(local))
    return dest


def safe_local_name(name, max_len=80):
    import hashlib

    stem, ext = os.path.splitext(name)
    if len(stem) <= max_len:
        return name
    short = hashlib.sha256(stem.encode()).hexdigest()[:16]
    return f"{stem[:max_len]}_{short}{ext}"


def build_caption_manifest(dataset, rebuild=False):
    cfg = DATASETS[dataset]
    if cfg["caption_source"] == "fixed":
        return None

    manifest_key = f"{MANIFEST_PREFIX}/captions_{dataset}.json"
    if not rebuild:
        cached = get_json(manifest_key)
        if cached:
            return cached

    import pandas as pd

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    local = TMP_DIR / f"meta__{Path(cfg['meta_key']).name}"
    s3.download_file(BUCKET, cfg["meta_key"], str(local))
    try:
        df = pd.read_csv(local, on_bad_lines="skip", encoding="utf-8",
                         encoding_errors="replace")
    finally:
        local.unlink(missing_ok=True)

    for col in (cfg["caption_column"], cfg["id_column"]):
        if col not in df.columns:
            raise RuntimeError(
                f"column '{col}' missing from {cfg['meta_key']}; columns are "
                f"{list(df.columns)}")

    captions = {}
    for _, row in df.iterrows():
        url, caption = row[cfg["id_column"]], row[cfg["caption_column"]]
        if pd.isna(url) or pd.isna(caption) or not str(caption).strip():
            continue
        stem = Path(str(url).split("?")[0].split("/")[-1]).stem
        captions.setdefault(stem, str(caption).strip())

    if not captions:
        raise RuntimeError(f"no captions recovered from {cfg['meta_key']}")
    put_json(manifest_key, captions)
    print(f"  caption manifest -> s3://{BUCKET}/{manifest_key} ({len(captions)} entries)")
    return captions


def lookup_caption(dataset, captions, source_key):
    if DATASETS[dataset]["caption_source"] == "fixed":
        return DATASETS[dataset]["caption"]
    return captions.get(Path(source_key).stem)


def vp2_parse(output):
    output_lower = output.lower().strip()
    for key, val in VP2_NUM_MAP.items():
        if key in output_lower:
            return val
    digits = "".join([c for c in output_lower if c.isdigit()])
    if digits and int(digits) in VP2_NUM_MAP.values():
        return int(digits)
    return None


def _patch_vila_video_cache():
    """Decode each clip once per variant instead of once per call.

    VILA's `generate_content` runs `extract_media`, which calls
    `llava.utils.media._load_video(path, num_frames=..., fps=...)` every single
    time. WorldModelBench asks 8 questions of the same clip -- 1 instruction +
    5 physical laws + 2 common sense -- so 7 of every 8 decodes are thrown
    away. Measured 2026-08-26 on an L40S: 5.3 s/call for a 1.5B model, i.e.
    ~7 min per clip, nearly all of it re-reading the same mp4.

    Memoized on the full argument tuple, so a different num_frames or fps is a
    different entry and can never silently reuse the wrong frames. The cache
    holds exactly ONE clip -- calls arrive 8-at-a-time on one path, so a
    second entry is never useful and would just pin frame buffers in RAM.
    A copy of the list is returned so a caller appending to it cannot corrupt
    the cached entry.

    The model input is byte-identical: same function, same arguments, same
    frames. This changes throughput only.

    Best effort by design. VILA is vendored and its internals move between
    versions, so if the function is not where we expect, this prints and
    returns, and the run proceeds at the old speed rather than failing.
    """
    try:
        from llava.utils import media as _media
    except Exception as exc:
        print(f"  [vila] video cache not applied ({exc}); decoding per call")
        return
    if getattr(_media, "_ewm_video_cached", False):
        return
    orig = getattr(_media, "_load_video", None)
    if not callable(orig):
        print("  [vila] llava.utils.media._load_video missing; decoding per call")
        return

    state = {"key": None, "frames": None, "hits": 0, "misses": 0}

    def cached(video_path, *args, **kwargs):
        key = (str(video_path), args, tuple(sorted(kwargs.items())))
        if key != state["key"]:
            state["key"] = key
            state["frames"] = orig(video_path, *args, **kwargs)
            state["misses"] += 1
        else:
            state["hits"] += 1
        frames = state["frames"]
        return list(frames) if isinstance(frames, list) else frames

    _media._load_video = cached
    _media._ewm_video_cached = True
    _media._ewm_video_stats = state
    print("  [vila] video decode cached per clip (was: once per call)")


class VilaEwmJudge:
    name = "vila_ewm"
    s3_prefix = "models/vila-ewm-qwen2-1.5b/"

    def __init__(self, cot=WMB_COT, pass2=False, paraphrase=None):
        self.cot = cot
        self.pass2 = pass2
        self.paraphrase = paraphrase
        # BOTH the templates and the question pool are paraphrased -- the
        # questions are interpolated INTO the templates, so swapping only one
        # would be a half-reworded prompt.
        if paraphrase is None:
            self.templates = WMB_PROMPT_TEMPLATES
            self.questions = WMB_QUESTION_POOL
        else:
            P = _paraphrases()
            self.templates = {k: P.VILA_TEMPLATES[k]["paraphrases"][paraphrase]
                              for k in WMB_PROMPT_TEMPLATES}
            # "instruction" maps to None -- it is a template, not a pool
            self.questions = {
                g: (qs if not qs else
                    [P.VILA_QUESTIONS["%s_%d" % (g, i)]["paraphrases"][paraphrase]
                     for i in range(len(qs))])
                for g, qs in WMB_QUESTION_POOL.items()}
        self.llava = None
        self.judge = None

    def call_ids(self):
        ids = ["instruction"]
        ids += [f"physical_laws_{i}" for i in range(len(WMB_QUESTION_POOL["physical_laws"]))]
        ids += [f"common_sense_{i}" for i in range(len(WMB_QUESTION_POOL["common_sense"]))]
        return ids

    def load(self):
        try:
            import llava
        except ImportError as e:
            raise RuntimeError(
                "VILA judge needs the VILA codebase: git clone "
                "https://github.com/NVlabs/VILA.git && pip install -e VILA") from e
        model_dir = sync_prefix(self.s3_prefix, MODEL_CACHE / self.name)
        self.llava = llava
        self.judge = llava.load(str(model_dir))
        _patch_vila_video_cache()

    def build_prompt(self, call_id, caption):
        if call_id == "instruction":
            prompt = self.templates["instruction"].format(instruction=caption)
        else:
            eval_type, idx = call_id.rsplit("_", 1)
            question = self.questions[eval_type][int(idx)]
            prompt = self.templates[eval_type].format(**{eval_type: question.lower()})
        if not self.cot:
            prompt = prompt.replace(
                "Let's think step-by-step and conclude with", "Answer with"
            ).replace(
                "Let's analyze step-by-step and conclude with", "Answer with"
            )
        if self.pass2:
            prompt = _apply_rewrite(prompt, "vila_score", "vila_yesno")
        return prompt

    def parse(self, call_id, pred):
        if self.pass2:
            return self.parse_pass2(call_id, pred)
        if call_id == "instruction":
            try:
                return float(pred.split(":")[-1].strip(" ."))
            except ValueError:
                return None
        return "no" in pred.lower()

    def parse_pass2(self, call_id, pred):
        """Positional: the FIRST answer, rationale ignored.

        Upstream's yes/no test is a substring check for "no", which also fires
        inside "not"/"nothing"/"cannot" -- unusable once prose follows. The
        boolean keeps upstream's polarity: True means no violation found."""
        if call_id == "instruction":
            m = _P2_SCORE_RE.search(pred or "")
            if m:
                return float(m.group(1))
            # fallback is 0-3, this judge's actual scale -- see _P2_INSTR_RE
            m = _P2_INSTR_RE.search(pred or "")
            return float(m.group(1)) if m else None
        m = _P2_YESNO_RE.search(pred or "")
        if not m:
            return None
        # True == no violation found, matching upstream's stored polarity
        return m.group(1).lower() not in _P2_VIOLATION_WORDS

    def run(self, video_path, caption, call_id):
        prompt = self.build_prompt(call_id, caption)
        video = self.llava.Video(str(video_path))
        pred = str(self.judge.generate_content([video, prompt]))
        return pred, self.parse(call_id, pred)


class VideoPhy2AutoJudge:
    name = "videophy2_auto"
    s3_prefix = "models/videophy_2_auto/"
    def __init__(self, num_frames=VP2_NUM_FRAMES, pass2=False, paraphrase=None):
        self.num_frames = num_frames
        self.pass2 = pass2
        self.paraphrase = paraphrase
        self.torch = None
        self.model = None

    def call_ids(self):
        return ["SA", "PC"]

    def load(self):
        import sys

        import torch

        code_dir = sync_prefix(VIDEOPHY_CODE_PREFIX, MODEL_CACHE / "_videophy_code")
        sys.path.insert(0, str(code_dir))
        from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
        from mplug_owl_video.processing_mplug_owl import (MplugOwlImageProcessor,
                                                          MplugOwlProcessor)
        from transformers.models.llama.tokenization_llama import LlamaTokenizer

        model_dir = sync_prefix(self.s3_prefix, MODEL_CACHE / self.name)
        self.torch = torch
        self.tokenizer = LlamaTokenizer.from_pretrained(str(model_dir))
        image_processor = MplugOwlImageProcessor.from_pretrained(str(model_dir))
        self.processor = MplugOwlProcessor(image_processor, self.tokenizer)
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, device_map={"": "cpu"})
        print("Model Loaded")
        self.model.eval()
        self.model = self.model.to("cuda").to(torch.bfloat16)

    def build_prompt(self, call_id, caption):
        base = (PROMPT_SA.format(caption=caption) if call_id == "SA"
                else PROMPT_PHYSICS)
        if self.paraphrase is not None:
            # the paraphrase is the INSTRUCTION SENTENCE only; the system line,
            # the <|video|> token and the trailing "AI: " are structural
            entry = _paraphrases().VP2_INSTRUCTIONS[call_id]
            old = entry["original"]
            new = entry["paraphrases"][self.paraphrase]
            if call_id == "SA":
                old = old.format(caption=caption)
                new = new.format(caption=caption)
            if old not in base:
                raise RuntimeError(
                    "paraphrase: videophy2 %s instruction not found in the "
                    "prompt; paraphrases.py has drifted" % call_id)
            base = base.replace(old, new, 1)
        if not self.pass2 or VP2_PASS2_MODE == "two_turn":
            # under two_turn this IS turn 1, byte-identical to pass 1;
            # run() appends the second turn
            return base
        # the trailing "AI: " must stay last -- the model completes it
        head, sep, tail = base.rpartition("\nAI: ")
        return head + VP2_RATIONALE_SUFFIX + sep + tail

    def set_context(self, dataset, source_key, variant):
        """Told which (clip, variant) is being scored, so two_turn can find the
        pass-1 digit. process_clip calls this if the judge defines it."""
        self._ctx = (dataset, source_key, variant)

    def pass1_score(self, call_id):
        """The pass-1 digit for the current (clip, variant, call), read from S3.

        Raises rather than regenerating: a silent fallback would make the pass-2
        score sometimes inherited and sometimes freshly generated, with nothing
        in the record saying which.
        """
        ctx = getattr(self, "_ctx", None)
        if ctx is None:
            raise RuntimeError(
                "videophy2 two_turn needs set_context(dataset, source_key, "
                "variant) before run()")
        dataset, source_key, variant = ctx
        key = "%s/%s/%s/%s.json" % (RESULT_PREFIX, self.name, dataset,
                                    Path(source_key).stem)
        cached = getattr(self, "_p1_cache", None)
        if not cached or cached[0] != key:
            self._p1_cache = (key, get_json(key))
        rec = self._p1_cache[1]
        if not rec:
            raise RuntimeError(
                "videophy2 two_turn: no pass-1 record at s3://%s/%s -- score "
                "pass 1 for this clip first" % (BUCKET, key))
        call = rec.get("runs", {}).get(variant, {}).get("calls", {}).get(call_id)
        if not call or call.get("parsed") is None:
            raise RuntimeError(
                "videophy2 two_turn: pass 1 has no parsed score for %s/%s in %s"
                % (variant, call_id, key))
        return call["parsed"]

    def run(self, video_path, caption, call_id):
        torch = self.torch
        prompt = self.build_prompt(call_id, caption)
        score = None
        if self.pass2 and VP2_PASS2_MODE == "two_turn":
            score = self.pass1_score(call_id)
            prompt = "%s%s\nHuman: %s\nAI: " % (
                prompt, score, VP2_TWO_TURN_ASK)
        prompts = [prompt]
        inputs = self.processor(text=prompts, videos=[str(video_path)],
                                num_frames=self.num_frames, return_tensors="pt")
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v
                  for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        kwargs = (dict(VP2_GENERATE_KWARGS, max_length=VP2_RATIONALE_MAX_LENGTH)
                  if self.pass2 else VP2_GENERATE_KWARGS)
        with torch.no_grad():
            res = self.model.generate(**inputs, **kwargs)
        output = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        if self.pass2:
            if VP2_PASS2_MODE == "two_turn":
                # the score is pass 1's and is NOT re-parsed out of turn 2 --
                # turn 2 is the rationale, and any number in it is prose
                return output, score
            # positional, same vocabulary as pass 1: vp2_parse walks the map in
            # dict order, so "there is one ball" would beat a leading "3"
            m = _P2_VP2_RE.search(output or "")
            return output, (VP2_NUM_MAP[m.group(1).lower()] if m else None)
        return output, vp2_parse(output)


class ScalarFpsProcessor:
    def __init__(self, processor):
        self._processor = processor

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def __call__(self, *args, **kwargs):
        fps = kwargs.get("fps")
        if isinstance(fps, (list, tuple)) and len(fps) == 1:
            kwargs["fps"] = fps[0]
        return self._processor(*args, **kwargs)


class PhyJudge9BJudge:
    name = "phyjudge_9b"
    s3_prefix = "models/phyjudge-9B/"
    def __init__(self, fps=PHYJUDGE_FPS, max_pixels=PHYJUDGE_MAX_PIXELS,
                 max_new_tokens=None, pass2=False, paraphrase=None):
        self.fps = fps
        self.max_pixels = max_pixels
        self.pass2 = pass2
        self.paraphrase = paraphrase
        self.max_new_tokens = max_new_tokens or (
            PHYJUDGE_RATIONALE_MAX_NEW_TOKENS if pass2 else PHYJUDGE_MAX_NEW_TOKENS)
        self.infer = None

    def call_ids(self):
        return list(PHYJUDGE_GENERAL_KEYS) + list(PHYJUDGE_LAWS)

    def load(self):
        import sys

        import torch

        model_dir = sync_prefix(self.s3_prefix, MODEL_CACHE / self.name)
        if not (model_dir / "infer.py").exists():
            raise RuntimeError(
                f"{model_dir}/infer.py missing; re-mirror the full HF repo "
                f"(infer.py + {PHYJUDGE_PROMPT_YAML}) into s3://{BUCKET}/{self.s3_prefix}")
        sys.path.insert(0, str(model_dir))
        import infer

        self.torch = torch
        self.infer = infer
        processor, self.model, adapter_dir = infer.load_model(
            str(model_dir), dtype=torch.bfloat16, device_map="auto")
        self.processor = ScalarFpsProcessor(processor)
        self.cfg = infer.load_yaml(adapter_dir / PHYJUDGE_PROMPT_YAML)
        if self.paraphrase is not None:
            self.cfg = _paraphrase_phyjudge_cfg(self.cfg, self.paraphrase)
        self.device = next(self.model.parameters()).device
        unknown = [c for c in self.call_ids()
                   if c not in infer.GENERAL_SUB_QUESTIONS and c not in infer.PHYSICAL_CRITERIA]
        if unknown:
            raise RuntimeError(f"call ids not present in infer.py: {unknown}")

    def prepare_inputs(self, messages):
        from qwen_vl_utils import process_vision_info

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "video":
                        item.setdefault("fps", self.fps)
                        item.setdefault("max_pixels", self.max_pixels)
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, return_video_metadata=True)
        except TypeError:
            return self.infer.prepare_inputs(self.processor, messages, self.device,
                                             fps=self.fps, max_pixels=self.max_pixels)

        video_kwargs = dict(video_kwargs)
        if video_inputs and isinstance(video_inputs[0], (tuple, list)) and len(video_inputs[0]) == 2:
            videos, metadata = zip(*video_inputs)
            video_inputs = list(videos)
            video_kwargs["video_metadata"] = list(metadata)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt", **video_kwargs)
        return inputs.to(self.device)

    def run(self, video_path, caption, call_id):
        infer = self.infer
        metric = call_id if call_id in PHYJUDGE_GENERAL_KEYS else None
        law = None if metric else call_id
        system_prompt, user_prompt, score_key = infer.build_prompt(
            self.cfg, caption, metric=metric, law=law)
        if self.pass2:
            user_prompt = phyjudge_pass2_prompt(user_prompt, score_key)
        messages = infer.build_messages(system_prompt, user_prompt, Path(video_path))
        inputs = self.prepare_inputs(messages)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        raw = infer.decode_generated(self.processor, inputs, generated_ids)
        score = infer.parse_score(raw, score_key)
        if self.pass2 and score is None:
            # upstream's parser never had to cope with prose after the JSON;
            # retry it on the first {...} block before giving up
            m = _P2_JSON_RE.search(raw or "")
            if m:
                try:
                    score = infer.parse_score(m.group(0), score_key)
                except Exception:
                    score = None
        return raw, score


JUDGES = {
    "vila_ewm": VilaEwmJudge,
    "videophy2_auto": VideoPhy2AutoJudge,
    "phyjudge_9b": PhyJudge9BJudge,
}

def vp2_rationale_probe(dataset="test", n_clips=3, push_to_s3=True):
    """Elicitation ladder for a videophy2_auto score rationale, on clean clips
    that already have a pass-1 PC score.

    Every mode ECHOES the exact text handed to the processor and stores it in
    the report, so "was the follow-up actually in the prompt" is answered by
    the artifact. Modes, strongest first methodologically:

      rule_*         the model's OWN trained 0/1/2 per-rule task (native SFT)
      nle            VideoCon's trained explanation task, trained wording
      explain        the two-turn follow-up (VP2_PASS2_MODE="two_turn" path,
                     second phrasing)
      explain_wrong  same transcript with a deliberately WRONG digit planted:
                     reply echoes the plant -> turn 2 copies the transcript;
                     reply is the true pass-1 digit -> it recomputes from the
                     video and ignores the conversation
      explain_forced explain with EOS banned for 40 tokens  (diagnostic only)
      prefill        forced opener "I rated it N because"   (diagnostic only)
      describe       capability floor: ANY prose in-template at all?

    Reads pass-1 from RESULT_PREFIX explicitly -- result_key() follows the
    PASS2/PARAPHRASE globals and must not be used to reach pass-1 records.
    """
    judge = VideoPhy2AutoJudge()
    judge.load()
    torch = judge.torch

    def generate(text, video_path, min_new=None):
        kwargs = {k: v for k, v in VP2_GENERATE_KWARGS.items()
                  if k != "max_length"}
        kwargs["max_new_tokens"] = 256
        if min_new:
            kwargs["min_new_tokens"] = min_new
        inputs = judge.processor(text=[text], videos=[str(video_path)],
                                 num_frames=judge.num_frames,
                                 return_tensors="pt")
        inputs = {k: (v.bfloat16() if v.dtype == torch.float else v)
                  for k, v in inputs.items()}
        inputs = {k: v.to(judge.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = judge.model.generate(**inputs, **kwargs)
        return judge.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

    captions = build_caption_manifest(dataset)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    report, done = [], 0

    for source_key in list_source_videos(dataset, limit=max(n_clips * 4, 12)):
        if done >= n_clips:
            break
        stem = Path(source_key).stem
        caption = lookup_caption(dataset, captions, source_key)
        rec = get_json(f"{RESULT_PREFIX}/{judge.name}/{dataset}/{stem}.json")
        call = ((rec or {}).get("runs", {}).get("clean", {})
                .get("calls", {}).get("PC", {}))
        raw1, score1 = call.get("raw"), call.get("parsed")
        if caption is None or score1 is None:
            print(f"skip {stem[:44]}: no caption or no pass-1 clean PC score")
            continue

        # splice what the model actually said in pass 1; fall back to the
        # parsed digit if the raw reply is unexpectedly long
        splice = raw1.strip() if raw1 and len(raw1.strip()) <= 8 else str(score1)
        wrong = "1" if splice != "1" else "5"
        transcript = PROMPT_PHYSICS + splice + "\n"
        followup = transcript + "Human: " + VP2_EXPLAIN_TURN + "\nAI: "
        followup_wrong = (PROMPT_PHYSICS + wrong + "\nHuman: "
                          + VP2_EXPLAIN_TURN + "\nAI: ")

        modes = [
            ("nle", VP2_PROMPT_NLE.format(caption=caption), None),
            ("explain", followup, None),
            ("explain_wrong", followup_wrong, None),
            ("explain_forced", followup, 40),
            ("prefill", followup + "I rated it %s because" % splice, None),
            ("describe", VP2_PREAMBLE + "Human: <|video|>\n"
             "Human: Describe what happens in the video.\nAI: ", None),
        ] + [
            ("rule_%d" % i, VP2_PROMPT_RULE.format(rule=r), None)
            for i, r in enumerate(VP2_PROBE_RULES)
        ]

        local = TMP_DIR / f"probe__{safe_local_name(Path(source_key).name)}"
        s3.download_file(BUCKET, source_key, str(local))
        print("\n" + "=" * 72)
        print(f"{stem}\n  caption: {caption[:80]}\n  pass-1 PC: {raw1!r} -> "
              f"{score1}   (wrong-digit plant: {wrong})")
        entry = {"stem": stem, "caption": caption, "pass1_raw": raw1,
                 "pass1_pc": score1, "wrong_plant": wrong, "modes": {}}
        try:
            for name, text, min_new in modes:
                if done == 0:
                    print(f"\n--- {name}: exact prompt ---\n{text!r}")
                out = generate(text, local, min_new)
                entry["modes"][name] = {"prompt": text, "raw": out}
                print(f"  {name:15s} len {len(out):4d}  {out[:100]!r}")
        finally:
            local.unlink(missing_ok=True)
        report.append(entry)
        done += 1

    if push_to_s3 and report:
        key = f"{PASS2_RESULT_PREFIX}/videophy2_auto/_rationale_probe.json"
        put_json(key, {"dataset": dataset, "clips": report})
        print(f"\nprobe -> s3://{BUCKET}/{key}")
    return report


def video_key_for(dataset, source_key, variant):
    if variant == "clean":
        return source_key
    return f"attacks/{dataset}/{Path(source_key).stem}/{variant}.mp4"


def _paraphrase_phyjudge_cfg(cfg, k):
    """Swap phyjudge's 5 YAML templates for reworded index k.

    phyjudge's prompts are the one set that does not live in this file --
    infer.py renders them from subq+human.yaml -- so the swap is a recursive
    string replace on the loaded cfg rather than an edit to a constant here.
    Raises if a template is not found: silently scoring with the ORIGINAL
    prompt under a paraphrase prefix would be indistinguishable from a
    paraphrase that simply had no effect, which is the result being measured.
    """
    import copy

    P = _paraphrases()
    pairs = [(e["original"], e["paraphrases"][k])
             for e in P.PHYJUDGE_PROMPTS.values()]
    hits = {old: 0 for old, _ in pairs}

    def walk(o):
        if isinstance(o, dict):
            return {kk: walk(vv) for kk, vv in o.items()}
        if isinstance(o, list):
            return [walk(vv) for vv in o]
        if isinstance(o, str):
            for old, new in pairs:
                if old in o:
                    hits[old] += 1
                    o = o.replace(old, new)
        return o

    out = walk(copy.deepcopy(cfg))
    missed = [old[:40] for old, c in hits.items() if c == 0]
    if missed:
        raise RuntimeError(
            "paraphrase: %d phyjudge template(s) not found in the loaded YAML "
            "(%s); paraphrases.py has drifted from subq+human.yaml"
            % (len(missed), missed))
    return out


def result_key(model, dataset, source_key):
    if PARAPHRASE is not None:
        prefix = f"{PARAPHRASE_RESULT_PREFIX}/p{PARAPHRASE}"
    elif PASS2:
        prefix = PASS2_RESULT_PREFIX
    else:
        prefix = RESULT_PREFIX
    return f"{prefix}/{model}/{dataset}/{Path(source_key).stem}.json"


def missing_items(record, call_ids, variants=None):
    """Outstanding (variant, [call_id]) pairs. `variants` narrows the run to a
    named subset; None means all of VARIANTS.

    Narrowing changes what "already complete" means -- a clip with clean and
    shuffle done counts as finished even though seven variants were never
    touched. That is intended for a subset run, and it is also why the subset
    belongs in the caller's command rather than in a default.
    """
    runs = record.get("runs", {}) if record else {}
    missing = []
    for variant in (variants or VARIANTS):
        done = runs.get(variant, {}).get("calls", {})
        todo = [c for c in call_ids if c not in done]
        if todo:
            missing.append((variant, todo))
    return missing


def refresh_unparsed(record):
    record["unparsed"] = [
        f"{v}/{cid}" for v, run in record["runs"].items()
        for cid, out in run["calls"].items() if out.get("parsed") is None
    ]
    return record


def process_clip(judge, dataset, source_key, caption, items, push_to_s3):
    """Score one clip's outstanding calls. Returns (record, calls_attempted).

    The count is attempts, not successes: a call that raises still consumed
    model time, so it belongs in the rate. Calls skipped because a variant was
    never rendered are *not* counted -- they cost nothing and run_judges takes
    them back off the pending total instead.
    """
    key = result_key(judge.name, dataset, source_key)
    record = get_json(key) or {
        "model": judge.name, "dataset": dataset, "clip": Path(source_key).stem,
        "source_key": source_key, "caption": caption,
        "pass": 2 if PASS2 else 1, "runs": {},
    }

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    calls_run = 0
    for variant, call_ids in items:
        video_key = video_key_for(dataset, source_key, variant)
        if not key_exists(video_key):
            print(f"  missing video s3://{BUCKET}/{video_key} - run attack_suite first")
            continue
        local = TMP_DIR / f"{os.getpid()}__{safe_local_name(Path(video_key).name)}"
        s3.download_file(BUCKET, video_key, str(local))
        try:
            run = record["runs"].setdefault(
                variant, {"video_key": video_key, "caption": caption, "calls": {}})
            setter = getattr(judge, "set_context", None)
            if setter is not None:
                setter(dataset, source_key, variant)
            for call_id in call_ids:
                calls_run += 1
                try:
                    raw, parsed = judge.run(local, caption, call_id)
                    run["calls"][call_id] = {"raw": raw, "parsed": parsed}
                except Exception as e:
                    print(f"  FAILED {variant}/{call_id}: {e}")
            print(f"  {variant}: " + str(
                {c: run["calls"].get(c, {}).get("parsed") for c in call_ids}))
            if push_to_s3:
                put_json(key, refresh_unparsed(record))
        finally:
            local.unlink(missing_ok=True)

    refresh_unparsed(record)
    if push_to_s3:
        put_json(key, record)
        print(f"  -> s3://{BUCKET}/{key}")
    return record, calls_run


def results_frame(records):
    import pandas as pd

    rows = []
    for rec in records:
        for variant, run in rec["runs"].items():
            for cid, out in run["calls"].items():
                rows.append({
                    "clip": rec["clip"], "model": rec["model"], "variant": variant,
                    "call": cid, "parsed": out.get("parsed"),
                    "raw": (out.get("raw") or "").replace("\n", " ")[:120],
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # ALL_VARIANTS, not VARIANTS: pd.Categorical maps anything outside
    # `categories` to NaN, so an identity run would silently blank its own
    # variant column and sort into a single undifferentiated block.
    df["variant"] = pd.Categorical(df["variant"], categories=ALL_VARIANTS,
                                   ordered=True)
    return df.sort_values(["clip", "model", "variant", "call"]).reset_index(drop=True)


def show_results(df):
    try:
        from IPython.display import display
    except ImportError:
        display = print

    if df.empty:
        print("no results")
        return
    for (clip, model), group in df.groupby(["clip", "model"], observed=True):
        print(f"\n=== {clip} | {model} ===")
        table = group.pivot(index="variant", columns="call", values="parsed")
        display(table.dropna(how="all"))
    n_unparsed = int(df["parsed"].isna().sum())
    print(f"\n{len(df)} calls, {n_unparsed} unparsed")


def run_judges(dataset="test", num_clips=1, push_to_s3=True, models=None,
               rebuild_captions=False, show=None, shard=None,
               require_attacks=True, pass2=False, paraphrase=None,
               variants=None, require_pass1=None):
    """pass2=True asks each judge to score AND justify, in one generation.

    `variants` narrows the run to a named subset of VARIANTS (e.g.
    ("clean", "shuffle")); None runs all ten. A subset also tightens
    require_attacks, which otherwise only checks that a variant DIRECTORY
    exists -- see clips_with_variants.

    `require_pass1` is a judge name: only clips whose results/pass1 record for
    that judge is complete over the requested variants are eligible. Pass 2 is
    only ever read as a difference against pass 1, so a clip with no pass-1
    side to difference against is 32 phyjudge generations bought for nothing.
    Both filters run BEFORE the num_clips limit and the shard, so num_clips
    keeps counting clips that can actually produce a comparison.

    Pass 2 is score + rationale, NOT chain-of-thought. Each judge gets its own
    pass-1 prompt with its FORMAT INSTRUCTION rewritten to ask for the answer
    and then a justification -- see PASS2_REWRITES for why appending a sentence
    after it does not work. Each judge produces its own score followed by the
    justification. Answer first,
    explanation after: reasoning before answering would change the score, and
    whether requiring a justification changes the score is precisely what the
    two passes exist to compare -- so pass 2 must generate its own score
    rather than be shown pass 1's, which would only anchor it.

    The reasoning configuration is identical between passes and across all
    three judges: no thinking mode, no step-by-step instruction, one shared
    request. Scores are parsed POSITIONALLY in pass 2 -- see the note by
    _P2_YESNO_RE for why pass 1's parsers cannot survive trailing prose.

    Results go to PASS2_RESULT_PREFIX, never merged into results/pass1.
    """
    global PASS2, PARAPHRASE
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}")
    models = models or list(JUDGES)
    show = (not push_to_s3) if show is None else show

    if variants is not None:
        variants = list(variants)
        unknown = [v for v in variants if v not in ALL_VARIANTS]
        if unknown:
            raise ValueError(f"unknown variant(s) {unknown}; "
                             f"expected a subset of {ALL_VARIANTS}")
        if not variants:
            raise ValueError("variants is empty; omit it to run all of VARIANTS")
        print(f"variants: {', '.join(variants)} "
              f"({len(VARIANTS) - len(variants)} of {len(VARIANTS)} skipped)")
    if require_pass1 is not None and require_pass1 not in JUDGES:
        raise ValueError(f"require_pass1 must be one of {list(JUDGES)}")

    if paraphrase is not None:
        if pass2:
            raise ValueError(
                "paraphrase and pass2 are separate experiments and share no "
                "prefix; run them one at a time")
        P = _paraphrases()
        if not isinstance(paraphrase, int) or not 0 <= paraphrase < P.N_PARAPHRASES:
            raise ValueError("paraphrase must be an int in 0..%d"
                             % (P.N_PARAPHRASES - 1))
        problems = P.check(verbose=False)
        if problems:
            raise RuntimeError(
                "paraphrases.py failed validation (%d problem(s)); fix before "
                "spending GPU time: %s" % (len(problems), problems[:3]))
    PARAPHRASE = paraphrase
    if PARAPHRASE is not None:
        print(f"PARAPHRASE {PARAPHRASE} -- same calls/decoding/parsing as pass 1, "
              f"reworded prompts -> "
              f"s3://{BUCKET}/{PARAPHRASE_RESULT_PREFIX}/p{PARAPHRASE}/")
    PASS2 = bool(pass2)
    if PASS2:
        print(f"PASS 2 (score + rationale) -- writing to "
              f"s3://{BUCKET}/{PASS2_RESULT_PREFIX}/ (pass 1 is untouched)")

    captions = build_caption_manifest(dataset, rebuild=rebuild_captions)
    source_keys = list_source_videos(dataset)
    if not source_keys:
        print("no source videos found")
        return results_frame([])

    # Filter BEFORE the num_clips limit, so num_clips counts clips that can
    # actually be scored rather than positions in the raw listing.
    if require_attacks:
        # a named subset needs those exact renders, not just a directory
        have = (clips_with_attacks(dataset) if variants is None
                else clips_with_variants(dataset, variants))
        before = len(source_keys)
        if have is not None:
            source_keys = [k for k in source_keys if Path(k).stem in have]
        want = "rendered attacks" if variants is None else \
            "every requested variant rendered"
        print(f"{len(source_keys)} of {before} source clips have {want}"
              f" ({before - len(source_keys)} skipped;"
              f" pass require_attacks=False to score them clean-only)")
        if not source_keys:
            print(f"nothing under attacks/{dataset}/ - run attack_suite first")
            return results_frame([])

    if require_pass1 is not None:
        # call_ids() is static on all three judges -- no weights load here
        need = JUDGES[require_pass1]().call_ids()
        scored = clips_with_pass1(dataset, require_pass1, need,
                                  variants or VARIANTS,
                                  [Path(k).stem for k in source_keys])
        before = len(source_keys)
        source_keys = [k for k in source_keys if Path(k).stem in scored]
        print(f"{len(source_keys)} of {before} clips kept "
              f"(complete pass-1 {require_pass1} record)")
        if not source_keys:
            print(f"no clip has a complete pass-1 record for {require_pass1} "
                  f"- run pass 1 on this dataset first")
            return results_frame([])

    if num_clips and len(source_keys) < num_clips:
        print(f"WARNING only {len(source_keys)} eligible clips, fewer than the "
              f"{num_clips} asked for; running all of them")
    source_keys = source_keys[:num_clips] if num_clips else source_keys
    # After the limit, never before: the stripes then cover exactly the clips a
    # single unsharded run of the same num_clips would have taken.
    if shard is not None:
        total = len(source_keys)
        source_keys = shard_keys(source_keys, shard)
        print(f"shard {shard[0]}/{shard[1]}: {len(source_keys)} of {total} clips")

    records = []
    for model in models:
        if PARAPHRASE is not None:
            judge = JUDGES[model](paraphrase=PARAPHRASE)
        else:
            judge = JUDGES[model](pass2=True) if PASS2 else JUDGES[model]()
        call_ids = judge.call_ids()
        shard_note = f", shard={shard[0]}/{shard[1]}" if shard else ""
        print(f"\n########## {judge.name} (dataset={dataset}, clips={num_clips}, "
              f"calls/variant={len(call_ids)}, push={push_to_s3}{shard_note}) ##########")

        todo, done_records, skipped = [], [], 0
        for source_key in source_keys:
            caption = lookup_caption(dataset, captions, source_key)
            if caption is None:
                print(f"  no caption for {source_key}, skipping")
                continue
            existing = get_json(result_key(judge.name, dataset, source_key))
            missing = missing_items(existing, call_ids, variants)
            if missing:
                todo.append((source_key, caption, missing))
            else:
                skipped += 1
                done_records.append(existing)
        records.extend(done_records)

        print(f"{len(source_keys)} clips selected, {skipped} already complete, "
              f"{len(todo)} to process")
        if not todo:
            continue

        pending = sum(len(cids) for _, _, items in todo for _, cids in items)
        print(f"{pending} generations to run")

        try:
            print(f"[{judge.name}] loading")
            judge.load()
        except Exception as e:
            print(f"[{judge.name}] SKIPPED: {e}")
            continue

        # after load(): weight download and model init are one-off and would
        # otherwise poison the per-call rate for the whole run
        t_start = time.perf_counter()
        done_calls = 0

        for i, (source_key, caption, items) in enumerate(todo, 1):
            print(f"\n=== [{i}/{len(todo)}] {source_key} ===")
            planned = sum(len(c) for _, c in items)
            try:
                record, ran = process_clip(judge, dataset, source_key, caption,
                                           items, push_to_s3)
                records.append(record)
            except Exception as e:
                ran = 0
                print(f"FAILED (clip-level) {source_key}: {e}")

            done_calls += ran
            # Calls that never ran -- a dead clip, or a variant that was never
            # rendered -- come off the total rather than counting as done, so
            # the remaining count stays honest and the rate stays real.
            pending -= planned - ran
            elapsed = time.perf_counter() - t_start
            rate = elapsed / done_calls if done_calls else 0.0
            eta = max(pending - done_calls, 0) * rate
            print(f"[{i:>4}/{len(todo)}] {ran}/{planned} calls  "
                  f"{done_calls:>6}/{pending}  {rate:5.1f}s/call  "
                  f"elapsed {fmt_secs(elapsed)}  eta {fmt_secs(eta)}  "
                  f"{Path(source_key).stem[:40]}")

        total = time.perf_counter() - t_start
        print(f"[{judge.name}] {done_calls} generations in {fmt_secs(total)}"
              + (f" ({total/done_calls:.1f}s/call)" if done_calls else ""))
        del judge

    df = results_frame(records)
    if show:
        show_results(df)
    print("\nDone.")
    return df
