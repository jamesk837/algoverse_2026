import json
import os
import re
import time
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
    os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('AWS_ACCESS_KEY_ID')
    os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
    try:
        os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    except Exception:
        print("no HF_TOKEN secret; phyjudge_9b base model must be public or cached")
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')

BUCKET = "nickb-aarj"
s3 = boto3.client('s3')
MODEL_CACHE = Path("./models_cache")
TMP_DIR = Path("./tmp_judge")
RESULT_PREFIX = "results/pass1"

# CoT runs write to their OWN prefix. This is the load-bearing part of the
# whole option: checkpointing is per (clip, variant, call), so if a CoT run
# shared results/pass1 it would merge CoT answers into the non-CoT records
# call by call, silently destroying both the existing data and the comparison
# the flag exists to make. Same layout underneath, so check_results / stats /
# monitor read it by pointing their own RESULT_PREFIX here.
COT_RESULT_PREFIX = "results/pass1_cot"
COT = False  # set by run_judges(cot=...); read by result_key()
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

VARIANTS = ["clean"] + ATTACK_FILES

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

# The ONLY prompts in this file that are not upstream's. VideoPhy-2 ships no
# CoT form -- these are written here, following the wording of arXiv 2503.03064
# ("Begin your evaluation by providing a short explanation. Be as objective as
# possible. After providing your explanation, please rate ...") and adapted to
# VideoPhy-2's 1-5 scale and its two questions. The rating sentence itself is
# left as upstream wrote it; the explanation instruction is inserted before it
# and an explicit "Score: <number>" anchor is appended after it.
#
# That anchor is not decoration. vp2_parse substring-matches number WORDS, so
# on prose it reads "the video shows one ball" as a score of 1 -- a wrong
# number that looks like a real one. CoT runs therefore use vp2_parse_cot,
# which accepts nothing but the explicit anchor.
#
# Unlike vila and phyjudge, this condition has no upstream to be faithful to,
# so SMOKE-TEST IT: run one clip with push_to_s3=False and read the raw text
# before spending GPU hours.
_VP2_COT_INSTRUCTION = (
    "Begin your evaluation by providing a short explanation. "
    "Be as objective as possible. After providing your explanation, "
)
_VP2_COT_ANCHOR = " End your reply with \"Score: <number>\"."

PROMPT_SA_COT = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: Does this video match the description: \"{caption}\"? "
    + _VP2_COT_INSTRUCTION +
    "rate the video on a scale from 1 to 5, where 5 indicates a perfect match "
    "and 1 indicates no relevance."
    + _VP2_COT_ANCHOR + "\n"
    "AI: "
)

PROMPT_PHYSICS_COT = (
    "The following is a conversation between a curious human and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\n"
    "Human: Does this video adhere to the physical laws? "
    + _VP2_COT_INSTRUCTION +
    "rate the video on a scale from 1 to 5, where 5 means full compliance "
    "and 1 means significant violations."
    + _VP2_COT_ANCHOR + "\n"
    "AI: "
)

# max_length counts prompt + completion, so an explanation needs headroom.
VP2_GENERATE_KWARGS_COT = dict(VP2_GENERATE_KWARGS, max_length=512)

# Only an explicit "Score: N" with N in 1..5. Last match wins, so a model that
# restates the scale mid-explanation and commits at the end is read correctly.
VP2_SCORE_RE = re.compile(r"score\s*[:=]\s*([1-5])(?![0-9])", re.IGNORECASE)
VP2_NUM_FRAMES = 32

VP2_NUM_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}

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
# With Qwen3 thinking on, the reasoning trace comes BEFORE the JSON, so the
# non-CoT budget truncates every call mid-thought and each one parses as None.
# Greedy decoding makes a large budget free -- identical tokens either way, it
# only removes the ceiling.
PHYJUDGE_MAX_NEW_TOKENS_COT = 1024
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


def vp2_parse_cot(output):
    """Strict counterpart to vp2_parse, for CoT output.

    Returns None rather than guessing. That is the point: vp2_parse's loose
    substring match is exactly what makes prose unparseable-but-plausible, so
    the fix cannot be a slightly-less-loose version of it. An unparsed call is
    recorded in record["unparsed"] and is visible; a wrong score is not.
    """
    if not output:
        return None
    found = VP2_SCORE_RE.findall(output)
    return int(found[-1]) if found else None


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
    # upstream ships both forms; `--cot` is store_true, so its default is off
    supports_cot = True

    def __init__(self, cot=WMB_COT):
        self.cot = cot
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
            prompt = WMB_PROMPT_TEMPLATES["instruction"].format(instruction=caption)
        else:
            eval_type, idx = call_id.rsplit("_", 1)
            question = WMB_QUESTION_POOL[eval_type][int(idx)]
            prompt = WMB_PROMPT_TEMPLATES[eval_type].format(**{eval_type: question.lower()})
        if not self.cot:
            prompt = prompt.replace(
                "Let's think step-by-step and conclude with", "Answer with"
            ).replace(
                "Let's analyze step-by-step and conclude with", "Answer with"
            )
        return prompt

    def parse(self, call_id, pred):
        if call_id == "instruction":
            try:
                return float(pred.split(":")[-1].strip(" ."))
            except ValueError:
                return None
        return "no" in pred.lower()

    def run(self, video_path, caption, call_id):
        prompt = self.build_prompt(call_id, caption)
        video = self.llava.Video(str(video_path))
        pred = str(self.judge.generate_content([video, prompt]))
        return pred, self.parse(call_id, pred)


class VideoPhy2AutoJudge:
    name = "videophy2_auto"
    s3_prefix = "models/videophy_2_auto/"
    # The one judge whose CoT condition is OURS, not upstream's -- VideoPhy-2
    # ships no CoT form. It is entailment-style mPLUG-Owl-Video completing
    # "AI: " after a fixed conversation string, SFT'd on that exact template,
    # so its CoT numbers are off-distribution in a way vila's and phyjudge's
    # are not. Report them as a separate condition, never pooled with those.
    supports_cot = True

    def __init__(self, num_frames=VP2_NUM_FRAMES, cot=False):
        self.num_frames = num_frames
        self.cot = cot
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
        if self.cot:
            return (PROMPT_SA_COT.format(caption=caption) if call_id == "SA"
                    else PROMPT_PHYSICS_COT)
        if call_id == "SA":
            return PROMPT_SA.format(caption=caption)
        return PROMPT_PHYSICS

    def run(self, video_path, caption, call_id):
        torch = self.torch
        prompts = [self.build_prompt(call_id, caption)]
        inputs = self.processor(text=prompts, videos=[str(video_path)],
                                num_frames=self.num_frames, return_tensors="pt")
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v
                  for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs = VP2_GENERATE_KWARGS_COT if self.cot else VP2_GENERATE_KWARGS
        with torch.no_grad():
            res = self.model.generate(**inputs, **gen_kwargs)
        output = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        # strict parser under CoT: vp2_parse would read "one ball" as a 1
        return output, (vp2_parse_cot(output) if self.cot else vp2_parse(output))


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
    # Qwen3's native thinking mode, enabled on the chat template. The
    # prompts stay exactly what the repo's infer.py renders.
    supports_cot = True

    def __init__(self, fps=PHYJUDGE_FPS, max_pixels=PHYJUDGE_MAX_PIXELS,
                 max_new_tokens=None, cot=False):
        self.fps = fps
        self.max_pixels = max_pixels
        self.cot = cot
        # CoT here is Qwen3's OWN thinking mode, not a prompt we wrote. The
        # prompts still come from the repo's infer.py, untouched.
        self.max_new_tokens = max_new_tokens or (
            PHYJUDGE_MAX_NEW_TOKENS_COT if cot else PHYJUDGE_MAX_NEW_TOKENS)
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
            enable_thinking=self.cot)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt", **video_kwargs)
        return inputs.to(self.device)

    def run(self, video_path, caption, call_id):
        infer = self.infer
        metric = call_id if call_id in PHYJUDGE_GENERAL_KEYS else None
        law = None if metric else call_id
        system_prompt, user_prompt, score_key = infer.build_prompt(
            self.cfg, caption, metric=metric, law=law)
        messages = infer.build_messages(system_prompt, user_prompt, Path(video_path))
        inputs = self.prepare_inputs(messages)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        raw = infer.decode_generated(self.processor, inputs, generated_ids)
        return raw, infer.parse_score(raw, score_key)


JUDGES = {
    "vila_ewm": VilaEwmJudge,
    "videophy2_auto": VideoPhy2AutoJudge,
    "phyjudge_9b": PhyJudge9BJudge,
}

def video_key_for(dataset, source_key, variant):
    if variant == "clean":
        return source_key
    return f"attacks/{dataset}/{Path(source_key).stem}/{variant}.mp4"


def result_key(model, dataset, source_key):
    prefix = COT_RESULT_PREFIX if COT else RESULT_PREFIX
    return f"{prefix}/{model}/{dataset}/{Path(source_key).stem}.json"


def missing_items(record, call_ids):
    runs = record.get("runs", {}) if record else {}
    missing = []
    for variant in VARIANTS:
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
        "source_key": source_key, "caption": caption, "pass": 1, "runs": {},
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
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANTS, ordered=True)
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
               require_attacks=True, cot=False):
    """cot=True runs the judges' OWN chain-of-thought forms and writes to
    COT_RESULT_PREFIX, never mixing with the greedy records in results/pass1.

    Two of the three use their own CoT form, unmodified. vila_ewm simply stops
    rewriting upstream's "Let's think step-by-step and conclude with ..." into
    "Answer with" -- WorldModelBench ships both, and `--cot` being store_true
    is the only reason off is the default. phyjudge_9b turns on Qwen3's own
    thinking mode and raises the token budget so the JSON is not truncated
    behind the reasoning.

    videophy2_auto is the exception and must be read as one: VideoPhy-2 ships
    no CoT form, so PROMPT_*_COT are OURS. Its CoT numbers are off-distribution
    in a way the other two are not -- report them as a separate condition,
    never pooled with vila's or phyjudge's. It also swaps in vp2_parse_cot,
    because the greedy parser substring-matches number words and would read
    "the video shows one ball" as a score of 1.

    Motivation is arXiv 2503.03064: CoT sharpens the judgment distribution
    (they measure a strictly lower standard deviation) and shifts the mean, so
    a judge's gameability under CoT is a different measurement, not a better
    one. That is why this is a parallel prefix and not a replacement.
    """
    global COT
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}")
    models = models or list(JUDGES)
    show = (not push_to_s3) if show is None else show

    COT = bool(cot)
    if COT:
        blocked = [m for m in models if not getattr(JUDGES[m], "supports_cot", False)]
        if blocked:
            print(f"cot=True: skipping {blocked} -- no upstream CoT form; "
                  f"see supports_cot in judge_harness.py")
            models = [m for m in models if m not in blocked]
        if not models:
            print("cot=True: nothing left to run")
            return results_frame([])
        print(f"CoT ON -- writing to s3://{BUCKET}/{COT_RESULT_PREFIX}/ "
              f"(results/pass1 is untouched)")

    captions = build_caption_manifest(dataset, rebuild=rebuild_captions)
    source_keys = list_source_videos(dataset)
    if not source_keys:
        print("no source videos found")
        return results_frame([])

    # Filter BEFORE the num_clips limit, so num_clips counts clips that can
    # actually be scored rather than positions in the raw listing.
    if require_attacks:
        have = clips_with_attacks(dataset)
        before = len(source_keys)
        source_keys = [k for k in source_keys if Path(k).stem in have]
        print(f"{len(source_keys)} of {before} source clips have rendered attacks"
              f" ({before - len(source_keys)} skipped;"
              f" pass require_attacks=False to score them clean-only)")
        if not source_keys:
            print(f"nothing under attacks/{dataset}/ - run attack_suite first")
            return results_frame([])
    source_keys = source_keys[:num_clips] if num_clips else source_keys
    # After the limit, never before: the stripes then cover exactly the clips a
    # single unsharded run of the same num_clips would have taken.
    if shard is not None:
        total = len(source_keys)
        source_keys = shard_keys(source_keys, shard)
        print(f"shard {shard[0]}/{shard[1]}: {len(source_keys)} of {total} clips")

    records = []
    for model in models:
        judge = JUDGES[model](cot=True) if COT else JUDGES[model]()
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
            missing = missing_items(existing, call_ids)
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
