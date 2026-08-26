"""Print exactly what each judge is handed, pass 1 vs pass 2. CPU only.

No model is loaded and no GPU is touched: build_prompt is pure string work and
judge_harness defers every heavy import into each judge's load().

Colab:
    !wget -q -O judge_harness.py https://raw.githubusercontent.com/jamesk837/algoverse_2026/main/judge_harness.py
    !wget -q -O show_prompts.py  https://raw.githubusercontent.com/jamesk837/algoverse_2026/main/show_prompts.py
    !python show_prompts.py

phyjudge needs its prompts pulled from S3 (they live in the mirrored repo, not
here), so that section needs credentials. The other two need nothing.
"""
import os
import sys

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

CAPTION = "a ball rolls off a table and falls to the floor"
BAR = "=" * 78

import judge_harness as J


def show(title, text):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)
    print(text)


# ---------------------------------------------------------------- vila -----
print(BAR)
print("vila_ewm  --  8 calls per variant")
print(BAR)
a, b = J.VilaEwmJudge(), J.VilaEwmJudge(pass2=True)
for cid in a.call_ids():
    p1, p2 = a.build_prompt(cid, CAPTION), b.build_prompt(cid, CAPTION)
    show("%s   PASS 1" % cid, p1)
    show("%s   PASS 2" % cid, p2)
    added = p2.replace(p1, "")
    print("\n   >>> pass 2 adds exactly: %r" % added)
print("\n   all %d vila calls get the same sentence appended, once each"
      % len(a.call_ids()))

# ----------------------------------------------------------- videophy2 -----
print("\n" + BAR)
print("videophy2_auto  --  2 calls per variant")
print(BAR)
a, b = J.VideoPhy2AutoJudge(), J.VideoPhy2AutoJudge(pass2=True)
for cid in a.call_ids():
    p1, p2 = a.build_prompt(cid, CAPTION), b.build_prompt(cid, CAPTION)
    show("%s   PASS 1" % cid, repr(p1))
    show("%s   PASS 2" % cid, repr(p2))

# ------------------------------------------------------------ phyjudge -----
print("\n" + BAR)
print("phyjudge_9b  --  16 calls per variant")
print(BAR)
print("Its prompts are rendered by the mirrored repo's infer.py, not by this")
print("file, so they have to come from S3.\n")

try:
    from pathlib import Path

    import boto3

    s3 = boto3.client("s3")
    dest = Path("./phyjudge_prompt_src")
    dest.mkdir(exist_ok=True)
    got = []
    pg = s3.get_paginator("list_objects_v2")
    for page in pg.paginate(Bucket=J.BUCKET, Prefix="models/phyjudge-9B/"):
        for o in page.get("Contents", []):
            k = o["Key"]
            if k.endswith(("infer.py", ".yaml", ".yml")):
                local = dest / Path(k).name
                s3.download_file(J.BUCKET, k, str(local))
                got.append(local)
    print("pulled: %s" % [p.name for p in got])

    sys.path.insert(0, str(dest))
    import infer                                   # may need torch; Colab has it

    yml = [p for p in got if p.suffix in (".yaml", ".yml")][0]
    # load_yaml calls .open() on what it is given, so it wants a Path, not a
    # str -- judge_harness passes `adapter_dir / PHYJUDGE_PROMPT_YAML`. Try the
    # str form too in case a vendored copy differs.
    try:
        cfg = infer.load_yaml(yml)
    except AttributeError:
        cfg = infer.load_yaml(str(yml))
    for metric, law in (("SA", None), (None, "gravity")):
        sysp, userp, score_key = infer.build_prompt(cfg, CAPTION,
                                                    metric=metric, law=law)
        label = metric or law
        show("%s   PASS 1  (system)" % label, sysp)
        show("%s   PASS 1  (user)" % label, userp)
        p2 = J._apply_rewrite(userp, "phyjudge_only")
        anchor = "with exactly one key: %s." % score_key
        p2 = p2.replace(anchor, anchor[:-1] + J.PHYJUDGE_RATIONALE_SUFFIX, 1)
        show("%s   PASS 2  (user)" % label, p2)
        print("\n   >>> score_key the parser looks for: %r" % score_key)
except Exception as exc:
    print("could not render phyjudge's prompts: %s: %s" % (type(exc).__name__, exc))
    print("\nFalling back to the raw YAML template. {questions_block} is filled")
    print("by infer.py from its sub-question checklists, so it shows unfilled:\n")
    try:
        import re

        import boto3

        s3 = boto3.client("s3")
        pg = s3.get_paginator("list_objects_v2")
        for page in pg.paginate(Bucket=J.BUCKET, Prefix="models/phyjudge-9B/"):
            for o in page.get("Contents", []):
                if o["Key"].endswith((".yaml", ".yml")):
                    body = s3.get_object(Bucket=J.BUCKET,
                                         Key=o["Key"])["Body"].read().decode()
                    print(body)
                    raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc2:
        print("  and the YAML could not be read either: %s" % exc2)

print("\n" + BAR)
print("pass 2 REWRITES each judge's format instruction (appending after it does")
print("not work -- every judge states its output contract emphatically and last):")
for k, (old, new) in J.PASS2_REWRITES.items():
    print("  %-14s %r" % (k, old))
    print("  %-14s -> %r" % ("", new))
print(BAR)
