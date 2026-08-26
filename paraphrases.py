"""Prompt paraphrases for the three judges -- FILL IN THE BLANKS.

Every judge was SFT'd on its exact template, so this measures how much a
judge's score moves when the rubric is reworded. Note the standing caveat
in CLAUDE.md: that is partly off-distribution drift rather than pure
gameability. It is README Step 2 and was previously out of scope.

HOW TO USE
    Each entry has an `original` (do not edit -- it is checked against
    judge_harness at runtime and the check fails if they drift apart) and a
    `paraphrases` list of 5 empty strings. Write your rewordings there.

RULES, in order of how badly they break things

  1. PLACEHOLDERS MUST SURVIVE. {caption}, {instruction},
     {physical_laws} and {common_sense} are substituted at run time.
     Exactly one of the right one per template, spelled identically.

  2. KEEP THE ANSWER FORMAT. The parsers read what the judge emits, not
     what you asked for. If you drop 'Score: [score]' or '"Yes" or "No"'
     or 'scale from 1 to 5', the reply stops parsing and every call lands
     in record["unparsed"]. Reword the QUESTION, keep the FORMAT.

  3. KEEP THE SCALE. WorldModelBench instruction is 0-3. VideoPhy-2 is
     1-5. Do not renumber them.

  4. Indentation and the trailing wrapper are handled for you. Write the
     vila templates as plain multi-line text; _wmb_template restores the
     12-space continuation indent. Write the videophy2 entries as the
     INSTRUCTION SENTENCE ONLY -- the system line, the <|video|> token and
     the trailing "AI: " are structural and get reattached.

  5. Leave a slot "" to skip it. check() reports how many are filled.

CHECK YOUR WORK BEFORE A RUN
    python paraphrases.py            # validates and prints a summary
"""

N_PARAPHRASES = 5


# ===========================================================================
# vila_ewm -- 3 templates
# ---------------------------------------------------------------------------
# The template wraps a question from the pool below. The final line is the
# answer-format instruction: if you change its shape, VilaEwmJudge.parse
# stops working. The 0-3 criteria list is part of the rubric -- rewording the
# descriptions is fair game, renumbering them is not.
# ===========================================================================

VILA_TEMPLATES = {
    "instruction": {
        "original": """\
Evaluate if this video follows the instruction: '{instruction}'.
            Use the following scoring criteria:

            - 0: The video does not follow the instruction at all.
            - 1: The video includes the correct object but performs the wrong action, or vice versa.
            - 2: The video follows the instruction and shows a tendency toward the intended goal.
            - 3: The video follows the instruction precisely and successfully achieves the goal.

            Let's analyze step-by-step and conclude with 'Score: [score]'.
""",
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
    "physical_laws": {
        "original": """\
Watch the video and determine if it shows any '{physical_laws}'
            Let's think step-by-step and conclude with "Yes" or "No".
""",
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
    "common_sense": {
        "original": """\
Does the video exhibit '{common_sense}'?
            Let's think step-by-step and conclude with "Yes" or "No".
""",
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
}


# ===========================================================================
# vila_ewm -- 7 questions
# ---------------------------------------------------------------------------
# Inserted into the template above, LOWERCASED automatically. Each names a
# physical law and describes the violation; keep both halves.
# ===========================================================================

VILA_QUESTIONS = {
    "physical_laws_0": {
        "original": "Violation of Newton's Law: Objects move without any external force.",
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
    "physical_laws_1": {
        "original": 'Violation of the Law of Conservation of Mass or Solid Constitutive Law: Objects deform irregularly.',
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
    "physical_laws_2": {
        "original": 'Violation of Fluid Constitutive Law: Liquids flow in an unnatural manner.',
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
    "physical_laws_3": {
        "original": 'Violation of Non-physical Penetration: Objects unnaturally pass through each other.',
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
    "physical_laws_4": {
        "original": 'Violation of Gravity: Objects behave inconsistently with gravity.',
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
    "common_sense_0": {
        "original": 'Poor Aesthetics: Visually unappealing or low-quality content.',
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
    "common_sense_1": {
        "original": 'Temporal Inconsistency: Noticeable flickering or abrupt changes.',
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
}


# ===========================================================================
# videophy2_auto -- 2 instruction sentences
# ---------------------------------------------------------------------------
# THE INSTRUCTION SENTENCE ONLY. The surrounding transcript --
#   "The following is a conversation ... user's questions."
#   "Human: <|video|>"
#   "AI: "
# -- is structural: the <|video|> token is how the clip is fed in and the
# trailing "AI: " is what the model completes. Both are reattached for you.
# Keep "scale from 1 to 5" or the score stops parsing.
# ===========================================================================

VP2_INSTRUCTIONS = {
    "SA": {
        "original": 'Does this video match the description: "{caption}"? Please rate the video on a scale from 1 to 5, where 5 indicates a perfect match and 1 indicates no relevance.',
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
    "PC": {
        "original": 'Does this video adhere to the physical laws? Rate the video on a scale from 1 to 5, where 5 means full compliance and 1 means significant violations.',
        "paraphrases": [
        "",
        "",
        "",
        "",
        "",
    ],
    },
}


# ===========================================================================
# phyjudge_9b -- NOT YET POPULATED
# ---------------------------------------------------------------------------
# Its prompts are not in this repo. PhyJudge9BJudge puts the mirrored repo on
# sys.path and calls that repo's own infer.py against the subq+human.yaml
# shipped beside the adapter, so the text lives in the YAML plus
# infer.GENERAL_SUB_QUESTIONS / infer.PHYSICAL_SUB_QUESTIONS.
#
# Dump it on a box that has the mirror (no GPU needed):
#     cat models_cache/phyjudge_9b/subq+human.yaml
#
# Paste that back and the 3 general dims + 13 laws get slots here.
# ===========================================================================

PHYJUDGE_CRITERIA = {}


# ===========================================================================

_ALL = {
    "vila_templates": VILA_TEMPLATES,
    "vila_questions": VILA_QUESTIONS,
    "vp2_instructions": VP2_INSTRUCTIONS,
    "phyjudge_criteria": PHYJUDGE_CRITERIA,
}

# What each entry must still contain after rewording, per SECTION -- the
# templates carry the placeholders, the questions are substituted INTO a
# template and so have none of their own.
_REQUIRED = {
    "vila_templates": {
        "instruction": ["{instruction}", "Score:"],
        "physical_laws": ["{physical_laws}"],
        "common_sense": ["{common_sense}"],
    },
    "vila_questions": {},          # no placeholder; free text
    "vp2_instructions": {
        "SA": ["{caption}", "1 to 5"],
        "PC": ["1 to 5"],
    },
    "phyjudge_criteria": {},
}


def check(verbose=True):
    """Validate before spending GPU time. Returns a list of problems."""
    problems, filled, total = [], 0, 0

    try:
        import judge_harness as _J
        live = dict(_J.WMB_PROMPT_TEMPLATES)
        for k, v in VILA_TEMPLATES.items():
            if " ".join(v["original"].split()) != " ".join(live[k].split()):
                problems.append("vila template %r original has drifted from "
                                "judge_harness -- re-generate this file" % k)
        pool = _J.WMB_QUESTION_POOL
        for group, qs in pool.items():
            if not qs:
                continue
            for i, q in enumerate(qs):
                key = "%s_%d" % (group, i)
                if VILA_QUESTIONS[key]["original"] != q:
                    problems.append("vila question %r has drifted" % key)
    except Exception as exc:
        problems.append("could not cross-check against judge_harness: %s" % exc)

    for section, entries in _ALL.items():
        for key, entry in entries.items():
            need = _REQUIRED.get(section, {}).get(key, [])
            for i, p in enumerate(entry["paraphrases"]):
                total += 1
                if not p.strip():
                    continue
                filled += 1
                for token in need:
                    if token not in p:
                        problems.append("%s/%s[%d] is missing %r"
                                        % (section, key, i, token))
                if p.strip() == entry["original"].strip():
                    problems.append("%s/%s[%d] is identical to the original"
                                    % (section, key, i))

    if verbose:
        print("paraphrase slots: %d filled / %d total" % (filled, total))
        for section, entries in _ALL.items():
            if not entries:
                print("  %-18s (empty -- not populated yet)" % section)
                continue
            done = sum(1 for e in entries.values()
                       for p in e["paraphrases"] if p.strip())
            print("  %-18s %d entries, %d/%d slots filled"
                  % (section, len(entries), done,
                     len(entries) * N_PARAPHRASES))
        if problems:
            print("\n%d PROBLEM(S):" % len(problems))
            for p in problems:
                print("  - %s" % p)
        else:
            print("\nno problems found")
    return problems


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(1 if check() else 0)
