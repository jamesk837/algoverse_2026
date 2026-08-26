"""Prompt paraphrases for the three judges -- FILL IN THE BLANKS.

Every judge was SFT'd on its exact template, so this measures how much a
judge's score moves when the rubric is reworded. Note the standing caveat
in CLAUDE.md: that is partly off-distribution drift rather than pure
gameability. It is README Step 2 and was previously out of scope.

HOW TO USE
    Each entry has an `original` (do not edit -- it is checked against
    judge_harness at runtime and the check fails if they drift apart) and a
    `paraphrases` list of 2 rewordings: a NEAR one (same structure,
    reworded) and a FAR one (recast syntax), so the pair spans the space
    rather than clustering.

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
     vila templates with the SAME 12-space continuation indent the
     originals use -- _wmb_template only pads blank lines, it does not
     indent content, so whitespace would otherwise vary alongside the
     wording and you could not tell which moved the score. Write the videophy2 entries as the
     INSTRUCTION SENTENCE ONLY -- the system line, the <|video|> token and
     the trailing "AI: " are structural and get reattached.

  6. PHYJUDGE ONLY: keep the DOUBLED braces in the JSON example --
     {{"SA": 3}} is a .format() escape that renders as {"SA": 3}. Keep the
     JSON key name too; infer.parse_score looks for it exactly. And keep
     {questions_block}, which injects the sub-question checklist the model
     was fine-tuned with.

  5. Leave a slot "" to skip it. check() reports how many are filled.

CHECK YOUR WORK BEFORE A RUN
    python paraphrases.py            # validates and prints a summary
"""

N_PARAPHRASES = 2


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
            """\
Assess whether the video adheres to the instruction: '{instruction}'.
            Apply this scoring rubric:

            - 0: The video has no connection to the instruction whatsoever.
            - 1: The video shows the right object performing the wrong action, or the right action performed on the wrong object.
            - 2: The video follows the instruction and moves toward the intended goal.
            - 3: The video follows the instruction exactly and fully achieves the goal.

            Answer with 'Score: [score]'.
""",
            """\
Does this video carry out the instruction '{instruction}'? Judge it against this rubric:

            - 0: No relation to the instruction is shown.
            - 1: Either the object or the action is correct, but not both.
            - 2: The instruction is followed and progress toward the goal is visible.
            - 3: The instruction is followed exactly and the goal is fully achieved.

            Answer with 'Score: [score]'.
""",
            ],
    },
    "physical_laws": {
        "original": """\
Watch the video and determine if it shows any '{physical_laws}'
            Let's think step-by-step and conclude with "Yes" or "No".
""",
        "paraphrases": [
            """\
Watch the video closely and decide whether it displays any '{physical_laws}'
            Answer with "Yes" or "No".
""",
            """\
Does the video show any instance of '{physical_laws}'
            Answer with "Yes" or "No".
""",
        ],
    },
    "common_sense": {
        "original": """\
Does the video exhibit '{common_sense}'?
            Let's think step-by-step and conclude with "Yes" or "No".
""",
        "paraphrases": [
            """\
Does the video display '{common_sense}'?
            Answer with "Yes" or "No".
""",
            """\
Is '{common_sense}' present anywhere in the video?
            Answer with "Yes" or "No".
""",
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
            "Newton's Law violation: an object moves even though no external force acts on it.",
            "A Newton's Law violation, where objects gain motion despite no external force being applied.",
            ],
    },
    "physical_laws_1": {
        "original": 'Violation of the Law of Conservation of Mass or Solid Constitutive Law: Objects deform irregularly.',
        "paraphrases": [
            "Violation of Conservation of Mass or the Solid Constitutive Law: solid objects warp or deform in an irregular way.",
            "Violation of the Law of Conservation of Mass or the Solid Constitutive Law, seen as objects deforming irregularly.",
            ],
    },
    "physical_laws_2": {
        "original": 'Violation of Fluid Constitutive Law: Liquids flow in an unnatural manner.',
        "paraphrases": [
            "Fluid Constitutive Law violation: liquid moves in a way that looks unnatural.",
            "Violation of the Fluid Constitutive Law, where liquids flow in a manner that defies natural behavior.",
            ],
    },
    "physical_laws_3": {
        "original": 'Violation of Non-physical Penetration: Objects unnaturally pass through each other.',
        "paraphrases": [
            "Non-physical Penetration violation: solid objects pass through one another without colliding.",
            "Violation of Non-physical Penetration, where objects interpenetrate in a way that shouldn't be physically possible.",
            ],
    },
    "physical_laws_4": {
        "original": 'Violation of Gravity: Objects behave inconsistently with gravity.',
        "paraphrases": [
            "Gravity violation: objects act in ways that don't match how gravity should affect them.",
            "Violation of Gravity, where objects move or rest in ways gravity wouldn't allow.",
            ],
    },
    "common_sense_0": {
        "original": 'Poor Aesthetics: Visually unappealing or low-quality content.',
        "paraphrases": [
            "Poor Aesthetics: the visuals are unappealing or come across as low quality.",
            "Poor Aesthetics, meaning the footage is visually unappealing or poor in quality.",
            ],
    },
    "common_sense_1": {
        "original": 'Temporal Inconsistency: Noticeable flickering or abrupt changes.',
        "paraphrases": [
            "Temporal Inconsistency: visible flickering or sudden, jarring changes over time.",
            "Temporal Inconsistency, shown as noticeable flicker or abrupt shifts between frames.",
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
            'Does this video correspond to the description: "{caption}"? Rate it on a scale from 1 to 5, where 5 means a perfect match and 1 means it has no relevance at all.',
            'How well does this video reflect the description: "{caption}"? Give it a score from 1 to 5, where 5 is a perfect match and 1 is completely irrelevant.',
            ],
    },
    "PC": {
        "original": 'Does this video adhere to the physical laws? Rate the video on a scale from 1 to 5, where 5 means full compliance and 1 means significant violations.',
        "paraphrases": [
            "Does this video obey the laws of physics? Rate it on a scale from 1 to 5, where 5 means it fully complies and 1 means it violates them significantly.",
            "Is this video physically plausible? Score it from 1 to 5, where 5 indicates full adherence to physical laws and 1 indicates major violations.",
            ],
    },
}


# ===========================================================================
# phyjudge_9b -- 5 templates (NOT 16)
# ---------------------------------------------------------------------------
# From models/phyjudge-9B/subq+human.yaml. The 13 physical laws all share ONE
# template -- `physical_template` -- with {law} and {criteria} substituted per
# law, so there are five things to reword, not sixteen.
#
# These are the only prompts that do not live in judge_harness. infer.py
# renders them from that YAML, so a paraphrase here means swapping the YAML
# value before build_prompt runs, not editing a string in our code.
#
# THREE TRAPS, all of which fail silently:
#
#   * DOUBLED BRACES. `{{"SA": 3}}` is a .format() escape that renders as
#     {"SA": 3}. A single brace there is either a KeyError or a vanished
#     example. Keep them doubled.
#
#   * THE JSON KEY NAME IS THE PARSER'S CONTRACT. infer.parse_score(raw,
#     score_key) looks for exactly "SA" / "PTV" / "persistence" / the law
#     name. Reword the instruction around it however you like; do not rename
#     the key or translate it.
#
#   * {questions_block} IS THE SUB-QUESTION CHECKLIST, injected from
#     infer.GENERAL_SUB_QUESTIONS / infer.PHYSICAL_SUB_QUESTIONS. Dropping it
#     silently removes the checklists this model was fine-tuned with -- an
#     earlier version of the harness had exactly that bug.
#
# The 1-5 anchors are the rubric; rewording the descriptions is fair game,
# renumbering them is not.
# ===========================================================================

PHYJUDGE_PROMPTS = {
    "system_prompt": {
        "original": "You are a strict video evaluation model.",
        "paraphrases": [
            "You are a rigorous video evaluation model.",
            "You are a no-nonsense, strict evaluator of video content.",
        ],
    },
    "SA": {
        "original": """\
Evaluate Prompt Alignment (SA).

Caption:
"{prompt}"

The video was generated using a text+image-to-video (ti2v) model, conditioned on the first frame and the text prompt above.

Sub-questions to consider in your mind before scoring:
{questions_block}

Score 1-5:
5=fully aligned
4=mostly aligned with minor deviations
3=partially aligned with notable gaps
2=mostly misaligned
1=not aligned

Then output ONLY a JSON object with exactly one key: SA.

Example:
{{"SA": 3}}""",
        "paraphrases": [
            """\
Assess how well the video aligns with the prompt (SA).

Caption:
"{prompt}"

This video comes from a text+image-to-video (ti2v) model that was conditioned on the first frame together with the text prompt above.

Before scoring, work through these sub-questions in your mind:
{questions_block}

Score on a 1-5 scale:
5=fully aligned with the caption
4=mostly aligned, with small deviations
3=partially aligned, with clear gaps
2=largely misaligned
1=no alignment

Then output ONLY a JSON object with exactly one key: SA.

Example:
{{"SA": 3}}""",
            """\
Judge Prompt Alignment (SA) between the video and its caption.

Caption:
"{prompt}"

This is output from a text+image-to-video (ti2v) model conditioned on the first frame and the text prompt shown above.

Think through these sub-questions before scoring:
{questions_block}

Give a score of 1-5, where:
5=fully in line with the caption
4=mostly in line, small deviations
3=partly in line, noticeable gaps
2=mostly out of line
1=completely out of line

Then output ONLY a JSON object with exactly one key: SA.

Example:
{{"SA": 3}}""",
            ],
    },
    "PTV": {
        "original": """\
Evaluate Temporal Coherence (PTV).

Caption:
"{prompt}"

The video was generated using a text+image-to-video (ti2v) model, conditioned on the first frame and the text prompt above.

Sub-questions to consider in your mind before scoring:
{questions_block}

Score 1-5:
5=fully plausible event order
4=mostly plausible with minor timing issues
3=partially plausible
2=mostly implausible
1=completely implausible order

Then output ONLY a JSON object with exactly one key: PTV.

Example:
{{"PTV": 4}}""",
        "paraphrases": [
            """\
Assess Temporal Coherence (PTV) in this video.

Caption:
"{prompt}"

This clip comes from a text+image-to-video (ti2v) model, conditioned on the first frame and the text prompt above.

Before scoring, think through these sub-questions:
{questions_block}

Score on a 1-5 scale:
5=events unfold in a fully plausible order
4=mostly plausible, with minor timing issues
3=only partially plausible
2=mostly implausible
1=the event order is completely implausible

Then output ONLY a JSON object with exactly one key: PTV.

Example:
{{"PTV": 4}}""",
            """\
Judge how temporally coherent (PTV) this video is.

Caption:
"{prompt}"

This is the output of a text+image-to-video (ti2v) model, conditioned on the first frame and the text prompt shown above.

Work through these sub-questions in your mind before scoring:
{questions_block}

Give a score of 1-5, where:
5=the sequence of events is fully plausible
4=mostly plausible, minor timing problems
3=plausible in some respects only
2=mostly implausible
1=the sequence of events is entirely implausible

Then output ONLY a JSON object with exactly one key: PTV.

Example:
{{"PTV": 4}}""",
            ],
    },
    "persistence": {
        "original": """\
Evaluate Object Persistence.

Caption, for context only:
"{prompt}"

The video was generated using a text+image-to-video (ti2v) model, conditioned on the first frame and the text prompt above.

Sub-questions to consider in your mind before scoring:
{questions_block}

Score 1-5:
5=fully consistent
4=mostly consistent with minor flicker
3=noticeable issues
2=major inconsistencies
1=severe disappearance or identity changes

Then output ONLY a JSON object with exactly one key: persistence.

Example:
{{"persistence": 4}}""",
        "paraphrases": [
            """\
Assess Object Persistence in this video.

Caption, for context only:
"{prompt}"

This clip comes from a text+image-to-video (ti2v) model, conditioned on the first frame and the text prompt above.

Before scoring, think through these sub-questions:
{questions_block}

Score on a 1-5 scale:
5=objects stay fully consistent throughout
4=mostly consistent, with minor flicker
3=some noticeable issues
2=major inconsistencies appear
1=objects severely disappear or change identity

Then output ONLY a JSON object with exactly one key: persistence.

Example:
{{"persistence": 4}}""",
            """\
Judge the Object Persistence shown in this video.

Caption, for context only:
"{prompt}"

This is the output of a text+image-to-video (ti2v) model, conditioned on the first frame and the text prompt shown above.

Work through these sub-questions in your mind before scoring:
{questions_block}

Give a score of 1-5, where:
5=objects remain fully consistent
4=mostly consistent, with minor flickering
3=there are noticeable problems
2=inconsistencies are significant
1=objects vanish or change identity severely

Then output ONLY a JSON object with exactly one key: persistence.

Example:
{{"persistence": 4}}""",
            ],
    },
    # shared by all 13 laws: gravity, inertia, momentum, impenetrability,
    # collision, material, buoyancy, displacement, flow_dynamics,
    # boundary_interaction, fluid_continuity, reflection, shadow
    "physical_template": {
        "original": """\
Evaluate physical realism for one physical law: {law}.

Criterion:
{criteria}

Caption, for context only:
"{prompt}"

Sub-questions to consider in your mind before scoring:
{questions_block}

Judge the video itself. Do not penalize prompt mismatch unless it affects whether this physical law can be evaluated.

Score 1-5:
5=clearly correct
4=mostly correct with minor issues
3=partially correct or ambiguous
2=mostly incorrect
1=severely incorrect

Then output ONLY a JSON object with exactly one key: {law}.

Example:
{{"{law}": 3}}""",
        "paraphrases": [
            """\
Judge physical realism for a single physical law: {law}.

Criterion:
{criteria}

Caption, for context only:
"{prompt}"

Sub-questions to think through before you score:
{questions_block}

Base your judgment on the video itself. Only penalize a mismatch with the caption if it affects whether this physical law can be evaluated at all.

Score on a 1-5 scale:
5=clearly correct
4=mostly correct, minor issues only
3=partially correct or unclear
2=mostly incorrect
1=severely incorrect

Then output ONLY a JSON object with exactly one key: {law}.

Example:
{{"{law}": 3}}""",
            """\
Rate the video's physical realism for a single law: {law}.

Criterion:
{criteria}

Caption, included only for context:
"{prompt}"

Work through these sub-questions before scoring:
{questions_block}

Score what you see in the video. A mismatch with the caption should only lower the score if it stops you from judging this physical law.

Give a score of 1-5, where:
5=clearly accurate
4=mostly accurate, minor flaws
3=partly accurate or hard to tell
2=mostly inaccurate
1=badly inaccurate

Then output ONLY a JSON object with exactly one key: {law}.

Example:
{{"{law}": 3}}""",
            ],
    },
}


# ===========================================================================

_ALL = {
    "vila_templates": VILA_TEMPLATES,
    "vila_questions": VILA_QUESTIONS,
    "vp2_instructions": VP2_INSTRUCTIONS,
    "phyjudge_prompts": PHYJUDGE_PROMPTS,
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
    "phyjudge_prompts": {
        # {questions_block} injects the sub-question checklist; the JSON key
        # name is what infer.parse_score looks for; the doubled braces are a
        # .format() escape for the literal JSON example.
        "system_prompt": [],
        "SA": ["{prompt}", "{questions_block}", "SA", "1-5", '{{"SA"'],
        "PTV": ["{prompt}", "{questions_block}", "PTV", "1-5", '{{"PTV"'],
        "persistence": ["{prompt}", "{questions_block}", "persistence", "1-5",
                        '{{"persistence"'],
        "physical_template": ["{law}", "{criteria}", "{prompt}",
                              "{questions_block}", "1-5", '{{"{law}"'],
    },
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