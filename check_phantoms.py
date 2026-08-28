"""Settles one question: are the ~120 test-corpus download failures real data
loss, or doc-level entries that were never renderable under any name and were
already excluded from everything scored so far.

Read-only LISTs, a handful of small GETs. No torch, no downloads."""
import embed_vjepa as E

doc = E.load_doc("videophy2_test")
doc_stems = set(doc["clips"])
print(f"{len(doc_stems)} stems in the videophy2_test doc")

rendered = set()
for k in E.list_keys("attacks/test/"):
    parts = k.split("/")
    if len(parts) >= 3:
        rendered.add(parts[2])
print(f"{len(rendered)} distinct rendered attacks/test/<stem>/ directories")

no_render = sorted(doc_stems - rendered)
print(f"{len(no_render)} doc stems have NO rendered directory under any name")

judge_keys = E.list_keys("results/pass1/videophy2_auto/test/")
judge_stems = {k.split("/")[-1].removesuffix(".json") for k in judge_keys}
print(f"{len(judge_stems)} clips actually scored by videophy2_auto on test")

overlap = set(no_render) & judge_stems
print(f"{len(overlap)} of the no-render stems were ever scored by a judge "
      f"(this MUST be 0 for 'no data lost' to hold)")

print(f"\nreal, renderable, scoreable test corpus: "
      f"{len(rendered & judge_stems)} clips")
print(f"\nfirst 10 no-render stems:")
for s in no_render[:10]:
    print(" ", s)
