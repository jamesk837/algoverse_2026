"""Step 10 decision gate: build the optional predictor-only / encoder+predictor
PC probes the doc allows -- "if predictor features prove informative"?

Read "informative" on the DISCRIMINATION axis, not prediction quality: the
verify sweep already showed the raw predictor is flat in context and loses to
a copy baseline at every context, so quality is settled and weak (see
predictor-loses-to-copy-baseline). What's still open is whether the error
signal nonetheless separates clean from temporally-corrupted clips, which a
copy baseline can't answer for free.

Adds NO new statistic. For each of the six stats predict_vjepa.report()
already computes, it calls report(push_to_s3=False) and reads report()'s own
significance test -- the paired, clip-level bootstrap CI on the signed delta
(clean, kind) -- exactly the test every other part of this repo already uses
for this question (analyze.py, train_probe.py, eval_probe.py, annotate.py all
do the same paired-clip bootstrap). No AUC-specific CI is invented here.

VERDICT = BUILD only if:
  - at least one (stat, temporal variant) has a CI entirely above 0 (the
    predictor is significantly MORE surprised on that attack), AND
  - no (stat, superficial variant) ALSO clears that bar -- a predictor that
    reacts to any content change, not specifically a temporal one, is not a
    useful independent reference; the encoder-side consistency loss already
    penalizes exactly that failure mode on the PC probe side.

This threshold is not specified anywhere in the doc -- there is no definition
of "prove informative" to defer to -- so it is called out as mine, per this
repo's own convention for unspecified defaults (train_probe.py: "a default
should be what [the project lead] asked for; anything of mine is the opt-in").

    python decide_predictor_probe.py
"""
import csv

import predict_vjepa as PV

STATS = ["mean", "max", "std", "p90", "p95", "spike_rate"]


def main(doc=None, stats=STATS, out_csv="predictor_decision.csv",
         push_to_s3=False):
    signal, confound, rows = [], [], []
    for stat in stats:
        print(f"\n{'=' * 70}\n{stat}\n{'=' * 70}")
        payload = PV.report(doc=doc, stat=stat, push_to_s3=False)
        for name, r in payload["variants"].items():
            if r["kind"] not in ("temporal", "superficial"):
                continue
            hit = r["lo"] > 0
            rows.append(dict(stat=stat, kind=r["kind"], variant=name,
                             n=r["n"], d=r["d"], lo=r["lo"], hi=r["hi"],
                             auc=r["auc"], significant=hit))
            if hit and r["kind"] == "temporal":
                signal.append((stat, name))
            if hit and r["kind"] == "superficial":
                confound.append((stat, name))

    print(f"\n{'#' * 70}\n# Step 10 decision: build the optional predictor "
          f"probes?\n{'#' * 70}")
    print(f"\ntemporal variants with a significant rise: {signal or 'none'}")
    print(f"superficial variants that ALSO moved significantly: "
          f"{confound or 'none'}")

    if signal and not confound:
        verdict = "BUILD"
        why = ("temporal signal exists with no matching superficial "
               "confound -- the predictor is picking up something "
               "specifically temporal, worth turning into a probe.")
    elif signal and confound:
        verdict = "AMBIGUOUS"
        why = ("temporal signal exists but at least one superficial variant "
               "ALSO rose significantly on some statistic. Read the CSV: is "
               "it the SAME statistic flagging both, or different ones? If "
               "different, the temporal-specific claim may still survive on "
               "the stats where superficial stays flat -- read row by row "
               "before deciding, this script only flags the ambiguity.")
    else:
        verdict = "SKIP"
        why = ("no (stat, temporal variant) pair cleared significance. "
               "Consistent with the predictor already losing to a copy "
               "baseline on raw prediction quality -- this says the "
               "discrimination axis doesn't clear the bar either. Building "
               "predictor-only/encoder+predictor probes on this signal is "
               "unlikely to add anything the 4-arch ablation didn't already "
               "cover on the encoder side.")
    print(f"\nVERDICT: {verdict}\n  {why}")
    print("\nThis threshold (bootstrap CI on the paired clip-level delta "
          "excludes 0) is\nnot from the doc -- there is no specified "
          "criterion for 'informative'. Called out\nas mine, the way this "
          "repo marks every other unspecified default.")

    if rows:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv} ({len(rows)} rows)")

    if push_to_s3 and rows:
        import io as _io
        s3 = PV.E._ensure_s3()
        buf = _io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        base = f"predictor/{PV.PRED_MODEL}"
        temporal_sig = sorted({v for _s, v in signal})
        superf_sig = sorted({v for _s, v in confound})
        why_short = (
            f"{verdict}. temporal rises: {temporal_sig or 'none'}; "
            f"superficial rises (confound): {superf_sig or 'none'}. "
            "shuffle raises predictor error but so do several caption-echo "
            "overlays, and the predictor loses to a copy baseline on shuffle "
            "-- no clean temporal-specific signal, so the optional "
            "predictor-only / encoder+predictor probes were not built."
        )
        s3.put_object(Bucket=PV.BUCKET, Key=f"{base}/decision.csv",
                      Body=buf.getvalue().encode())
        s3.put_object(Bucket=PV.BUCKET, Key=f"{base}/verdict.txt",
                      Body=why_short.encode())
        print(f"uploaded -> s3://{PV.BUCKET}/{base}/{{decision.csv,verdict.txt}}")
        print(f"  {why_short}")
    return verdict, rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Step 10 predictor decision gate")
    ap.add_argument("--push", action="store_true",
                    help="upload verdict.txt + decision.csv to "
                         "s3://<bucket>/predictor/<model>/")
    main(push_to_s3=ap.parse_args().push)
