"""
event_size_diagnostic.py
========================
Objective 1, Step 4 — event-size diagnostic for the controlled fault classes.

READ-ONLY. Reads the processed dataset and the source event register and
describes how large each labelled fault event actually is. It writes only
its own report; it does not touch the dataset.

WHY THIS EXISTS
---------------
The paper induced each controlled fault for 10 minutes, which at 1 Hz is
~600 samples. The released labels contain many far shorter runs. Counting
raw label runs as "events" therefore overstates how much independent fault
evidence exists, which in turn overstates how well a grouped split can be
supported. This diagnostic measures the real size structure.

THE SIZE THRESHOLD IS DESCRIPTIVE METADATA ONLY
------------------------------------------------
Nothing here removes, relabels, merges, or excludes any row. The dataset
authors' f_nv labels are authoritative and every labelled row stays in the
classification data. Short runs are *reported* as probable label-edge
flicker so that event counts can be interpreted honestly — they are not
filtered out, and this script writes no dataset of any kind.

INPUT : data/fault_dataset.csv
        outputs/source_event_register.csv
OUTPUT: outputs/event_size_diagnostic.txt
        outputs/run_log.txt (one appended line)
"""

import datetime
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR     = Path(__file__).resolve().parent
PROJECT_DIR    = SCRIPT_DIR.parent
DATA_CSV       = PROJECT_DIR / "data" / "fault_dataset.csv"
OUTPUTS_DIR    = PROJECT_DIR / "outputs"
EVENT_REGISTER = OUTPUTS_DIR / "source_event_register.csv"
REPORT_TXT     = OUTPUTS_DIR / "event_size_diagnostic.txt"
RUN_LOG        = OUTPUTS_DIR / "run_log.txt"

# Same threshold Step 5a will use; sits inside the previously established
# stable band (387-42,335 s) where every threshold recovers 16 groups.
GAP_THRESHOLD = 3600

CONTROLLED = [1, 2, 3]
LABEL_NAMES = {0: "Normal", 1: "Short-Circuit", 2: "Degradation",
               3: "Open-Circuit", 4: "Shadowing"}
NOMINAL_EVENT_SAMPLES = 600   # paper: 10 minutes at 1 Hz

_report = []


def emit(line=""):
    print(line)
    _report.append(line)


emit("=" * 78)
emit("  EVENT-SIZE DIAGNOSTIC — controlled fault classes (read-only)")
emit("=" * 78)

for p in (DATA_CSV, EVENT_REGISTER):
    if not p.exists():
        raise SystemExit(f"Missing input: {p}")

df = pd.read_csv(DATA_CSV, usecols=["sample_index", "source_event_id", "f_nv"])
df = df.sort_values("sample_index").reset_index(drop=True)
register = pd.read_csv(EVENT_REGISTER)

emit(f"\nCleaned rows        : {len(df):,}")
emit(f"Source events (raw) : {len(register):,}")

# Recording groups, recomputed from sample_index gaps.
df["recording_group"] = (df["sample_index"].diff() > GAP_THRESHOLD).fillna(False).cumsum().astype(int)
n_groups = df["recording_group"].nunique()
emit(f"Recording groups    : {n_groups} (gap threshold {GAP_THRESHOLD:,} samples)")

# Retained size and group membership per event.
agg = (df.groupby("source_event_id")
         .agg(retained=("sample_index", "size"),
              group_first=("recording_group", "first"),
              group_last=("recording_group", "last"))
         .reset_index())
ev = register.merge(agg, on="source_event_id", how="left")
ev["retained"] = ev["retained"].fillna(0).astype(int)
spanning = ev[(ev["retained"] > 0) & (ev["group_first"] != ev["group_last"])]
if len(spanning):
    emit(f"\nWARNING: {len(spanning)} event(s) span a recording-group boundary.")
else:
    emit("No event spans a recording-group boundary.")


# ─────────────────────────────────────────────────────────────────────────────
# PER-CLASS EVENT LISTINGS
# ─────────────────────────────────────────────────────────────────────────────
for lbl in CONTROLLED:
    sub = ev[ev["f_nv"] == lbl].sort_values("retained", ascending=False)
    emit("\n" + "=" * 78)
    emit(f"Class {lbl} ({LABEL_NAMES[lbl]}) — {len(sub)} source events")
    emit("=" * 78)
    emit(f"  {'event_id':>9} {'raw_samples':>12} {'retained':>9} "
         f"{'recording_group':>16} {'start_idx':>12} {'end_idx':>12}")
    for _, r in sub.iterrows():
        grp = "-" if pd.isna(r["group_first"]) else str(int(r["group_first"]))
        emit(f"  {int(r['source_event_id']):>9} {int(r['raw_sample_count']):>12,} "
             f"{int(r['retained']):>9,} {grp:>16} "
             f"{int(r['start_sample_index']):>12,} {int(r['end_sample_index']):>12,}")


# ─────────────────────────────────────────────────────────────────────────────
# SIZE DISTRIBUTION AND NATURAL BREAKS
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  SAMPLE-COUNT DISTRIBUTION")
emit("=" * 78)

all_ctrl = ev[ev["f_nv"].isin(CONTROLLED)].copy()
sizes_sorted = np.sort(all_ctrl["retained"].to_numpy())
emit(f"\nAll {len(sizes_sorted)} controlled-fault events, retained sizes ascending:")
emit("  " + ", ".join(f"{int(s):,}" for s in sizes_sorted))

emit(f"\n  min {int(sizes_sorted.min()):,} | median {int(np.median(sizes_sorted)):,} | "
     f"max {int(sizes_sorted.max()):,} | mean {sizes_sorted.mean():.1f}")

# Natural breaks = largest multiplicative jumps between consecutive sizes.
emit("\nLargest multiplicative jumps between consecutive event sizes:")
jumps = []
for i in range(len(sizes_sorted) - 1):
    lo, hi = sizes_sorted[i], sizes_sorted[i + 1]
    if lo > 0 and hi > lo:
        jumps.append((hi / lo, int(lo), int(hi)))
jumps.sort(reverse=True)
for ratio, lo, hi in jumps[:6]:
    emit(f"  {lo:>6,} → {hi:>6,}   x{ratio:.1f}")

if jumps and jumps[0][0] >= 5:
    br_lo, br_hi = jumps[0][1], jumps[0][2]
    break_desc = (f"clear break between {br_lo:,} and {br_hi:,} samples "
                  f"(x{jumps[0][0]:.1f}) — separates fragment runs from "
                  f"full-length inductions")
else:
    br_lo = br_hi = None
    break_desc = "none apparent"


# ─────────────────────────────────────────────────────────────────────────────
# ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  EVENT SIZE STRUCTURE")
emit("=" * 78)
emit(f"  Paper's controlled inductions: 10 minutes = ~{NOMINAL_EVENT_SAMPLES} "
     f"samples at 1 Hz")

for lbl in CONTROLLED:
    sub = ev[ev["f_nv"] == lbl]
    big = sub[sub["retained"] >= 300]
    mid = sub[(sub["retained"] > 10) & (sub["retained"] < 300)]
    small = sub[sub["retained"] <= 10]
    emit("")
    emit(f"  Class {lbl} ({LABEL_NAMES[lbl]}):")
    emit(f"    Events >=300 samples : {len(big):<3} (mean size "
         f"{big['retained'].mean():.0f})" if len(big) else
         f"    Events >=300 samples : 0")
    if len(mid):
        detail = ", ".join(
            f"{int(r['retained'])} samples (grp "
            f"{'-' if pd.isna(r['group_first']) else int(r['group_first'])})"
            for _, r in mid.sort_values("retained", ascending=False).iterrows())
        emit(f"    Events 11-299        : {len(mid):<3} → {detail}")
    else:
        emit(f"    Events 11-299        : 0")
    emit(f"    Events <=10          : {len(small)}")

emit("")
emit(f"  Natural break in the distribution: {break_desc}")

emit("")
emit("  INTERPRETATION")
emit("    Runs of <=10 samples are almost certainly label-edge flicker: they")
emit("    cluster immediately adjacent to full-length events, where the label")
emit("    bounces between classes for a few seconds at a transition.")
emit("    Counting them as independent events inflates apparent fault support.")
emit("")
emit("  THIS THRESHOLD IS DESCRIPTIVE ONLY")
emit("    No row is removed, relabelled, or excluded on the basis of event")
emit("    size. Every labelled row remains in the classification data, and")
emit("    this script writes no dataset. The counts above are support")
emit("    metadata for judging how much independent fault evidence exists.")

# Substantial-event support per recording group — what a fold split rests on.
emit("")
emit("  SUBSTANTIAL EVENTS (>=300 retained samples) PER RECORDING GROUP")
big_all = all_ctrl[all_ctrl["retained"] >= 300]
piv = (big_all.groupby(["group_first", "f_nv"])["source_event_id"]
       .nunique().unstack(fill_value=0))
piv.columns = [LABEL_NAMES[c] for c in piv.columns]
for c in [LABEL_NAMES[l] for l in CONTROLLED]:
    if c not in piv.columns:
        piv[c] = 0
piv = piv[[LABEL_NAMES[l] for l in CONTROLLED]]
emit(f"    {'group':>6} {'Short-Circuit':>14} {'Degradation':>12} {'Open-Circuit':>13}")
for g in sorted(piv.index):
    r = piv.loc[g]
    emit(f"    {int(g):>6} {int(r['Short-Circuit']):>14} "
         f"{int(r['Degradation']):>12} {int(r['Open-Circuit']):>13}")
covered = [int(g) for g in piv.index if (piv.loc[g] > 0).all()]
emit(f"    Groups with all three controlled classes: {covered}")
absent = [g for g in range(n_groups) if g not in piv.index]
emit(f"    Groups with no substantial controlled event: {absent}")

REPORT_TXT.write_text("\n".join(_report) + "\n")
print(f"\nSaved: {REPORT_TXT}")

with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"event_size_diagnostic.py | "
            f"input: {DATA_CSV.name} {len(df):,} rows, {EVENT_REGISTER.name} "
            f"{len(register):,} events | output: {REPORT_TXT.name} | "
            f"notes: read-only, descriptive threshold only, no rows removed "
            f"or relabelled; controlled events {len(all_ctrl)}, "
            f">=300 samples {len(big_all)}\n")
print(f"Run log appended: {RUN_LOG}")
