"""
event_support_analysis.py
=========================
Read-only fault-event support analysis for the processed Brazil dataset.

PURPOSE
-------
Establish how many labelled fault EVENTS (not rows) actually exist, and
whether recording-group structure can be recovered from the data. This is
the evidence a split strategy must be chosen from — it does NOT choose one.

This script modifies nothing. It reads:
  data/fault_dataset.csv      (cleaned rows, carries source_event_id)
  outputs/source_event_register.csv     (pre-filter event boundaries)

and writes only:
  outputs/event_support_table.csv
  outputs/event_support_report.txt
  outputs/run_log.txt                   (one appended line)

RECORDING GROUPS ARE INFERRED, NOT GIVEN
----------------------------------------
The paper describes 16 recording days of roughly 07:30-17:00, but the
released MAT arrays contain no day-boundary marker and no time variable.
Group boundaries can only be inferred from breaks that the daylight filter
leaves in sample_index.

The threshold is NOT chosen to make the group count land on 16. Choosing a
threshold by its answer would manufacture the structure it claims to find.
Instead the gap distribution is probed across four orders of magnitude, and
a reconstruction is adopted only if it survives three independent tests:

  A. SEPARATION — a visible cliff in the gap distribution: an empty band
     between "within-session" gaps and "between-session" gaps.
  B. STABILITY  — a wide band of thresholds yields the SAME group count.
  C. PLAUSIBILITY — the resulting groups have physically sensible
     sample_index spans.

If any test fails the reconstruction is refused outright and sections 3b-3d
are skipped. A wrong day structure would silently corrupt every downstream
grouped split, so refusing is the safe failure.

Group SIZE imbalance is a soft flag, NOT an acceptance test. Retained row
counts are shaped by the irr_panel_wm2 >= 20 cleaning rule: an overcast day
loses far more rows around sunrise and sunset than a clear one, so unequal
retained sizes do not by themselves indicate wrong boundaries. When the flag
triggers, the low-count groups are diagnosed by span and retention density:
a NORMAL span with few rows points to weather, whereas an ABNORMAL span
alongside an abnormal row count points to a boundary problem.

The paper's statement of 16 recording days is reported as corroboration
after the fact. It is never used to select the threshold.
"""

import datetime
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_DIR  = SCRIPT_DIR.parent
DATA_CSV     = PROJECT_DIR / "data" / "fault_dataset.csv"
OUTPUTS_DIR  = PROJECT_DIR / "outputs"
EVENT_REGISTER = OUTPUTS_DIR / "source_event_register.csv"
TABLE_CSV    = OUTPUTS_DIR / "event_support_table.csv"
REPORT_TXT   = OUTPUTS_DIR / "event_support_report.txt"
RUN_LOG      = OUTPUTS_DIR / "run_log.txt"

LABEL_NAMES = {0: "Normal", 1: "Short-Circuit", 2: "Degradation",
               3: "Open-Circuit", 4: "Shadowing"}
CONTROLLED = [1, 2, 3]          # deliberately induced fault classes
MANDATED_THRESHOLDS = [1800, 3600, 7200, 14400, 21600, 28800, 43200, 57600]

# Acceptance criteria for the group reconstruction (fixed in advance).
SEPARATION_RATIO_MIN = 5.0   # smallest between-session gap / largest within
STABILITY_BAND_MIN   = 4.0   # widest threshold band agreeing on group count
MAX_SPAN_H           = 24.0  # a recording session cannot exceed a day
MIN_SPAN_H           = 0.5
# Soft flag only — never rejects a grouping. See module docstring.
SIZE_RATIO_FLAG      = 4.0   # largest/smallest retained group row count
PAPER_RECORDING_DAYS = 16    # reported as corroboration, never as a target

_report = []


def emit(line=""):
    print(line)
    _report.append(line)


emit("=" * 70)
emit("  FAULT-EVENT SUPPORT ANALYSIS (read-only)")
emit("=" * 70)

for p in (DATA_CSV, EVENT_REGISTER):
    if not p.exists():
        raise SystemExit(f"Missing input: {p}\nRun fault_preprocessing.py first.")

df = pd.read_csv(DATA_CSV).sort_values("sample_index").reset_index(drop=True)
register = pd.read_csv(EVENT_REGISTER)
emit(f"\nCleaned rows        : {len(df):,}")
emit(f"Source events (raw) : {len(register):,}")
emit(f"sample_index range  : {df['sample_index'].min():,} .. "
     f"{df['sample_index'].max():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 3a — RECORDING GROUP DETECTION (evidence-driven)
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 70)
emit("  3a. RECORDING GROUP DETECTION")
emit("=" * 70)

gaps = df["sample_index"].diff()
gaps_valid = gaps.dropna()

emit(f"\nGap statistics (consecutive sample_index differences):")
emit(f"  contiguous steps (gap == 1): {int((gaps_valid == 1).sum()):,}")
emit(f"  discontinuities (gap > 1)  : {int((gaps_valid > 1).sum()):,}")

top50 = gaps_valid.sort_values(ascending=False).head(50)
emit(f"\nLargest 50 gaps (seconds at 1 Hz nominal):")
for rank, (idx, g) in enumerate(top50.items(), 1):
    at = df.loc[idx, "sample_index"]
    emit(f"  {rank:>3}. {int(g):>10,} s  ({g/3600:>7.2f} h)  before sample_index {int(at):,}")

emit(f"\nGroup count by threshold (mandated probe set):")
for threshold in MANDATED_THRESHOLDS:
    n_groups = int((gaps_valid > threshold).sum()) + 1
    emit(f"  gap > {threshold:6,}s → {n_groups} groups")

# Fine log-spaced probe to locate any stable plateau.
probe = np.unique(np.logspace(np.log10(120), np.log10(90000), 80).astype(int))
probe_counts = np.array([int((gaps_valid > t).sum()) + 1 for t in probe])

plateaus = {}
start = 0
for i in range(1, len(probe) + 1):
    if i == len(probe) or probe_counts[i] != probe_counts[start]:
        count = int(probe_counts[start])
        lo, hi = int(probe[start]), int(probe[i - 1])
        band = hi / lo if lo > 0 else 1.0
        if band > plateaus.get(count, (0, 0, 0))[2]:
            plateaus[count] = (lo, hi, band)
        start = i

best_count, (band_lo, band_hi, band_width) = max(
    plateaus.items(), key=lambda kv: kv[1][2])

emit(f"\nWidest stable plateau across probed thresholds:")
emit(f"  group count {best_count} holds for thresholds "
     f"{band_lo:,}s .. {band_hi:,}s  (band width x{band_width:.1f})")

# Condition A — separation across that empty band.
above = gaps_valid[gaps_valid > band_hi]
below = gaps_valid[gaps_valid <= band_lo]
if len(above) and len(below):
    sep_ratio = float(above.min()) / float(below.max())
else:
    sep_ratio = float("inf") if len(below) == 0 else 0.0
cond_a = sep_ratio >= SEPARATION_RATIO_MIN
emit(f"\n  A. SEPARATION")
emit(f"     smallest gap above band : "
     f"{int(above.min()) if len(above) else 'n/a':>10} s")
emit(f"     largest gap below band  : "
     f"{int(below.max()) if len(below) else 'n/a':>10} s")
emit(f"     separation ratio        : {sep_ratio:.2f} "
     f"(need >= {SEPARATION_RATIO_MIN})  → {'PASS' if cond_a else 'FAIL'}")

# Condition B — stability of the plateau.
cond_b = band_width >= STABILITY_BAND_MIN
emit(f"  B. STABILITY")
emit(f"     threshold band width    : x{band_width:.1f} "
     f"(need >= x{STABILITY_BAND_MIN})  → {'PASS' if cond_b else 'FAIL'}")

# Adopt the geometric centre of the empty band — not tuned to any target.
adopted_threshold = int(np.sqrt(band_lo * band_hi))
group_id = (gaps > adopted_threshold).fillna(False).cumsum()
df["recording_group"] = group_id.astype(int)

grp = df.groupby("recording_group")["sample_index"]
spans_h = ((grp.max() - grp.min()) / 3600.0)
sizes = df.groupby("recording_group").size()

cond_c = (bool((spans_h <= MAX_SPAN_H).all())
          and bool((spans_h >= MIN_SPAN_H).all()))
emit(f"  C. PLAUSIBILITY  (at adopted threshold {adopted_threshold:,}s)")
emit(f"     groups formed           : {len(sizes)}")
emit(f"     span range              : {spans_h.min():.2f} h .. {spans_h.max():.2f} h "
     f"(need {MIN_SPAN_H}-{MAX_SPAN_H} h)")
emit(f"     → {'PASS' if cond_c else 'FAIL'}")

accepted = cond_a and cond_b and cond_c

# ── Soft flag: retained group-size imbalance (never rejects) ────────────────
size_imbalance = float(sizes.max() / sizes.min()) if sizes.min() > 0 else float("inf")
median_span = float(spans_h.median())
emit(f"\n  SOFT FLAG — retained group-size imbalance (not an acceptance test)")
emit(f"     rows per group          : {int(sizes.min()):,} .. {int(sizes.max()):,} "
     f"(ratio x{size_imbalance:.1f}, flag at > x{SIZE_RATIO_FLAG})")
if size_imbalance > SIZE_RATIO_FLAG:
    emit(f"     STATUS: FLAGGED FOR REVIEW — grouping NOT rejected.")
    emit(f"     Retained counts are shaped by the irr >= 20 W/m^2 filter, so an")
    emit(f"     overcast session legitimately keeps fewer rows. Diagnosing the")
    emit(f"     low-count groups by span and retention density:")
    cutoff = sizes.max() / SIZE_RATIO_FLAG
    low_groups = sizes[sizes < cutoff].index.tolist()
    emit(f"     median span across groups: {median_span:.2f} h")
    emit(f"     {'Grp':>5} {'Rows':>9} {'Span (h)':>9} {'vs median':>10} "
         f"{'rows/h':>8}  diagnosis")
    for g in low_groups:
        sp = float(spans_h.loc[g])
        ratio = sp / median_span if median_span > 0 else float("nan")
        density = sizes.loc[g] / sp if sp > 0 else float("nan")
        if 0.6 <= ratio <= 1.4:
            diag = "NORMAL span, low retention → weather-consistent"
        else:
            diag = "ABNORMAL span → possible boundary problem"
        emit(f"     {g:>5} {int(sizes.loc[g]):>9,} {sp:>9.2f} {ratio:>9.2f}x "
             f"{density:>8.0f}  {diag}")
    if not low_groups:
        emit(f"     (no group falls below max/{SIZE_RATIO_FLAG:.0f}; "
             f"imbalance driven by spread rather than outliers)")
else:
    emit(f"     STATUS: normal (<= x{SIZE_RATIO_FLAG})")

# ── Corroboration only — never used to choose the threshold ─────────────────
emit(f"\n  CORROBORATION (reported after the fact, not a tuning target)")
emit(f"     groups recovered        : {len(sizes)}")
emit(f"     paper's recording days  : {PAPER_RECORDING_DAYS}")
if len(sizes) == PAPER_RECORDING_DAYS:
    emit(f"     → matches the paper's stated 16 recording days")
else:
    emit(f"     → differs from the paper by {len(sizes) - PAPER_RECORDING_DAYS:+d}; "
         f"the threshold was NOT adjusted to close this gap")

if not accepted:
    emit("")
    emit("RECORDING GROUP BOUNDARIES NOT RELIABLY RECOVERABLE")
    emit("")
    emit("  The gap evidence above does not support a defensible day/session")
    emit("  reconstruction. No boundaries have been invented, and sections")
    emit("  3b-3d are skipped. Review the gap distribution to decide whether")
    emit("  an external source of recording structure is required.")
    REPORT_TXT.write_text("\n".join(_report) + "\n")
    with open(RUN_LOG, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
                f"event_support_analysis.py | input: {DATA_CSV.name} {len(df):,} rows | "
                f"output: {REPORT_TXT.name} (no table) | "
                f"notes: groups NOT recoverable "
                f"(A={cond_a}, B={cond_b}, C={cond_c}); 3b-3d skipped\n")
    emit(f"\nSaved: {REPORT_TXT}")
    raise SystemExit(0)

emit(f"\n  ALL THREE TESTS PASS — reconstruction adopted")
emit(f"  Adopted threshold: {adopted_threshold:,}s "
     f"(geometric centre of the empty band {band_lo:,}-{band_hi:,}s)")
emit(f"  Recording groups : {len(sizes)}")
emit(f"\n  Per-group detail:")
emit(f"    {'Group':>5} {'Rows':>10} {'min idx':>12} {'max idx':>12} {'Span (h)':>10}")
for g in sorted(df["recording_group"].unique()):
    sub = df[df["recording_group"] == g]["sample_index"]
    emit(f"    {g:>5} {len(sub):>10,} {int(sub.min()):>12,} {int(sub.max()):>12,} "
         f"{(sub.max()-sub.min())/3600.0:>10.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3b — EVENT COUNTS FROM source_event_id
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 70)
emit("  3b. EVENT COUNTS (from pre-filter source_event_id)")
emit("=" * 70)

retained = (df.groupby("source_event_id")
              .agg(retained_samples=("sample_index", "size"),
                   f_nv=("f_nv", "first"))
              .reset_index())
ev = register.merge(retained[["source_event_id", "retained_samples"]],
                    on="source_event_id", how="left")
ev["retained_samples"] = ev["retained_samples"].fillna(0).astype(int)
ev["survived"] = ev["retained_samples"] > 0

emit("\nWhole dataset — distinct source events per class:")
emit(f"  {'Class':<18} {'raw events':>11} {'retained':>10} {'raw samples':>13} {'retained samples':>17}")
for lbl in sorted(LABEL_NAMES):
    sub = ev[ev["f_nv"] == lbl]
    if not len(sub):
        continue
    emit(f"  {lbl} {LABEL_NAMES[lbl]:<15} {len(sub):>11,} "
         f"{int(sub['survived'].sum()):>10,} "
         f"{int(sub['raw_sample_count'].sum()):>13,} "
         f"{int(sub['retained_samples'].sum()):>17,}")

emit("\nControlled-fault events — raw vs retained sample counts:")
for lbl in CONTROLLED:
    sub = ev[ev["f_nv"] == lbl].sort_values("source_event_id")
    emit(f"\n  {lbl} — {LABEL_NAMES[lbl]} ({len(sub)} source events):")
    emit(f"    {'event_id':>9} {'start idx':>12} {'end idx':>12} {'raw':>8} {'retained':>9}")
    for _, r in sub.iterrows():
        emit(f"    {int(r['source_event_id']):>9} {int(r['start_sample_index']):>12,} "
             f"{int(r['end_sample_index']):>12,} {int(r['raw_sample_count']):>8,} "
             f"{int(r['retained_samples']):>9,}")

emit("\nShadowing is reported as observed. Its runs are not part of the")
emit("controlled schedule and fragmentation is expected; no attempt is made")
emit("to merge runs or reconcile them against the fault schedule.")


# ─────────────────────────────────────────────────────────────────────────────
# 3c — SUPPORT TABLE
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 70)
emit("  3c. SUPPORT TABLE")
emit("=" * 70)

rows = []
for g in sorted(df["recording_group"].unique()):
    sub = df[df["recording_group"] == g]
    si = sub["sample_index"]
    row = {
        "group": int(g),
        "retained_rows": int(len(sub)),
        "span_h": round(float((si.max() - si.min()) / 3600.0), 2),
    }
    for lbl in sorted(LABEL_NAMES):
        s = sub[sub["f_nv"] == lbl]
        row[f"{LABEL_NAMES[lbl].lower().replace('-', '_')}_events"] = int(
            s["source_event_id"].nunique())
        row[f"{LABEL_NAMES[lbl].lower().replace('-', '_')}_samples"] = int(len(s))
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(TABLE_CSV, index=False)

emit(f"\n  {'Grp':>3} {'Rows':>9} {'Span':>7} | {'Nrm':>4} {'SC':>4} {'Deg':>4} "
     f"{'OC':>4} {'Shd':>5} | {'SC smp':>8} {'Deg smp':>8} {'OC smp':>8} {'Shd smp':>9}")
for _, r in table.iterrows():
    emit(f"  {int(r['group']):>3} {int(r['retained_rows']):>9,} {r['span_h']:>7.2f} | "
         f"{int(r['normal_events']):>4} {int(r['short_circuit_events']):>4} "
         f"{int(r['degradation_events']):>4} {int(r['open_circuit_events']):>4} "
         f"{int(r['shadowing_events']):>5} | "
         f"{int(r['short_circuit_samples']):>8,} {int(r['degradation_samples']):>8,} "
         f"{int(r['open_circuit_samples']):>8,} {int(r['shadowing_samples']):>9,}")

n_groups = len(table)
totals = {lbl: int(ev[(ev["f_nv"] == lbl) & ev["survived"]].shape[0])
          for lbl in sorted(LABEL_NAMES)}

emit("\nTotal distinct source events per class (whole dataset, retained):")
for lbl in sorted(LABEL_NAMES):
    emit(f"  {LABEL_NAMES[lbl]:<14}: {totals.get(lbl, 0)}")

emit("\nPaper's scheduled controlled events (Table 2): 2 per class per day "
     "x 16 days = 32 each")
emit(f"Observed vs scheduled: SC {totals.get(1,0)}/32, "
     f"Deg {totals.get(2,0)}/32, OC {totals.get(3,0)}/32")
emit("NOTE: shortfall is expected — reported sample counts indicate not every")
emit("scheduled event survived into the released data.")

cov_cols = {1: "short_circuit_events", 2: "degradation_events",
            3: "open_circuit_events"}
full_cov = table[(table[cov_cols[1]] > 0) & (table[cov_cols[2]] > 0)
                 & (table[cov_cols[3]] > 0)]
emit(f"\nGroups containing at least one event of each controlled fault class: "
     f"{len(full_cov)} of {n_groups}")
for lbl, col in cov_cols.items():
    zero = table[table[col] == 0]["group"].tolist()
    emit(f"Groups with zero {LABEL_NAMES[lbl].lower()} events : {zero}")


# ─────────────────────────────────────────────────────────────────────────────
# 3d — SPLIT FEASIBILITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 70)
emit("  SPLIT FEASIBILITY ASSESSMENT")
emit("=" * 70)

min_class_events = min(totals.get(l, 0) for l in CONTROLLED)
emit("\n  For meaningful per-class recall, a test partition needs at least")
emit("  2 source events of each controlled fault class.")
emit("")
emit(f"  Groups with full controlled-fault coverage: {len(full_cov)} of {n_groups}")
emit(f"  Minimum events available in any single class: {min_class_events}")
emit("")
if len(full_cov) >= 3 and min_class_events >= 6:
    emit('  "FIXED SPLIT VIABLE — sufficient groups have full class coverage to')
    emit('   construct train/validation/test partitions each containing all classes"')
elif len(full_cov) >= 1 and min_class_events >= 4:
    emit('  "FIXED SPLIT MARGINAL — coverage is uneven; grouped cross-validation')
    emit('   over recording groups is likely more appropriate"')
else:
    emit('  "FIXED SPLIT NOT VIABLE — too few events per class for a held-out test')
    emit('   partition; grouped cross-validation is required, with per-class')
    emit('   variance reported across folds"')

emit("")
emit("  This assessment states what the data supports. No folds have been")
emit("  selected and no split has been applied.")

REPORT_TXT.write_text("\n".join(_report) + "\n")
print(f"\nSaved: {TABLE_CSV}")
print(f"Saved: {REPORT_TXT}")

with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"event_support_analysis.py | "
            f"input: {DATA_CSV.name} {len(df):,} rows, {EVENT_REGISTER.name} "
            f"{len(register):,} events | "
            f"output: {TABLE_CSV.name} {len(table)} groups, {REPORT_TXT.name} | "
            f"notes: threshold={adopted_threshold}s, groups={n_groups}, "
            f"full-coverage groups={len(full_cov)}, min class events={min_class_events}; "
            f"read-only, no split chosen\n")
print(f"Run log appended: {RUN_LOG}")
