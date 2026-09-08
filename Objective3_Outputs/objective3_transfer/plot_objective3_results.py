"""Objective 3 — Consolidation: five figures and a written summary.

Reads frozen Objective 3 outputs only. No modelling, no retraining, no
transfer, no new experiments. Consolidation is self-contained within
Objective 3: no Objective 1 or Objective 2 output, and no dataset, is read.

Figure 5's Arm B / Arm C comparison is built solely from the frozen
objective3_objective2_agreement.csv, which is an Objective 3 artefact.
"""
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ═════════════════════════════════════════════════════════════════════════════
# GUARDS
# ═════════════════════════════════════════════════════════════════════════════
# The prompt's explicit banned-phrase list, plus the terminology table's
# "Never use" entries that do not conflict with it.
#
# NOTE: a bare "hk fault" substring is deliberately NOT banned. The prompt
# mandates two verbatim strings containing it - the Figure 5 annotation
# ("neither has verified HK fault labels") and the required conclusion
# ("HK fault-type predictions are diagnostic outputs only"). The specific
# banned forms "hk fault classification" and "hk fault type" are retained and
# do not match either, since the conclusion hyphenates "fault-type".
#
# FRAGILE, READ BEFORE EDITING: the "hk fault type" ban survives ONLY because
# the mandated conclusion hyphenates the phrase as "HK fault-type". Rewriting
# that sentence to say "HK fault type predictions" without the hyphen would
# trip this guard on the required conclusion text itself. If that wording must
# change, drop "hk fault type" from BANNED rather than altering the mandated
# conclusion, and record why here.
BANNED = ["hk accuracy", "hk f1", "hk recall", "hk precision",
          "validated on hk", "fault ground truth", "confirmed fault",
          "p(fault)", "no collapse", "collapse threshold", "detection rate",
          "anomaly detected", "fault detected", "fault rate",
          "confirmed event", "hk fault classification", "hk fault type",
          "probability of fault"]
_checked = []


def check(text, where):
    low = str(text).lower()
    for b in BANNED:
        assert b not in low, f"BANNED PHRASE {b!r} in {where}: {text!r}"
    _checked.append((where, str(text)))
    return text


HERE = Path(__file__).resolve().parent           # Objective3_Outputs/objective3_transfer
NAVE = HERE.parent.parent
TRANSFER = HERE
DESIGN = NAVE / "Objective3_Outputs" / "objective3_design"
OUT = NAVE / "Objective3_Outputs" / "objective3_figures"

PERMITTED = {
    "objective3_source_checkpoint.csv", "objective3_hk_screening.csv",
    "objective3_screening_distribution.csv", "objective3_per_fold_screening.csv",
    "objective3_temporal_clustering.csv", "objective3_objective2_agreement.csv",
    "objective3_screening_checks.csv", "objective3_stress_tests.csv",
    "objective3_capacity_sensitivity.csv", "objective3_transfer_report.md",
}
FORBIDDEN_TOKENS = ["01_HK_Detection", "02_Fault_Classification",
                    "residuals_full", "pv_cross_dataset", "no_sensor",
                    "lstm_with_sensor", ".csv.gz", "fault_dataset"]
_read = []


def rd(p, **kw):
    s = str(p)
    for t in FORBIDDEN_TOKENS:
        assert t not in s, f"FORBIDDEN PATH: {s}"
    nm = Path(p).name
    assert (nm in PERMITTED) or (DESIGN in Path(p).parents), \
        f"NOT ON THE PERMITTED LIST: {nm}"
    _read.append(str(Path(p).relative_to(NAVE)))
    return pd.read_csv(p, **kw)


print("=" * 78)
print("  OBJECTIVE 3 — CONSOLIDATION (read-only; nothing trained)")
print("=" * 78)

VERIFY_ONLY = "--verify-only" in sys.argv

# ═════════════════════════════════════════════════════════════════════════════
# LOAD — permitted Objective 3 artefacts only
# ═════════════════════════════════════════════════════════════════════════════
chk = rd(TRANSFER / "objective3_source_checkpoint.csv")
hkp = rd(TRANSFER / "objective3_hk_screening.csv", index_col=0)
hkp.index = pd.to_datetime(hkp.index)
dist = rd(TRANSFER / "objective3_screening_distribution.csv")
perfold = rd(TRANSFER / "objective3_per_fold_screening.csv")
temporal = rd(TRANSFER / "objective3_temporal_clustering.csv")
agree = rd(TRANSFER / "objective3_objective2_agreement.csv")
p5g = rd(TRANSFER / "objective3_screening_checks.csv")
stress = rd(TRANSFER / "objective3_stress_tests.csv")
capsens = rd(TRANSFER / "objective3_capacity_sensitivity.csv")

print("\nINPUT FILES READ:")
for f in _read:
    print("  " + f)

CLASSES = ["Normal", "Short-Circuit", "Degradation", "Open-Circuit", "Shadowing"]
COL = {"Normal": "dimgrey", "Short-Circuit": "tomato",
       "Degradation": "darkorange", "Open-Circuit": "mediumpurple",
       "Shadowing": "steelblue"}
PRIMARY = "clean-sensor 2023 (Jan/Feb/Nov/Dec)"
N_PRIM = int(stress[stress.population == PRIMARY]
             .feature_complete_denominator.iloc[0])

# ── derive the 5x5 and 2x2 from the checkpoint file ────────────────────────
CM5 = np.zeros((5, 5), dtype=np.int64)
for i, t in enumerate(CLASSES):
    for j, p in enumerate(CLASSES):
        row = chk[chk.quantity == f"cm[{t}->{p}]"]
        assert len(row) == 1, f"missing checkpoint cell cm[{t}->{p}]"
        CM5[i, j] = int(row.reproduced.iloc[0])
CELLS_EXACT = bool((chk[chk.quantity.str.startswith("cm[")].abs_diff == 0).all())


def cval(q):
    r = chk[chk.quantity == q]
    assert len(r) == 1, f"missing checkpoint quantity {q}"
    return r.reproduced.iloc[0]


TN, FP = int(cval("binary_TN")), int(cval("binary_FP"))
FN, TP = int(cval("binary_FN")), int(cval("binary_TP"))
ANOM_F1 = float(chk[chk.quantity == "anomalous_f1"].frozen.iloc[0])
BAL_ACC = float(chk[chk.quantity == "balanced_accuracy"].frozen.iloc[0])
MACRO = float(chk[chk.quantity == "macro_f1"].frozen.iloc[0])

# ── recompute fold agreement from the permitted per-row predictions ────────
fold_cols = [f"fold{i}_pred" for i in range(1, 6)]
FP_ARR = hkp[fold_cols].to_numpy()
bin_arr = (FP_ARR != "Normal").astype(int)
UNANIMOUS_BIN = float(((bin_arr.sum(axis=1) == 0)
                       | (bin_arr.sum(axis=1) == 5)).mean())
pair = []
for i in range(5):
    for j in range(i + 1, 5):
        pair.append(float((bin_arr[:, i] == bin_arr[:, j]).mean()))
PAIR_MIN, PAIR_MAX = min(pair), max(pair)
ENS_RATE = float(hkp["binary_screened_non_normal"].mean())
SCORE = hkp["ensemble_anomalous_score_uncalibrated"].to_numpy()
QS = {p: float(np.percentile(SCORE, p)) for p in (5, 25, 50, 75, 95)}

# ── recompute run lengths from timestamps and cross-check saved values ──────
srt = hkp.sort_index()
ts = srt.index
fl = srt["binary_screened_non_normal"].to_numpy()
STEP = pd.Timedelta(minutes=15)
runs, cur = [], 0
for i in range(len(ts)):
    if fl[i] == 1:
        cont = (i > 0 and fl[i - 1] == 1 and (ts[i] - ts[i - 1]) == STEP)
        cur = cur + 1 if cont else 1
    else:
        if cur:
            runs.append(cur)
        cur = 0
if cur:
    runs.append(cur)
runs = np.array(runs)


def tval(k):
    return float(temporal[temporal.statistic == k].value.iloc[0])


assert len(runs) == int(tval("n_anomalous_runs")), "run count disagrees"
assert runs.max() == int(tval("max_run_length_intervals")), "max run disagrees"
assert int((runs == 1).sum()) == int(tval("runs_of_length_1")), "singletons"
print(f"\n  Run-length recomputation matches the frozen summary: PASS "
      f"({len(runs)} runs, max {runs.max()}, {int((runs==1).sum())} singletons)")

# Cross-check the recomputed pairwise range against the saved Part 5g text.
_p5g_txt = " ".join(p5g["result"].dropna().astype(str))
assert f"{PAIR_MIN*100:.2f}-{PAIR_MAX*100:.2f}%" in _p5g_txt, \
    "recomputed pairwise agreement disagrees with the frozen Part 5g record"
print(f"  Pairwise fold agreement matches the frozen record: PASS "
      f"({PAIR_MIN*100:.2f}-{PAIR_MAX*100:.2f}%)")

ex = p5g[p5g.fold.notna()].sort_values("fold")

# ═════════════════════════════════════════════════════════════════════════════
# TEXT — every string passes the terminology guard before anything is drawn
# ═════════════════════════════════════════════════════════════════════════════
QUAL = ("HK outputs are classifier-screened diagnostic predictions, "
        "not validated physical fault assignments.")

T1 = check("Brazil source performance on the two-feature shared contract", "fig1 title")
A1 = check("This establishes that the shared two-feature representation carries "
           "substantial source-domain Normal versus non-Normal discrimination, "
           "so a weak HK result cannot be dismissed as arising from a "
           "representation with no demonstrated source-domain discriminatory "
           "value.\nAll 25 cells of the pooled out-of-fold matrix reproduced "
           "the frozen matrix exactly.", "fig1 annotation")

T2 = check("HK diagnostic five-class output versus Brazil reference distributions",
           "fig2 title")
A2 = check("Output is highly concentrated on Shadowing (65.71%) and Normal "
           "(30.62%). Short-Circuit and Open-Circuit are each predicted for 6 "
           "rows (0.13%). The near-absence of Short-Circuit and Open-Circuit "
           "predictions is consistent with the design conclusion that those "
           "fault-type assignments are not defensible on HK using the available "
           "aggregate sensing.\nNeither Brazil distribution is an expected HK "
           "prior, and HK predictions are not required to match either. "
           + QUAL, "fig2 annotation")

T3 = check("Screening behaviour is stable across source folds", "fig3 title")
A3 = check(f"The concentration is not attributable to a single unstable fold "
           f"model: unanimous binary agreement {UNANIMOUS_BIN*100:.2f}%, "
           f"pairwise {PAIR_MIN*100:.2f}-{PAIR_MAX*100:.2f}%. The uncalibrated "
           f"ensemble anomalous score is used for ranking and shape only; it is "
           f"not an estimated probability that the HK system is faulty, and the "
           f"binary flag derives solely from the five-class argmax. No cut-off "
           f"is drawn on it.", "fig3 annotation")

T4 = check("Screening rate by population, and HK containment within source support",
           "fig4 title")
A4L = check("Aggregate screening prevalence was nearly unchanged between the "
            "clean-sensor and March-October populations, but this is NOT a "
            "matched counterfactual: the periods differ in time of year and "
            "operating conditions, not only sensor integrity, and identical "
            "aggregate rates may conceal different row-level predictions.\n"
            "The Saola figure rests on 475 of 662 eligible rows with 187 "
            "excluded (186 missing irr_meas, 1 missing power_W), so the "
            "retained subset may be selective; it is evidence neither for nor "
            "against PV-event detection.", "fig4 left annotation")
A4R = check("Most HK rows lie within broad source support, so gross "
            "out-of-range extrapolation alone is unlikely to explain the very "
            "high screening rate; local and conditional distribution shift "
            "remain possible and are not excluded, and approximately 10-14% "
            "fall outside occupied training cells.", "fig4 right annotation")

T5 = check("Temporal structure of screened flags, and comparison with "
           "sensor-free screening", "fig5 title")
A5 = check("Agreement with Objective 2 sensor-free Arm B / Arm C screening is "
           "descriptive only and is not independent fault validation. Arms B/C "
           "do not consume the corrupted measured-irradiance signal used by "
           "Objective 3, but both analyses concern the same HK site and period "
           "and neither has verified HK fault labels; therefore agreement "
           "cannot be presented as corroboration of a true PV event.\n"
           "Flags form long coherent daytime blocks rather than scattered "
           "isolated timestamps. The two screening signals differ markedly - "
           "Objective 3 screens approximately 69% of rows where Objective 2 "
           "flags under 1%.", "fig5 annotation")

for lab in ["classifier-screened non-Normal flags", "uncalibrated ensemble "
            "anomalous score", "screening rate (% of feature-complete rows)",
            "predicted proportion", "run length (consecutive 15-min intervals)",
            "containment (%)", "rows"]:
    check(lab, "axis label")

print(f"\nTerminology guard: {len(_checked)} strings checked against "
      f"{len(BANNED)} banned phrases — all PASS")
print("\nFIGURE TEXT:")
for w, t in _checked:
    print(f"  [{w}] {t[:150]}{'...' if len(t) > 150 else ''}")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY DOCUMENT
# ═════════════════════════════════════════════════════════════════════════════
sp = stress.set_index("population")


def srow(pop, col):
    return sp.loc[pop, col]


SUMMARY = check(f"""# Objective 3 — Cross-Domain Transfer: Summary

Generated: {datetime.now():%Y-%m-%dT%H:%M:%S}
All results frozen. Nothing was retrained or re-run for this consolidation.

## 1. What Objective 3 tested

Objective 3 asks whether a fault classifier trained on the Brazil source domain
can be applied directly to the Hong Kong SQ1 installation, and what its output
can legitimately mean there. The question was **reframed during design** away
from validated fault-type assignment on the target site, and towards
**cross-domain compatibility and potential-fault screening**.

The original framing was not testable. Hong Kong has no per-timestep fault
labels, so no accuracy-style quantity can be computed on HK at all. Brazil
measures pre-inverter DC array power while HK records post-inverter AC active
power. The two sites differ in sensing (Brazil has string-level voltage and
current; HK has none at any station), in capacity (5.00 kW against 27.6 kW), and
in temporal support (1 Hz against 15-minute means).

## 2. The shared feature contract and why it is small

The two quantities retained for exploratory transfer were capacity-normalised
power and measured irradiance, both treated as proxy-comparable rather than
physically equivalent across domains.

Objective 1 showed that the strongest five-class source performance relied on
string-level electrical measurements, with Short-Circuit discrimination
particularly dependent on voltage-derived information unavailable in HK.

The three gates were resolved as follows.

| Gate | Status |
|---|---|
| **1. Semantic** | Fails for fault-type transfer; partial for binary screening. Both shared quantities are proxy-comparable only. |
| **2. Domain overlap** | Marginal — a qualitative design judgement, not a predefined statistical threshold. Meaningful overlap exists, but the relationship to Brazil Normal is shifted. |
| **3. Source performance** | Evidenced from frozen Objective 1 outputs: five-class Macro F1 {MACRO:.4f}, collapsed anomalous F1 {ANOM_F1:.4f}, balanced accuracy {BAL_ACC:.4f}. |

## 3. Method

Five frozen `POWER_PLUS_IRR` fold pipelines were refitted, because no fitted
Brazil artefact survived Objective 1. Before any HK data was accessed, the refit
was verified against the frozen out-of-fold evidence **at count level**: all 25
cells of the pooled five-class matrix and all four cells of the collapsed binary
matrix reproduced exactly.

Brazil-fitted preprocessing was then applied unchanged. HK power was converted
into Brazil-capacity-equivalent watts (x 0.18112005) so that the scaler, fitted
on raw Brazil watts, received values in its own parameterisation; irradiance
passed through in native W/m². The algebraic identity underpinning this held to
6.7e-16 across all five folds.

Each HK row passed through each fold's own imputer, scaler and model
independently. Probabilities were reindexed into the canonical class order via
each model's `classes_` before averaging. The mean-probability ensemble is a
**pre-specified deployment summary**: Gate 3 validates the underlying
configuration and representation, not the aggregation rule, which was never
evaluated out-of-fold.

No target-domain adaptation was performed — no recentring, no calibration, no
rescaling on HK statistics, no threshold tuning and no model shopping.

## 4. Results

**Source performance.** Pooled out-of-fold, the two-feature configuration
reaches Macro F1 {MACRO:.4f} across five classes. Collapsed to Normal versus
non-Normal it reaches anomalous F1 {ANOM_F1:.4f} and balanced accuracy
{BAL_ACC:.4f} (TN {TN:,} / FP {FP:,} / FN {FN:,} / TP {TP:,}).

**HK diagnostic class distribution**, clean-sensor primary population,
denominator {N_PRIM:,} feature-complete rows:

| Class | Predicted | Brazil empirical | Brazil balanced training |
|---|---|---|---|
""" + "\n".join(
    f"| {c} | {int(dist[(dist.population==PRIMARY)&(dist['class']==c)].predicted_count.iloc[0]):,} "
    f"({float(dist[(dist.population==PRIMARY)&(dist['class']==c)].predicted_proportion.iloc[0])*100:.2f}%) | "
    f"{float(dist[(dist.population==PRIMARY)&(dist['class']==c)].brazil_empirical_label_proportion.iloc[0])*100:.2f}% | "
    f"{float(dist[(dist.population==PRIMARY)&(dist['class']==c)].brazil_balanced_training_proportion.iloc[0])*100:.2f}% |"
    for c in CLASSES) + f"""

Neither Brazil column is an expected HK prior. Output is highly concentrated on
Shadowing, and Short-Circuit and Open-Circuit are each predicted for only 6 rows.

**Per-fold stability.** Individual fold screening rates span
{perfold.screening_rate.min()*100:.2f}-{perfold.screening_rate.max()*100:.2f}%
against an ensemble rate of {ENS_RATE*100:.2f}%, with unanimous binary agreement
of {UNANIMOUS_BIN*100:.2f}% and pairwise agreement of
{PAIR_MIN*100:.2f}-{PAIR_MAX*100:.2f}%. The concentration is not the product of
one aberrant fold.

**Uncalibrated ensemble anomalous score.** Median {QS[50]:.4f}, p5 {QS[5]:.4f},
p25 {QS[25]:.4f}, p75 {QS[75]:.4f}, p95 {QS[95]:.4f}. The distribution is
strongly bimodal, with dense mass at both extremes. It is used for ranking and
shape only, is not an estimated probability that the HK system is faulty, and no
cut-off was applied to it — the binary flag follows solely from the five-class
argmax.

**Temporal structure.** {len(runs):,} runs of consecutive screened flags, mean
{runs.mean():.2f} intervals, median {int(np.median(runs))}, maximum {runs.max()}
(approximately {runs.max()*15/60:.1f} hours), with {int((runs==1).sum())}
singletons. Runs are defined by exact 15-minute timestamp continuity, breaking at
any gap and at the February-November discontinuity. Flags form long coherent
daytime blocks.

**Populations.** Every rate below is stated against its own denominator of
feature-complete rows.

| Population | Role | Eligible | Feature-complete | Excluded | Screening rate |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.population} | {r.label} | {int(r.raw_eligible):,} | "
    f"{int(r.feature_complete_denominator):,} | "
    f"{int(r.excluded_missing_inputs):,} | {r.screening_rate*100:.2f}% |"
    for r in stress.itertuples()) + f"""

Aggregate prevalence is nearly identical between the clean-sensor and
March-October populations. This is **not a matched counterfactual**: the periods
differ in time of year and operating conditions, not only in sensor integrity,
and identical aggregate rates may conceal different row-level predictions and
class distributions. No conclusion is drawn about the effect of irradiance
corruption in either direction.

**Support containment.** Per fold, HK convex-hull containment spans
{ex.hull_containment_pct.min():.2f}-{ex.hull_containment_pct.max():.2f}% and
occupied-bin containment {ex.bin_containment_pct.min():.2f}-{ex.bin_containment_pct.max():.2f}%,
measured against each fold's own training data by the same descriptive methods
used for Gate 2. Roughly {100-ex.bin_containment_pct.max():.0f}-{100-ex.bin_containment_pct.min():.0f}%
of HK rows fall outside occupied training cells.

**Cross-screening comparison.** On {int(agree.overlapping_rows_denominator.iloc[0]):,}
overlapping rows, Objective 3 and Objective 2 Arm B agree on
{float(agree[agree.arm=='Arm B'].raw_agreement.iloc[0])*100:.2f}% of rows and Arm C on
{float(agree[agree.arm=='Arm C'].raw_agreement.iloc[0])*100:.2f}%. The two
signals differ markedly: Objective 3 screens about 69% of rows where the
Objective 2 arms flag under 1%.

**Capacity sensitivity.** Substituting the 5.28 kW nameplate for the 5.00 kW
rating changes the screening rate from
{capsens.screening_rate.iloc[0]*100:.2f}% to {capsens.screening_rate.iloc[1]*100:.2f}%.
Conclusions are unchanged.

## 5. What the results support

The two-feature shared contract carries substantial source-domain discrimination
between Normal and non-Normal, established at count level against frozen
evidence. Applied directly to Hong Kong, it produces a high screening rate
concentrated on a single class during a clean-sensor population with no known
fault event. That behaviour is stable across all five source-fold models and
occurs predominantly within broad source support. Together these constitute
evidence of limited direct cross-domain transferability.

## 6. What the results do not support

No validated HK fault assignment of any kind is supported; the five-class output
is diagnostic only. The experiment cannot establish that the documented
operating-distribution offset uniquely caused the predictions. It does not
support the conclusion that irradiance corruption is irrelevant — the
clean-sensor and March-October comparison is not a matched counterfactual. It
provides evidence neither for nor against PV-event detection in the Saola proxy
window. And agreement with Objective 2 screening is not independent
corroboration, since neither analysis has verified HK labels.

## 7. Limitations

The shared contract is two features. AC versus DC power semantics are
unresolved, and no inverter conversion efficiency was assumed or established.
The HK irradiance plane and the weather-station-to-SQ1 pairing remain
undocumented. Hong Kong has no per-timestep labels of any kind. The Saola proxy
window retains {int(srow('Saola proxy window (1-15 Sept 2023)','feature_complete_denominator')):,}
of {int(srow('Saola proxy window (1-15 Sept 2023)','raw_eligible')):,} eligible
rows, excluding roughly 28%, so the retained subset may be selective. Brazil
carries no timestamps, so no clock-aligned temporal analysis was possible on the
source side. The clean-sensor population contains no known fault event, so it
compares transfer behaviour rather than screening ability against truth.

## 8. Figures

**Figure 1 — Brazil source performance on the two-feature shared contract.**
Pooled five-class out-of-fold confusion matrix (row-normalised, counts
annotated) and its Normal versus non-Normal collapse, on the full Brazil
dataset under the frozen group-disjoint five-fold protocol. All 25 cells
reproduced the frozen matrix exactly.

**Figure 2 — HK diagnostic five-class output versus Brazil reference
distributions.** Predicted class proportions on the clean-sensor primary
population (n = {N_PRIM:,} feature-complete rows) against two Brazil reference
distributions, neither of which is an expected HK prior. Outputs are
classifier-screened diagnostics, not validated fault assignments.

**Figure 3 — Screening behaviour is stable across source folds.** Per-fold and
ensemble screening rates (n = {N_PRIM:,}) and the distribution of the
uncalibrated ensemble anomalous score. No cut-off is drawn on the score.

**Figure 4 — Screening rate by population, and HK containment within source
support.** Rates for all four populations with denominators printed, corrupted-
input stress tests hatched, alongside per-fold hull and occupied-bin containment
of HK within each fold's own training data.

**Figure 5 — Temporal structure of screened flags, and comparison with
sensor-free screening.** Run-length distribution of consecutive screened flags
under exact 15-minute timestamp continuity, and the Objective 3 versus
Objective 2 Arm B / Arm C contingency on
{int(agree.overlapping_rows_denominator.iloc[0]):,} overlapping rows. The
comparison is descriptive and is not independent fault validation.

## 9. Conclusion

The two-feature Brazil configuration demonstrated substantial grouped
source-domain Normal/non-Normal discrimination, but direct transfer to HK
produced a high, Shadowing-concentrated classifier-screening rate during the
clean-sensor/no-known-fault population. The pattern was stable across
source-fold models and occurred predominantly within broad source support. These
findings provide evidence of limited direct cross-domain transferability under
the documented semantic, sensing and operating-distribution differences. HK
fault-type predictions are diagnostic outputs only and are not validated
physical fault assignments.
""", "summary document")

print(f"\nSummary document: {len(SUMMARY):,} characters, terminology guard PASS")

if VERIFY_ONLY:
    print("\n" + "=" * 78)
    print("  VERIFY-ONLY — no directory created, no file written.")
    print(f"  Output directory exists already: {OUT.exists()}")
    print("  Would write 13 files to Objective3_Outputs/objective3_figures/:")
    for f in ["fig1_source_performance.png/.pdf",
              "fig2_hk_class_distribution.png/.pdf",
              "fig3_fold_stability_and_score.png/.pdf",
              "fig4_populations_and_support.png/.pdf",
              "fig5_temporal_and_agreement.png/.pdf",
              "objective3_summary.md", "figure_source_manifest.txt",
              "run_log.txt"]:
        print(f"    {f}")
    print("  Anomalous-score panel: the ONLY vertical lines are the five "
          "quantile markers")
    print("  (p5/p25/p50/p75/p95) the prompt requires. No threshold, cut-off "
          "or decision")
    print("  boundary is drawn, and the score is never compared against any "
          "value.")
    sys.exit(0)

# Generate figures and summary from retained transfer outputs.
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def wrap(t, w=118):
    """Wrap each paragraph separately. Long single-line fig.text under
    bbox_inches="tight" stretches the saved canvas to the width of the text."""
    return "\n".join(textwrap.fill(par, width=w) for par in t.split("\n"))


def note(fig, text, y=-0.02):
    fig.text(0.5, y, wrap(text), ha="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="whitesmoke", ec="grey"))


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 2, figsize=(15, 6.4))
rn = CM5 / CM5.sum(axis=1, keepdims=True)
im = ax[0].imshow(rn, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax[0].text(j, i, f"{rn[i, j]*100:.1f}%\n{CM5[i, j]:,}",
                   ha="center", va="center", fontsize=8,
                   color="white" if rn[i, j] > 0.5 else "black")
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(CLASSES, rotation=30,
                                                  ha="right", fontsize=9)
ax[0].set_yticks(range(5)); ax[0].set_yticklabels(CLASSES, fontsize=9)
ax[0].set_xlabel("predicted", fontsize=11); ax[0].set_ylabel("true", fontsize=11)
ax[0].set_title("Pooled five-class out-of-fold matrix (row-normalised)\n"
                f"Macro F1 {MACRO:.4f} — all 25 cells reproduced exactly",
                fontsize=12)
fig.colorbar(im, ax=ax[0], fraction=0.046, label="row proportion")

b = np.array([[TN, FP], [FN, TP]])
bn = b / b.sum(axis=1, keepdims=True)
im2 = ax[1].imshow(bn, cmap="Greens", vmin=0, vmax=1)
lab2 = ["Normal", "non-Normal"]
for i in range(2):
    for j in range(2):
        ax[1].text(j, i, f"{bn[i, j]*100:.2f}%\n{b[i, j]:,}", ha="center",
                   va="center", fontsize=12,
                   color="white" if bn[i, j] > 0.5 else "black")
ax[1].set_xticks(range(2)); ax[1].set_xticklabels(lab2, fontsize=10)
ax[1].set_yticks(range(2)); ax[1].set_yticklabels(lab2, fontsize=10)
ax[1].set_xlabel("predicted", fontsize=11); ax[1].set_ylabel("true", fontsize=11)
ax[1].set_title(f"Collapsed Normal vs non-Normal\n"
                f"anomalous F1 {ANOM_F1:.4f} | balanced accuracy {BAL_ACC:.4f}",
                fontsize=12)
fig.colorbar(im2, ax=ax[1], fraction=0.046, label="row proportion")
fig.suptitle(T1, fontsize=13)
note(fig, A1, -0.16)
fig.tight_layout()
save(fig, "fig1_source_performance")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2
# ═════════════════════════════════════════════════════════════════════════════
fig, a = plt.subplots(figsize=(12, 6.4))
d0 = dist[dist.population == PRIMARY].set_index("class").reindex(CLASSES)
x = np.arange(5)
a.bar(x - 0.27, d0.predicted_proportion.to_numpy(), 0.27,
      label=f"HK predicted (n = {N_PRIM:,})",
      color=[COL[c] for c in CLASSES], edgecolor="black", linewidth=0.6)
a.bar(x, d0.brazil_empirical_label_proportion.to_numpy(), 0.27,
      label="Brazil empirical labels", color="lightgrey", edgecolor="black",
      linewidth=0.6)
a.bar(x + 0.27, d0.brazil_balanced_training_proportion.to_numpy(), 0.27,
      label="Brazil balanced source-training", color="white", edgecolor="black",
      hatch="//", linewidth=0.6)
for c, v in [("Shadowing", 0.6571), ("Normal", 0.3062)]:
    a.annotate(f"{v*100:.2f}%", (CLASSES.index(c) - 0.27, v),
               textcoords="offset points", xytext=(0, 6), ha="center",
               fontsize=10, fontweight="bold")
a.set_xticks(x); a.set_xticklabels(CLASSES, fontsize=10)
a.set_ylim(0, 0.82)          # headroom: keep the legend clear of every bar
a.set_ylabel("predicted proportion", fontsize=11)
a.set_title(T2 + f"\nclean-sensor primary population, denominator "
                 f"{N_PRIM:,} feature-complete rows", fontsize=12)
a.legend(fontsize=9, loc="upper center", framealpha=0.95)
a.grid(alpha=0.3, axis="y")
note(fig, A2, -0.10)
fig.tight_layout()
save(fig, "fig2_hk_class_distribution")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
a = ax[0]
a.bar([f"fold {int(f)}" for f in perfold.fold],
      perfold.screening_rate.to_numpy() * 100, color="steelblue",
      edgecolor="black", linewidth=0.6)
a.axhline(ENS_RATE * 100, ls="--", c="black",
          label=f"ensemble {ENS_RATE*100:.2f}%")
for i, v in enumerate(perfold.screening_rate.to_numpy() * 100):
    # inside the bar: the ensemble reference line sits at ~69.4%, so labels
    # placed just above each bar top would collide with it
    a.text(i, v - 4.0, f"{v:.2f}%", ha="center", fontsize=9.5,
           color="white", fontweight="bold")
a.set_ylabel("screening rate (% of feature-complete rows)", fontsize=11)
a.set_ylim(0, 80)
a.set_title(f"Per-fold classifier-screened non-Normal rate\n"
            f"all rates on n = {N_PRIM:,} feature-complete clean-sensor HK rows",
            fontsize=12)
a.legend(fontsize=9); a.grid(alpha=0.3, axis="y")

a = ax[1]
a.hist(SCORE, bins=60, color="slategrey", edgecolor="none")
# p75 (0.9997) and p95 (1.0000) almost coincide, so labels are staggered
# vertically and alternate side rather than overprinting each other.
_top = a.get_ylim()[1]
for p, c, frac, side in [(5, "tab:green", 0.97, "left"),
                         (25, "tab:olive", 0.97, "left"),
                         (50, "tab:red", 0.97, "right"),
                         (75, "tab:olive", 0.72, "right"),
                         (95, "tab:green", 0.47, "right")]:
    a.axvline(QS[p], color=c, ls=":", lw=1.4)
    a.annotate(f"p{p} = {QS[p]:.4f}", xy=(QS[p], _top * frac),
               xytext=(-6 if side == "right" else 6, 0),
               textcoords="offset points", fontsize=8, color=c,
               ha="right" if side == "right" else "left", va="top",
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c, lw=0.6))
a.set_xlabel("uncalibrated ensemble anomalous score", fontsize=11)
a.set_ylabel("rows", fontsize=11)
a.set_title("Distribution of the uncalibrated ensemble anomalous score\n"
            "quantile markers only — no cut-off is applied to this score",
            fontsize=12)
a.grid(alpha=0.3)
fig.suptitle(T3, fontsize=13)
note(fig, A3 + "\n" + QUAL, -0.22)
fig.tight_layout()
save(fig, "fig3_fold_stability_and_score")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 2, figsize=(16, 6.4))
a = ax[0]
short = {PRIMARY: "clean-sensor\n(PRIMARY)", "March-October 2023":
         "March-October\n(corrupted-input\nstress test)",
         "full 2023": "full 2023\n(mixed-period\nstress summary)",
         "Saola proxy window (1-15 Sept 2023)":
         "Saola window\n(corrupted-input\nstress test)"}
hatches = {PRIMARY: "", "March-October 2023": "//", "full 2023": "..",
           "Saola proxy window (1-15 Sept 2023)": "//"}
cols = {PRIMARY: "steelblue", "March-October 2023": "indianred",
        "full 2023": "darkgrey",
        "Saola proxy window (1-15 Sept 2023)": "indianred"}
for i, r in enumerate(stress.itertuples()):
    a.bar(i, r.screening_rate * 100, color=cols[r.population],
          hatch=hatches[r.population], edgecolor="black", linewidth=0.7)
    a.text(i, r.screening_rate * 100 + 1.2, f"{r.screening_rate*100:.2f}%",
           ha="center", fontsize=10, fontweight="bold")

a.set_xticks(range(len(stress)))
a.set_xticklabels([f"{short[p]}\nn = {int(d):,}" for p, d in
                   zip(stress.population,
                       stress.feature_complete_denominator)], fontsize=8.5)
a.set_ylim(0, 82)
a.set_ylabel("screening rate (% of feature-complete rows)", fontsize=11)
a.set_title("Screening rate by population\n"
            "hatched = corrupted-input stress test; dotted = mixed-period summary",
            fontsize=12)
a.grid(alpha=0.3, axis="y")

a = ax[1]
w = 0.38
fx = np.arange(len(ex))
a.bar(fx - w / 2, ex.hull_containment_pct.to_numpy(), w, label="convex-hull containment",
      color="cadetblue", edgecolor="black", linewidth=0.6)
a.bar(fx + w / 2, ex.bin_containment_pct.to_numpy(), w,
      label="occupied-bin containment (20x20)", color="sandybrown",
      edgecolor="black", linewidth=0.6)
for i, (h, bb) in enumerate(zip(ex.hull_containment_pct, ex.bin_containment_pct)):
    a.text(i - w / 2, h + 0.5, f"{h:.2f}", ha="center", fontsize=8)
    a.text(i + w / 2, bb + 0.5, f"{bb:.2f}", ha="center", fontsize=8)
a.set_xticks(fx); a.set_xticklabels([f"fold {int(f)}" for f in ex.fold], fontsize=10)
a.set_ylim(70, 100)
a.set_ylabel("containment (%)", fontsize=11)
a.set_title(f"HK containment within each fold's own Brazil training data\n"
            f"denominator {N_PRIM:,} feature-complete clean-sensor HK rows",
            fontsize=12)
a.legend(fontsize=9); a.grid(alpha=0.3, axis="y")
fig.suptitle(T4, fontsize=13)
note(fig, A4L + "\n" + A4R, -0.16)
fig.tight_layout()
save(fig, "fig4_populations_and_support")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
a = ax[0]
a.hist(runs, bins=np.arange(0.5, runs.max() + 1.5, 1), color="steelblue",
       edgecolor="black", linewidth=0.4)
a.axvline(runs.mean(), color="tab:red", ls="--",
          label=f"mean {runs.mean():.2f} intervals")
a.set_xlabel("run length (consecutive 15-min intervals)", fontsize=11)
a.set_ylabel("number of runs", fontsize=11)
a.set_title(f"Run lengths of consecutive classifier-screened non-Normal flags\n"
            f"{len(runs):,} runs | max {runs.max()} | "
            f"{int((runs==1).sum())} singletons | n = {N_PRIM:,} rows",
            fontsize=12)
a.legend(fontsize=9); a.grid(alpha=0.3)
a.text(0.97, 0.72, "runs defined by exact 15-minute timestamp\ncontinuity, "
       "breaking at gaps and at the\nFebruary-November discontinuity",
       transform=a.transAxes, ha="right", fontsize=8,
       bbox=dict(boxstyle="round", fc="white", ec="grey"))

a = ax[1]
cats = ["both", "Objective 3 only", "Objective 2 only", "neither"]
keys = ["both_flagged", "objective3_only", "objective2_only", "neither"]
xx = np.arange(4)
for k, (arm, off, colr) in enumerate([("Arm B", -0.2, "seagreen"),
                                      ("Arm C", 0.2, "mediumpurple")]):
    r = agree[agree.arm == arm].iloc[0]
    vals = [int(r[c]) for c in keys]
    a.bar(xx + off, vals, 0.4, label=arm, color=colr, edgecolor="black",
          linewidth=0.6)
    for i, v in enumerate(vals):
        a.text(i + off, v + 40, f"{v:,}", ha="center", fontsize=8)
a.set_xticks(xx); a.set_xticklabels(cats, fontsize=9.5)
a.set_ylabel("rows", fontsize=11)
a.set_title(f"Objective 3 vs Objective 2 sensor-free screening\n"
            f"{int(agree.overlapping_rows_denominator.iloc[0]):,} overlapping "
            f"rows — descriptive comparison only", fontsize=12)
a.legend(fontsize=9); a.grid(alpha=0.3, axis="y")
fig.suptitle(T5, fontsize=13)
note(fig, A5, -0.13)
fig.tight_layout()
save(fig, "fig5_temporal_and_agreement")

# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENTS
# ═════════════════════════════════════════════════════════════════════════════
(OUT / "objective3_summary.md").write_text(SUMMARY)

MANIFEST = f"""Objective 3 consolidation — figure source manifest
Generated: {datetime.now():%Y-%m-%dT%H:%M:%S}

All inputs are frozen Objective 3 artefacts. No Objective 1 or Objective 2
output and no dataset was read during consolidation.

fig1_source_performance
    objective3_source_checkpoint.csv   (pooled 5x5 cells, binary TN/FP/FN/TP,
                                  macro_f1, anomalous_f1, balanced_accuracy)

fig2_hk_class_distribution
    objective3_screening_distribution.csv  (HK predicted counts and proportions,
                                       both Brazil reference distributions)
    objective3_stress_tests.csv             (denominator for the primary population)

fig3_fold_stability_and_score
    objective3_per_fold_screening.csv         (per-fold screening rates)
    objective3_hk_screening.csv      (uncalibrated score; per-fold predictions,
                                  from which unanimous and pairwise agreement
                                  were recomputed)
    objective3_screening_checks.csv       (cross-check of the recomputed pairwise range)

fig4_populations_and_support
    objective3_stress_tests.csv        (screening rate, eligible / feature-complete /
                                  excluded counts, per-feature missingness)
    objective3_screening_checks.csv       (per-fold hull and occupied-bin containment)

fig5_temporal_and_agreement
    objective3_hk_screening.csv      (timestamps and flags; run lengths recomputed
                                  under 15-minute continuity)
    objective3_temporal_clustering.csv (frozen run statistics, used to verify the
                                  recomputation)
    objective3_objective2_agreement.csv (Arm B / Arm C contingency — the only source
                                   for the cross-screening panel)

objective3_summary.md
    all of the above, plus objective3_capacity_sensitivity.csv
"""
(OUT / "figure_source_manifest.txt").write_text(MANIFEST)

with open(OUT / "run_log.txt", "a") as fh:
    fh.write(f"{datetime.now():%Y-%m-%dT%H:%M:%S} | plot_objective3_results.py | "
             f"input: {len(_read)} frozen Objective 3 artefacts | "
             f"output: 13 files in Objective3_Outputs/objective3_figures/ | "
             f"notes: consolidation only; nothing trained, no transfer run, "
             f"no Objective 1 or 2 output read\n")

files = sorted(OUT.iterdir())
print(f"\nWrote {len(files)} files to {OUT}:")
for f in files:
    print(f"  {f.stat().st_size:>9,}  {f.name}")
print("\nOBJECTIVE 3 CONSOLIDATION COMPLETE")
