"""
plot_objective2_results.py
==========================
Objective 2 — consolidation into a dissertation-ready figure set and written
summary. Reads existing outputs only. No modelling, no retraining, no new
experiments.

PERMITTED INPUTS (read-only, all under outputs/objective2_modeling/):
  irradiance_estimator_metrics.csv          Level 1
  expected_power_accuracy.csv               Level 2
  anomaly_screening.csv                     Level 3
  arm_training_summary.csv
  residuals_full_*.csv                      four arms
  residual_diagnostic/residual_monthly_stats.csv
  residual_diagnostic/residual_by_irradiance_bin.csv
  residual_diagnostic/threshold_offset_analysis.csv
  residual_diagnostic/detrended_diagnostic.csv
  objective2_grid.csv                       FIGURE 1 ONLY — timestamp, measured
                                            irradiance, ghi_clear, irr_est.
                                            This is the final modelling grid.

FORBIDDEN: outputs/no_sensor/, outputs/lstm_with_sensor/, and any other
historical HK directory. Those results are excluded by assertion on every path
opened.
"""

import datetime
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
HK          = PROJECT_DIR / "outputs" / "objective2_modeling"
DIAG        = HK / "residual_diagnostic"
OUT_DIR     = PROJECT_DIR / "outputs" / "objective2_figures"
RUN_LOG     = PROJECT_DIR / "outputs" / "run_log.txt"

TZ = "Asia/Hong_Kong"
P0, P1 = pd.Timestamp("2023-09-01", tz=TZ), pd.Timestamp("2023-09-15", tz=TZ)
TRAIN_END = pd.Timestamp("2022-12-31", tz=TZ)
FAULT0, FAULT1 = pd.Timestamp("2023-03-01", tz=TZ), pd.Timestamp("2023-10-31", tz=TZ)
CLEAN_MONTHS = [1, 2, 11, 12]

ARMS = {"Reference (measured irradiance)": "reference",
        "Arm A (pvlib only)": "arm_a",
        "Arm B (ML estimate)": "arm_b",
        "Arm C (pvlib + ML estimate)": "arm_c"}
SHORT = {"Reference (measured irradiance)": "Reference arm\n(measured irradiance)",
         "Arm A (pvlib only)": "Arm A\n(pvlib clear-sky)",
         "Arm B (ML estimate)": "Arm B\n(ML same-time estimate)",
         "Arm C (pvlib + ML estimate)": "Arm C\n(clear-sky + ML estimate)"}
COLORS = {"Reference (measured irradiance)": "dimgrey",
          "Arm A (pvlib only)": "steelblue",
          "Arm B (ML estimate)": "seagreen",
          "Arm C (pvlib + ML estimate)": "mediumpurple"}

plt.rcParams.update({"xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.labelsize": 11, "axes.titlesize": 12,
                     "legend.fontsize": 9})

# ── Terminology guard ────────────────────────────────────────────────────────
# NOTE ON AN INTERNAL CONFLICT IN THE BRIEF: the banned list contains
# "fault ground truth" and "fault labels", yet the specified Figure 5
# annotation text used the former. The ban takes precedence; the same meaning
# is carried by "no per-timestep fault verification exists for this site".
BANNED = ["fault recall", "false positive rate", "false-positive rate",
          "fault precision", "fpr", "fault ground truth", "fault label",
          "gold standard", "upper bound", "false alarm rate", "ceiling"]
BANNED_WORD = ["recall"]          # standalone word, any use

_checked = []


def check(text, where):
    """Assert terminology compliance and record the string for reporting."""
    low = text.lower()
    hits = [b for b in BANNED if b in low]
    hits += [w for w in BANNED_WORD if re.search(rf"\b{w}\b", low)]
    assert not hits, f"TERMINOLOGY VIOLATION in {where}: {hits} -> {text!r}"
    _checked.append((where, text))
    return text


# ── Path guard ───────────────────────────────────────────────────────────────
FORBIDDEN_DIRS = ["outputs/no_sensor", "outputs/lstm_with_sensor",
                  "outputs/calibration", "outputs/eda_days",
                  "outputs/eda_features", "outputs/hk_data_audit",
                  "outputs/objective2_design"]
_read = []


def rd(path, **kw):
    p = Path(path)
    s = str(p).replace("\\", "/")
    bad = [f for f in FORBIDDEN_DIRS if f in s]
    assert not bad, f"FORBIDDEN DIRECTORY READ: {bad} -> {p}"
    assert "/outputs/objective2_modeling" in s, f"Read outside permitted tree: {p}"
    _read.append(str(p.relative_to(PROJECT_DIR)))
    return pd.read_csv(p, **kw)


print("=" * 78)
print("  OBJECTIVE 2 CONSOLIDATION")
print("=" * 78)

lvl1 = rd(HK / "irradiance_estimator_metrics.csv")
lvl2 = rd(HK / "expected_power_accuracy.csv")
lvl3 = rd(HK / "anomaly_screening.csv")
trainsum = rd(HK / "arm_training_summary.csv").set_index("arm")
mon = rd(DIAG / "residual_monthly_stats.csv")
bins = rd(DIAG / "residual_by_irradiance_bin.csv")
offs = rd(DIAG / "threshold_offset_analysis.csv")
detr = rd(DIAG / "detrended_diagnostic.csv")
grid = rd(HK / "objective2_grid.csv", index_col=0,
          usecols=["Unnamed: 0", "irr_meas", "ghi_clear", "irr_est", "eligible"])
grid.index = pd.to_datetime(grid.index, utc=True).tz_convert(TZ)

R = {}
for arm, tag in ARMS.items():
    d = rd(HK / f"residuals_full_{tag}.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True).tz_convert(TZ)
    R[arm] = d

print("\nINPUT FILES READ:")
for f in _read:
    print(f"  {f}")
print(f"\n  total {len(_read)} files; forbidden-directory assertion: PASS")

# ── Sourced values ───────────────────────────────────────────────────────────
clean = lvl1[lvl1["slice"].str.contains("clean-sensor")].iloc[0]
EST_MAE, EST_RMSE, EST_R2 = clean.est_MAE, clean.est_RMSE, clean.est_R2
CS_MAE, CS_RMSE, CS_R2 = clean.clearsky_MAE, clean.clearsky_RMSE, clean.clearsky_R2


def nmae(sl, arm):
    r = lvl2[(lvl2["slice"] == sl) & (lvl2["arm"] == arm)]
    return float(r.iloc[0].nMAE_pct) if len(r) else np.nan


SLICE_LABELS = {
    "Slice 1": "Slice 1: Jan–Feb + Nov–Dec 2023\n(clean sensor, no known fault event)",
    "Slice 2": "Slice 2: full 2023\n(Reference unreliable Mar–Oct)",
    "Slice 3": "Slice 3: Saola proxy window\n(416 rows)",
}

# ── Figure text, all guarded ─────────────────────────────────────────────────
T1 = check("Same-time irradiance estimation versus pvlib clear-sky", "fig1 title")
A1 = check(
    f"ML same-time estimation approximately halves irradiance error relative to "
    f"pvlib clear-sky: MAE {EST_MAE:.1f} vs {CS_MAE:.1f} W/m², "
    f"R² {EST_R2:.3f} vs {CS_R2:.3f}. Evaluated on clean-sensor months only "
    f"(Jan, Feb, Nov, Dec 2023).", "fig1 annotation")
T2 = check("Expected AC-power prediction accuracy by irradiance representation",
           "fig2 title")
A2 = check(
    f"1. On the fair slice the ML same-time estimate ({nmae('Slice 1','Arm B (ML estimate)'):.1f}%) "
    f"substantially outperforms pvlib clear-sky alone "
    f"({nmae('Slice 1','Arm A (pvlib only)'):.1f}%) and sits 1.9 points behind a "
    f"working sensor ({nmae('Slice 1','Reference (measured irradiance)'):.1f}%).\n"
    f"2. On full 2023 the Reference arm is the weakest arm "
    f"({nmae('Slice 2','Reference (measured irradiance)'):.1f}%) because its sensor was "
    f"corrupted: in this dataset and affected period, the degraded\n"
    f"   measured-irradiance arm performed worse than all three no-sensor arms.\n"
    f"3. Arm C ≈ Arm B throughout: combining clear-sky with the ML estimate "
    f"provides negligible additional benefit in the primary comparison "
    f"(Slice 1 {nmae('Slice 1','Arm B (ML estimate)'):.1f} → "
    f"{nmae('Slice 1','Arm C (pvlib + ML estimate)'):.1f}%; "
    f"Slice 2 {nmae('Slice 2','Arm B (ML estimate)'):.1f} → "
    f"{nmae('Slice 2','Arm C (pvlib + ML estimate)'):.1f}%).", "fig2 annotation")
T3 = check("Monthly mean residual, 2021–2023 — all arms shift together at the "
           "train/test boundary", "fig3 title")
A3 = check(
    "• Training period is flat (Arm A slope −0.8 W/month, −13 W total over 19 months)\n"
    "• All four arms drop together at January 2023 and recover partially through the year\n"
    "• NO step occurs at September 2023, eight months after the shift begins\n"
    "• Neither gradual degradation nor post-event damage is supported as the sole\n"
    "  explanation by this temporal pattern.\n"
    "• The January onset coincides with the train/test boundary, so part of the shift is\n"
    "  in-sample versus held-out prediction error and the cause cannot be attributed\n"
    "  from these outputs", "fig3 annotation")
T4 = check("Residual gap widens with irradiance, consistent with a proportional rather than fixed offset",
           "fig4 title")
A4 = check(
    "The gap widens with irradiance in every arm (Arm A −465 W in the bottom deciles "
    "against −2,248 W in the top). As a fraction of predicted power, 2023 runs −12.7% "
    "to −15.1% against −1.3% to +2.0% in the training period. 2022 residuals are "
    "in-sample and 2023 are held out, so a uniform component of the gap is expected "
    "from generalisation alone — the SLOPE with irradiance is the informative part.",
    "fig4 annotation")
T5 = check("Saola proxy event window shows no distinct residual signature", "fig5 title")
A5 = check(
    "• Pre / window / post means show no step (Arm A −1,466 / −1,558 / −1,257 W)\n"
    "• Causal detrending did not provide consistent evidence that long-term drift was\n"
    "  masking a Saola-specific anomaly contrast; the sign reverses with window length,\n"
    "  and the largest apparent contrast belongs to the arm whose sensor was faulty\n"
    "  throughout\n"
    "• Alert counts are inflated 22–57% by the year-level offset, most severely for the\n"
    "  Reference arm\n"
    "• Saola is a proxy event window; no per-timestep fault verification exists for this\n"
    "  site", "fig5 annotation")
for k, v in SLICE_LABELS.items():
    check(v, f"slice label {k}")
for k, v in SHORT.items():
    check(v, f"arm label {k}")
REF_FOOTNOTE = check(
    "Hatched: Reference arm's irradiance sensor was corrupted (Mar–Oct 2023) or "
    "absent during these periods.", "fig2 footnote")
INSAMPLE_ON_FIG = check(
    "IN-SAMPLE CAVEAT: 2021–22 residuals are training-period (in-sample) predictions; "
    "2023 residuals are held out. Part of the January step is prediction-error type, "
    "not physical change.", "fig3 in-sample caveat")

print(f"\nTERMINOLOGY ASSERTIONS: {len(_checked)} strings checked, all PASS")

# ── Written summary text, guarded before anything is written ────────────────
SUMMARY = f"""# Objective 2 — Same-Time Irradiance Estimation for Sensor-Free PV Anomaly Screening

## 1. What Objective 2 tested

Objective 2 asked whether a photovoltaic installation can be monitored without a
physical irradiance sensor, and if so which sensor-free representation of
irradiance performs best. Four arms were compared, each carrying exactly one
irradiance representation except the last, which carries two: a **Reference arm**
using the measured pyranometer signal, **Arm A** using pvlib clear-sky irradiance,
**Arm B** using a machine-learned same-time irradiance estimate, and **Arm C**
using clear-sky and the learned estimate together. Every other input — solar
geometry, ambient temperature and time encodings — is identical across arms, so
any difference between them is attributable to the irradiance representation
alone.

Two design decisions were taken on audit evidence and fixed before implementation.
First, the task is **same-time irradiance estimation**: an estimate of irradiance
at the current timestamp built only from information available without the sensor.
No future-prediction claim is made. Second, **no DC-to-AC conversion parameter** is
introduced. The measured `power(W)` is AC active power while PVWatts models DC
output, so rather than assume a fixed inverter efficiency the pipeline uses pvlib
for irradiance and solar geometry only and treats measured AC power as the
observed signal.

## 2. Method

**Sensor-free eligibility.** The historical pipeline decided which rows existed by
filtering on measured irradiance (20 to 1200 W/m²), which no sensor-free
deployment could reproduce. That rule was replaced by one computed from clear-sky
irradiance and solar geometry alone: `ghi_clear >= 50 W/m²` and
`solar_zenith < 85°`. The replacement is not merely a matter of principle. The
historical rule admitted 12,084 rows whose median solar zenith was 116.4° — the
sun well below the horizon — because the pyranometer carries a night floor
averaging 9.85 W/m². A geometry-based rule cannot make that error.

**Exogenous-only design.** No arm receives lagged PV power. Observed power at time
*t* is the prediction target and the residual-comparison signal, never a
predictor. The reason is direct: during a sustained fault, recent depressed power
would teach the model to expect depressed power, collapsing the residual and
allowing the fault to mask itself. Excluding it also prevents lagged power from
dominating the prediction and obscuring the differences between arms, which is
what Objective 2 exists to measure. Contemporaneous exogenous values — clear-sky
irradiance, solar geometry, ambient temperature and the same-time estimate — are
legitimate inputs at the target timestamp; only power is barred.

**Common evaluation population.** All four arms are trained and evaluated on an
identical set of **40,783 sequences**, enforced by an assertion comparing target
timestamps across arms. The Reference arm consumes the sensor as a feature but
does not receive its own row set. Sequences are built positionally on a gap-free
15-minute grid, with each window's timestamps asserted to be exactly 15 minutes
apart so that no artificial adjacency is created across removed rows.

**Training-only fitting.** Every learned quantity — the irradiance estimator, the
feature scaler, the residual scale and the alert threshold — is fitted on
2021-06 to 2022-12 and applied unchanged to 2023.

**Three slices.** Slice 1 (Jan–Feb and Nov–Dec 2023) is the fair head-to-head,
being the only period where the Reference arm's sensor is trustworthy; it contains
no known fault event, so it compares prediction accuracy rather than detection
ability. Slice 2 (full 2023) is the deployment-realistic evaluation, with the
Reference arm unreliable from March to October. Slice 3 (Saola proxy event window,
1–15 September 2023) is a 416-row case study.

## 3. Results

### Level 1 — irradiance estimation

| Estimate | MAE (W/m²) | RMSE (W/m²) | R² |
|---|---|---|---|
| ML same-time estimate | {EST_MAE:.1f} | {EST_RMSE:.1f} | {EST_R2:.3f} |
| pvlib clear-sky | {CS_MAE:.1f} | {CS_RMSE:.1f} | {CS_R2:.3f} |

Evaluated on clean-sensor months only. The learned estimate approximately halves
the error of the clear-sky model. The clear-sky representation explained
substantially less variation in measured irradiance than the ML estimate during
the clean-sensor evaluation period. Note that the downstream comparison was
`common + ghi_clear` against `common + irr_est`, where `common` already contains
solar zenith and azimuth, so it measures whether clear-sky adds information
beyond geometry rather than whether clear-sky is uninformative in general.

### Level 2 — expected AC-power prediction accuracy (nMAE %)

| Arm | Slice 1 | Slice 2 | Slice 3 |
|---|---|---|---|
| Reference (measured irradiance) | {nmae('Slice 1','Reference (measured irradiance)'):.1f} | {nmae('Slice 2','Reference (measured irradiance)'):.1f} | {nmae('Slice 3','Reference (measured irradiance)'):.1f} |
| Arm A (pvlib clear-sky) | {nmae('Slice 1','Arm A (pvlib only)'):.1f} | {nmae('Slice 2','Arm A (pvlib only)'):.1f} | {nmae('Slice 3','Arm A (pvlib only)'):.1f} |
| Arm B (ML same-time estimate) | {nmae('Slice 1','Arm B (ML estimate)'):.1f} | {nmae('Slice 2','Arm B (ML estimate)'):.1f} | {nmae('Slice 3','Arm B (ML estimate)'):.1f} |
| Arm C (clear-sky + ML estimate) | {nmae('Slice 1','Arm C (pvlib + ML estimate)'):.1f} | {nmae('Slice 2','Arm C (pvlib + ML estimate)'):.1f} | {nmae('Slice 3','Arm C (pvlib + ML estimate)'):.1f} |

On the fair slice a working sensor remains the most accurate input, but the margin
is narrow: the learned estimate trails it by roughly 1.9 percentage points while
requiring no irradiance instrument at inference. The gap between the learned
estimate and clear-sky alone is roughly three times larger than the gap between
the learned estimate and the physical sensor, which is the central quantitative
result of Objective 2.

Over the full test year the ordering inverts. The Reference arm becomes the
weakest of the four because its instrument was corrupted from March to October,
while every sensor-free arm remains near 30%. In this dataset and affected
period, the degraded measured-irradiance Reference arm performed worse than all
three no-sensor arms.

### Level 3 — anomaly screening

| Arm | Event-window detection rate | Alert rate outside the proxy window | 2023 alert slots |
|---|---|---|---|
| Reference (measured irradiance) | {lvl3.iloc[0].event_window_detection_rate_pct:.1f}% | {lvl3.iloc[0].alert_rate_outside_proxy_pct:.1f}% | {int(lvl3.iloc[0].total_2023_anomaly_slots):,} |
| Arm A (pvlib clear-sky) | {lvl3.iloc[1].event_window_detection_rate_pct:.1f}% | {lvl3.iloc[1].alert_rate_outside_proxy_pct:.1f}% | {int(lvl3.iloc[1].total_2023_anomaly_slots):,} |
| Arm B (ML same-time estimate) | {lvl3.iloc[2].event_window_detection_rate_pct:.1f}% | {lvl3.iloc[2].alert_rate_outside_proxy_pct:.1f}% | {int(lvl3.iloc[2].total_2023_anomaly_slots):,} |
| Arm C (clear-sky + ML estimate) | {lvl3.iloc[3].event_window_detection_rate_pct:.1f}% | {lvl3.iloc[3].alert_rate_outside_proxy_pct:.1f}% | {int(lvl3.iloc[3].total_2023_anomaly_slots):,} |

These rates are screening behaviour, not classification performance: the site has
no per-timestep fault verification, so any metric presupposing labels would imply
that every timestamp outside the proxy window is healthy, which is unknowable.

## 4. What the results support

Machine-learned same-time irradiance estimation substantially outperforms pvlib
clear-sky, both as an estimate of irradiance itself and as an input to downstream
expected-power prediction. This holds on every slice and by a consistent margin.

A sensor-free estimate sits close to a working sensor and well ahead of a
corrupted one. On the only period where the physical instrument can be trusted the
learned estimate is within about two percentage points of it; across the full test
year it is materially better, because the instrument failed and the estimate did
not.

Combining clear-sky with the ML estimate provides negligible additional benefit in
the primary comparison. The complementarity hypothesis behind Arm C was tested and
is not supported by Slices 1 and 2 (22.7 to 22.4%, and 30.0 to 30.1%). A plausible
explanation is that clear-sky irradiance is already one of the estimator's own
predictors, so its information is largely present in the estimate. Slice 3 shows a
larger difference but rests on 416 rows and is not used to claim complementarity
either way.

## 5. What the results do not support

Residual-based anomaly screening does not demonstrate sensitivity to the Saola
proxy event window. The window is statistically unremarkable against the eight
weeks either side of it, and the screening layer is best described as a **negative
/ inconclusive screening finding**.

Causal detrending did not provide consistent evidence that long-term drift was
masking a Saola-specific anomaly contrast. The sign of the contrast reverses
between a 30-day and a 90-day trailing window, and the largest apparent contrast
belongs to the arm whose sensor was faulty throughout the period.

The cause of the 2023 residual shift cannot be attributed from the saved outputs.
Neither gradual degradation nor post-event damage is supported as the sole
explanation by the observed temporal pattern — the training period is flat and no
step occurs in September — but the
January onset coincides exactly with the train/test boundary, so distribution
drift cannot be separated from ordinary in-sample versus held-out generalisation
error. The question is left open; Objective 2 does not require it to be settled.

## 6. Limitations

The plane of the measured irradiance signal is undocumented; for SQ1 this is inert
because the array is horizontal, but it would matter for any cross-station work.
The pairing of SQ1 to the single campus weather station is likewise undocumented,
and cloud fields decorrelate over campus-scale distances. The Saola window is a
calendar bracket around a storm rather than a verified per-timestep fault record.
The 2023 residual shift is confounded with the train/test boundary. Slice 1
contains no known fault event, so it measures prediction accuracy rather than
detection ability. Slice 3 rests on 416 rows. The Reference arm's sensor was
faulty across eight of the twelve test months, which limits how far any comparison
against it can be pushed.

## 7. Figures

**Figure 1 — Same-time irradiance estimation versus pvlib clear-sky.** Left, the
learned estimate and clear-sky irradiance each plotted against the measured signal
on clean-sensor months with a 1:1 reference line; right, daily mean irradiance
across 2023 for all three series, with the sensor-fault window and the Saola proxy
event window shaded. Evaluated on clean-sensor months only.

**Figure 2 — Expected AC-power prediction accuracy by irradiance representation.**
Normalised mean absolute error for each arm across the three slices. Reference-arm
bars for Slices 2 and 3 are hatched because its sensor was corrupted or absent in
those periods. Slice 1 contains no known fault event and therefore compares
prediction accuracy, not detection ability.

**Figure 3 — Monthly mean residual, 2021–2023.** Monthly mean residual per arm
with interquartile bands, the train/test boundary marked, and the sensor-fault and
Saola proxy windows shaded. The in-sample caveat is printed on the figure: the
January step coincides with the train/test boundary and is therefore partly
generalisation error rather than physical change.

**Figure 4 — Residual gap widens with irradiance, consistent with a proportional rather than fixed offset.** Mean residual by
clear-sky irradiance decile, 2022 against 2023, one panel per arm. The widening of
the gap with irradiance, rather than its vertical offset, is the informative
feature, since a uniform offset is expected from in-sample versus held-out
comparison alone.

**Figure 5 — Saola proxy event window shows no distinct residual signature.**
Left, mean residual z for the eight weeks before, the window itself, and the eight
weeks after; right, 2023 alert counts before and after removing the year-level
median offset. Saola is a proxy event window; no per-timestep fault verification
exists for this site.
"""
check(SUMMARY, "objective2_summary.md")
print(f"  summary document: {len(SUMMARY.splitlines())} lines checked, PASS")

import sys
if "--verify-only" in sys.argv:
    print("\nGUARDED STRINGS (all passed):")
    for w, t in _checked:
        first = t.splitlines()[0] if t.splitlines() else t
        extra = f"  (+{len(t.splitlines())-1} more lines)" if len(t.splitlines()) > 1 else ""
        print(f"  [{w}] {first[:120]}{extra}")
    print("\nVERIFY-ONLY MODE: assertions complete, no files written.")
    print(f"  output directory exists already: {OUT_DIR.exists()}")
    for f in ["fig1_irradiance_estimation", "fig2_power_accuracy_by_arm",
              "fig3_residual_trend", "fig4_residual_by_irradiance",
              "fig5_saola_window"]:
        print(f"    would write: {f}.png / .pdf")
    print("    would write: objective2_summary.md")
    print("    would write: figure_source_manifest.txt")
    sys.exit(0)

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = []


TARGETS = set()
for _a in sys.argv[1:]:
    if _a.startswith("--targets="):
        TARGETS = {t.strip() for t in _a.split("=", 1)[1].split(",")}


def save(fig, name, sources):
    if TARGETS and name not in TARGETS:
        plt.close(fig)
        MANIFEST.append((name, sources))
        print(f"  skipped {name} (not in --targets)")
        return
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    MANIFEST.append((name, sources))
    print(f"  saved {name}.png / .pdf")


def abox(ax, text, y=-0.30, size=8):
    ax.annotate(text, xy=(0.0, y), xycoords="axes fraction", fontsize=size,
                va="top", ha="left",
                bbox=dict(fc="#f7f7f7", ec="grey", boxstyle="round,pad=0.5"))


print("\nGENERATING FIGURES")
# ── Figure 1 ────────────────────────────────────────────────────────────────
g23 = grid[(grid.index.year == 2023) & grid["eligible"].astype(bool)]
gc = g23[g23.index.month.isin(CLEAN_MONTHS) & g23["irr_meas"].notna()
         & (g23["irr_meas"] <= 1200)]
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].scatter(gc["irr_meas"].to_numpy(), gc["ghi_clear"].to_numpy(), s=3,
              alpha=0.25, color="darkorange",
              label=f"pvlib clear-sky  MAE {CS_MAE:.0f}, RMSE {CS_RMSE:.0f}, R² {CS_R2:.3f}")
ax[0].scatter(gc["irr_meas"].to_numpy(), gc["irr_est"].to_numpy(), s=3,
              alpha=0.25, color="seagreen",
              label=f"ML same-time estimate  MAE {EST_MAE:.0f}, RMSE {EST_RMSE:.0f}, R² {EST_R2:.3f}")
lim = float(np.nanmax([gc["irr_meas"].max(), gc["ghi_clear"].max()])) * 1.02
ax[0].plot([0, lim], [0, lim], "r--", lw=1, label="1:1")
ax[0].set_xlabel("Measured irradiance (W/m²)")
ax[0].set_ylabel("Estimated irradiance (W/m²)")
ax[0].set_title("Clean-sensor months (Jan, Feb, Nov, Dec 2023)", fontsize=11)
ax[0].legend(markerscale=4, loc="upper left")
ax[0].grid(alpha=0.3)
d = g23[["irr_meas", "ghi_clear", "irr_est"]].resample("D").mean()
x = d.index.to_numpy()
ax[1].plot(x, d["irr_meas"].to_numpy(), color="black", lw=1, label="Measured")
ax[1].plot(x, d["ghi_clear"].to_numpy(), color="darkorange", lw=1, ls="--",
           label="pvlib clear-sky")
ax[1].plot(x, d["irr_est"].to_numpy(), color="seagreen", lw=1,
           label="ML same-time estimate")
ax[1].axvspan(FAULT0, FAULT1, color="gold", alpha=0.20)
ax[1].text(pd.Timestamp("2023-06-15", tz=TZ), ax[1].get_ylim()[1] * 0.06,
           "sensor-fault window (Mar–Oct 2023)", ha="center", fontsize=8,
           bbox=dict(fc="white", ec="none", alpha=0.75))
ax[1].axvspan(P0, P1, color="red", alpha=0.25)
ax[1].set_ylabel("Daily mean irradiance (W/m²)")
ax[1].set_title("Daily mean irradiance across 2023", fontsize=11)
ax[1].legend(loc="upper right")
ax[1].grid(alpha=0.3)
ax[1].tick_params(axis="x", rotation=30)
fig.suptitle(T1, fontsize=12)
abox(ax[0], A1, y=-0.16)
save(fig, "fig1_irradiance_estimation",
     "objective2_grid.csv (Fig 1 only); irradiance_estimator_metrics.csv")

# ── Figure 2 ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
slices = ["Slice 1", "Slice 2", "Slice 3"]
xs = np.arange(len(slices))
w = 0.2
for i, arm in enumerate(ARMS):
    vals = [nmae(s, arm) for s in slices]
    hatch = ["", "///", "///"] if arm.startswith("Reference") else ["", "", ""]
    for j, (xx, v, h) in enumerate(zip(xs + i * w, vals, hatch)):
        ax.bar(xx, v, width=w, color=COLORS[arm], alpha=0.9, hatch=h,
               edgecolor="black", linewidth=0.6,
               label=SHORT[arm].replace("\n", " ") if j == 0 else None)
        ax.text(xx, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(xs + w * 1.5)
ax.set_xticklabels([SLICE_LABELS[s] for s in slices], fontsize=9)
ax.set_ylabel("nMAE (%)")
ax.set_title(T2)
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0, max(nmae(s, a) for s in slices for a in ARMS) * 1.25)
abox(ax, A2 + "\n\n" + REF_FOOTNOTE, y=-0.16)
save(fig, "fig2_power_accuracy_by_arm", "expected_power_accuracy.csv")

# ── Figure 3 ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 7))
for arm in ARMS:
    m = mon[mon.arm == arm].copy()
    xm = pd.PeriodIndex(m["month"], freq="M").to_timestamp().to_numpy()
    ax.plot(xm, m["mean_W"].to_numpy(), marker="o", ms=3, color=COLORS[arm],
            label=SHORT[arm].replace("\n", " "))
    ax.fill_between(xm, m["q25_W"].to_numpy(), m["q75_W"].to_numpy(),
                    color=COLORS[arm], alpha=0.10)
ax.axvline(pd.Timestamp("2023-01-01"), color="black", ls="--", lw=1.4)
ax.text(pd.Timestamp("2023-01-05"), ax.get_ylim()[0] * 0.95,
        "train/test boundary\n(Jan 2023)", fontsize=8, va="bottom")
ax.axvspan(pd.Timestamp("2023-03-01"), pd.Timestamp("2023-10-31"),
           color="gold", alpha=0.18)
ax.text(pd.Timestamp("2023-06-15"), ax.get_ylim()[1] * 0.85,
        "sensor-fault window\n(Mar–Oct 2023)", ha="center", fontsize=8)
ax.axvspan(pd.Timestamp("2023-09-01"), pd.Timestamp("2023-09-15"),
           color="red", alpha=0.30)
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylabel("Monthly mean residual (W); bands = IQR")
ax.set_title(T3)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
ax.annotate(INSAMPLE_ON_FIG, xy=(0.02, 0.05), xycoords="axes fraction",
            fontsize=8, va="bottom",
            bbox=dict(fc="lightyellow", ec="darkgoldenrod", boxstyle="round,pad=0.4"))
abox(ax, A3, y=-0.14)
save(fig, "fig3_residual_trend", "residual_diagnostic/residual_monthly_stats.csv")

# ── Figure 4 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True)
for axx, arm in zip(axes.ravel(), ARMS):
    b = bins[bins.arm == arm]
    for y, ls, al, lab in [(2022, "-", 0.55, "2022 (in-sample)"),
                           (2023, "--", 1.0, "2023 (held out)")]:
        s = b[b.year == y].sort_values("ghi_clear_decile")
        axx.plot(s["ghi_clear_decile"].to_numpy(), s["mean_resid_W"].to_numpy(),
                 ls, marker="o", color=COLORS[arm], alpha=al, label=lab)
    axx.axhline(0, color="grey", lw=0.8)
    axx.set_title(SHORT[arm].replace("\n", " "), fontsize=11)
    axx.set_xlabel("clear-sky GHI decile")
    axx.set_ylabel("Mean residual (W)")
    axx.legend(fontsize=8)
    axx.grid(alpha=0.3)
import textwrap
fig.suptitle(T4, fontsize=15, y=0.975)
fig.tight_layout(rect=[0.03, 0.13, 0.99, 0.955])
fig.text(0.5, 0.065, textwrap.fill(A4, width=104), ha="center", va="center",
         fontsize=11.5,
         bbox=dict(fc="#f7f7f7", ec="grey", boxstyle="round,pad=0.8"))
save(fig, "fig4_residual_by_irradiance",
     "residual_diagnostic/residual_by_irradiance_bin.csv")

# ── Figure 5 ────────────────────────────────────────────────────────────────
pre0, post1 = P0 - pd.Timedelta(weeks=8), P1 + pd.Timedelta(weeks=8)
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
xs = np.arange(3)
for i, arm in enumerate(ARMS):
    sig = float(trainsum.loc[arm, "resid_sigma_train_W"])
    r = R[arm]
    z = r["resid"] / sig
    vals = [z[(r.index >= pre0) & (r.index < P0)].mean(),
            z[(r.index >= P0) & (r.index <= P1)].mean(),
            z[(r.index > P1) & (r.index <= post1)].mean()]
    ax[0].bar(xs + i * 0.2, vals, width=0.2, color=COLORS[arm],
              edgecolor="black", linewidth=0.6,
              label=SHORT[arm].replace("\n", " "))
ax[0].set_xticks(xs + 0.3)
ax[0].set_xticklabels(["8 weeks before", "Saola proxy\nevent window",
                       "8 weeks after"])
ax[0].set_ylabel("Mean residual z (training-period scale)")
ax[0].set_title("No step at the proxy event window", fontsize=11)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3, axis="y")
ax[0].axhline(0, color="grey", lw=0.8)
xo = np.arange(len(offs))
ax[1].bar(xo - 0.2, offs["alert_slots_2023"].to_numpy(), width=0.4,
          color=[COLORS[a] for a in offs["arm"]], edgecolor="black",
          linewidth=0.6, label="raw 2023 alert slots")
ax[1].bar(xo + 0.2, offs["alert_slots_after_median_removed"].to_numpy(),
          width=0.4, color=[COLORS[a] for a in offs["arm"]], alpha=0.45,
          hatch="//", edgecolor="black", linewidth=0.6,
          label="after removing the 2023 median offset")
for i, r in offs.iterrows():
    ax[1].text(i, r.alert_slots_2023 * 1.03,
               f"{r.pct_attributable_to_offset:.0f}% offset", ha="center",
               fontsize=8)
ax[1].set_xticks(xo)
ax[1].set_xticklabels([SHORT[a].replace(" same-time", "\nsame-time")
                       .replace("(clear-sky + ML estimate)", "(clear-sky\n+ ML est.)")
                       for a in offs["arm"]], fontsize=7)
ax[1].set_ylabel("2023 alert slots")
ax[1].set_title("Alert counts inflated by the year-level offset", fontsize=11)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis="y")
fig.suptitle(T5, fontsize=12)
abox(ax[0], A5, y=-0.18)
save(fig, "fig5_saola_window",
     "residuals_full_*.csv; arm_training_summary.csv; "
     "residual_diagnostic/threshold_offset_analysis.csv; detrended_diagnostic.csv")

if not TARGETS or "summary" in TARGETS:
    (OUT_DIR / "objective2_summary.md").write_text(SUMMARY)
    print("  saved objective2_summary.md")
else:
    print("  skipped objective2_summary.md (not in --targets)")
man = ["Objective 2 figure source manifest",
       f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
       "", "Input files read (all under outputs/objective2_modeling/):"]
man += [f"  {f}" for f in sorted(set(_read))]
man += ["", "Figure -> source:"]
for name, src in MANIFEST:
    man += [f"  {name}", f"      {src}"]
man += ["", "Forbidden directories (asserted not read): "
        + ", ".join(FORBIDDEN_DIRS)]
if not TARGETS or "manifest" in TARGETS:
    (OUT_DIR / "figure_source_manifest.txt").write_text("\n".join(man) + "\n")
    print("  saved figure_source_manifest.txt")
else:
    print("  skipped figure_source_manifest.txt (not in --targets)")

with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"plot_objective2_results.py | input: {len(set(_read))} permitted "
            f"files under objective2_modeling/ | output: 5 figures (PNG+PDF) + summary + "
            f"manifest = 12 files in objective2_figures/ | "
            f"notes: consolidation only, nothing trained or modified; "
            f"terminology assertions passed on {len(_checked)} strings\n")
print(f"\nSaved to: {OUT_DIR}")
