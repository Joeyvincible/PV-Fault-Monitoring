# Objective 3 — Cross-Domain Transfer: Summary

Generated: 2026-09-04T16:36:52
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
| **3. Source performance** | Evidenced from frozen Objective 1 outputs: five-class Macro F1 0.6840, collapsed anomalous F1 0.8268, balanced accuracy 0.8749. |

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
reaches Macro F1 0.6840 across five classes. Collapsed to Normal versus
non-Normal it reaches anomalous F1 0.8268 and balanced accuracy
0.8749 (TN 350,584 / FP 52,260 / FN 25,422 / TP 185,445).

**HK diagnostic class distribution**, clean-sensor primary population,
denominator 4,745 feature-complete rows:

| Class | Predicted | Brazil empirical | Brazil balanced training |
|---|---|---|---|
| Normal | 1,453 (30.62%) | 65.64% | 47.20% |
| Short-Circuit | 6 (0.13%) | 0.98% | 1.50% |
| Degradation | 162 (3.41%) | 1.69% | 2.60% |
| Open-Circuit | 6 (0.13%) | 0.98% | 1.51% |
| Shadowing | 3,118 (65.71%) | 30.71% | 47.20% |

Neither Brazil column is an expected HK prior. Output is highly concentrated on
Shadowing, and Short-Circuit and Open-Circuit are each predicted for only 6 rows.

**Per-fold stability.** Individual fold screening rates span
67.36-71.70%
against an ensemble rate of 69.38%, with unanimous binary agreement
of 87.84% and pairwise agreement of
92.16-95.43%. The concentration is not the product of
one aberrant fold.

**Uncalibrated ensemble anomalous score.** Median 0.9596, p5 0.0379,
p25 0.3060, p75 0.9997, p95 1.0000. The distribution is
strongly bimodal, with dense mass at both extremes. It is used for ranking and
shape only, is not an estimated probability that the HK system is faulty, and no
cut-off was applied to it — the binary flag follows solely from the five-class
argmax.

**Temporal structure.** 269 runs of consecutive screened flags, mean
12.22 intervals, median 8, maximum 39
(approximately 9.8 hours), with 49
singletons. Runs are defined by exact 15-minute timestamp continuity, breaking at
any gap and at the February-November discontinuity. Flags form long coherent
daytime blocks.

**Populations.** Every rate below is stated against its own denominator of
feature-complete rows.

| Population | Role | Eligible | Feature-complete | Excluded | Screening rate |
|---|---|---|---|---|---|
| clean-sensor 2023 (Jan/Feb/Nov/Dec) | PRIMARY | 4,749 | 4,745 | 4 | 69.38% |
| March-October 2023 | corrupted-input stress test | 11,126 | 10,939 | 187 | 69.47% |
| full 2023 | mixed-period stress summary (NOT a corruption contrast) | 15,875 | 15,684 | 191 | 69.44% |
| Saola proxy window (1-15 Sept 2023) | corrupted-input stress test | 662 | 475 | 187 | 52.63% |

Aggregate prevalence is nearly identical between the clean-sensor and
March-October populations. This is **not a matched counterfactual**: the periods
differ in time of year and operating conditions, not only in sensor integrity,
and identical aggregate rates may conceal different row-level predictions and
class distributions. No conclusion is drawn about the effect of irradiance
corruption in either direction.

**Support containment.** Per fold, HK convex-hull containment spans
93.07-95.68% and
occupied-bin containment 86.45-89.97%,
measured against each fold's own training data by the same descriptive methods
used for Gate 2. Roughly 10-14%
of HK rows fall outside occupied training cells.

**Cross-screening comparison.** On 4,745
overlapping rows, Objective 3 and Objective 2 Arm B agree on
31.15% of rows and Arm C on
30.90%. The two
signals differ markedly: Objective 3 screens about 69% of rows where the
Objective 2 arms flag under 1%.

**Capacity sensitivity.** Substituting the 5.28 kW nameplate for the 5.00 kW
rating changes the screening rate from
69.38% to 69.08%.
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
window retains 475
of 662 eligible
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
population (n = 4,745 feature-complete rows) against two Brazil reference
distributions, neither of which is an expected HK prior. Outputs are
classifier-screened diagnostics, not validated fault assignments.

**Figure 3 — Screening behaviour is stable across source folds.** Per-fold and
ensemble screening rates (n = 4,745) and the distribution of the
uncalibrated ensemble anomalous score. No cut-off is drawn on the score.

**Figure 4 — Screening rate by population, and HK containment within source
support.** Rates for all four populations with denominators printed, corrupted-
input stress tests hatched, alongside per-fold hull and occupied-bin containment
of HK within each fold's own training data.

**Figure 5 — Temporal structure of screened flags, and comparison with
sensor-free screening.** Run-length distribution of consecutive screened flags
under exact 15-minute timestamp continuity, and the Objective 3 versus
Objective 2 Arm B / Arm C contingency on
4,745 overlapping rows. The
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
