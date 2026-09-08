# Objective 3 — Direct-Transfer Results

Implementation of a frozen design. Nothing was tuned, no target-domain
adaptation was applied, and no threshold was fitted on HK.

Generated: 2026-09-06T19:47:02

## Step 2 — source correctness checkpoint (blocking)

Pooled 5x5 out-of-fold counts identical to frozen: **True**
Collapsed 2x2 counts identical to frozen: **True**

Recomputed from those counts: Macro F1 **0.6938** (frozen
0.6938), anomalous F1 **0.8180** (frozen
0.8180), balanced accuracy **0.8701** (frozen
0.8701).

The five fold pipelines were persisted only after this checkpoint passed.

## Step 3 — populations and the target-side transformation

| Population | Role | Raw eligible | Feature-complete | Excluded |
|---|---|---|---|---|
| clean-sensor 2023 (Jan/Feb/Nov/Dec) | PRIMARY | 4,749 | 4,745 | 4 |
| March-October 2023 | corrupted-input stress test | 11,126 | 10,939 | 187 |
| full 2023 | mixed-period stress summary (NOT a corruption contrast) | 15,875 | 15,684 | 191 |
| Saola proxy window (1-15 Sept 2023) | corrupted-input stress test | 662 | 475 | 187 |

Every rate below states its denominator. HK power was converted into
Brazil-capacity-equivalent watts (x 0.18112005); irradiance
passed through unchanged. The identity

`(p_hk/C_HK - mu/C_BR)/(sd/C_BR) == (p_hk*(C_BR/C_HK) - mu)/sd`

holds to **6.661e-16** across all five folds.

## Steps 4-5 — transfer behaviour on the clean-sensor primary population

Classifier-screened non-Normal rate: **67.97%** of
4,745 feature-complete rows.

Dominant predicted class: **Shadowing** at 64.19%.

Per-fold screening rates: fold 1 59.85%, fold 2 70.07%, fold 3 68.41%, fold 4 68.35%, fold 5 69.95%
Unanimous binary agreement across all five folds: 83.31%.

`ensemble_anomalous_score` is **uncalibrated** — the source models were trained
on balanced downsampled folds and no probability calibration was fitted. It is
used for ranking and shape only and is not an estimated probability that the HK
system is faulty. Median 0.9226, p5-p95
0.0248-1.0000. No threshold is applied to it anywhere; the binary flag
follows solely from the five-class ensemble argmax.

Temporal clustering, with runs defined by exact 15-minute timestamp continuity
(never dataframe adjacency): 262 runs, mean
12.30 intervals, max
39, 43 singletons.

## Framing

On HK, *anomalous* means **classifier-screened non-Normal only**. It is not a
verified anomaly or fault. The clean-sensor subset has no known fault event and
equally no per-timestamp Normal ground truth, so the screening rate is a
**transfer-behaviour statistic**; an elevated rate may primarily reflect
cross-domain mismatch rather than HK condition.

Agreement with Objective 2 sensor-free Arm B / Arm C screening is descriptive
only and is not independent fault validation. Arms B/C do not consume the
corrupted measured-irradiance signal used by Objective 3, but both analyses
concern the same HK site and period and neither has verified HK fault labels;
therefore agreement cannot be presented as corroboration of a true PV event.

The five-fold arithmetic-mean ensemble is a **pre-specified deployment summary**
on top of the frozen configuration. Gate 3 validates the underlying
POWER_PLUS_IRR configuration and representation; the ensemble aggregation rule
itself was never evaluated and is not out-of-fold validated.

## Interpretation

If direct transfer performs poorly on HK, the result cannot be dismissed as a
source representation with no discriminatory value: the same two-feature
representation demonstrated substantial grouped source-domain screening
performance. Poor HK behaviour would therefore provide evidence of limited
cross-domain transfer under the documented semantic, operating-distribution and
sensing differences.

The observed behaviour is reported as consistent with limited cross-domain
transfer and plausibly influenced by the documented operating-distribution
offset, together with the semantic, sensing and extrapolation evidence. This
experiment cannot establish that the offset uniquely caused the predictions, and
no attempt was made to tune the result away.

## Result-report interpretation

### 1. Corrupted-period comparison

Aggregate screening prevalence was nearly unchanged across the two populations:
**67.97%** of 4,745 clean-sensor
feature-complete rows against **68.28%** of
10,939 March-October feature-complete rows.

**This is not a matched counterfactual comparison.** The two populations differ
in time of year and in operating conditions, not only in irradiance-sensor
integrity. Identical aggregate rates may also conceal different row-level
predictions and different class distributions. No conclusion is drawn about
whether irradiance corruption affects the predictions, in either direction.

### 2. Output concentration (descriptive)

Output on the clean-sensor primary population is **highly concentrated**:
Shadowing 64.19%, Normal 32.03%, Degradation 3.48%, Short-Circuit 0.17%, Open-Circuit 0.13%. Per-fold screening behaviour is similar throughout
(59.85-70.07%), with
83.31% unanimous binary agreement across all five fold models
and pairwise binary agreement of 86.83-
96.31%. **The concentration is not attributable
to a single unstable fold.** No collapse threshold is applied; none was
preregistered.

### 3. Saola proxy window

The 51.37%
screening rate applies to 475 feature-complete rows
out of 662 eligible, with
187 excluded for missing transfer inputs. The
retained subset may therefore be selective. It is recorded as a
**corrupted-input stress-test observation only**, and is evidence **neither for
nor against** successful PV-event detection.

### 4. Extrapolation

Most HK rows lie within broad source support: per-fold convex-hull containment
93.07-95.68% and
occupied-bin containment 86.45-
89.97%. Gross out-of-range extrapolation alone is
therefore unlikely to explain the very high screening rate. **Local and
conditional distribution shift remain possible and are not excluded by these
measures**, and approximately 10-
14% of HK rows fall outside occupied
training cells.

### 5. Missing-input breakdown (verified separately per feature)

Verified by direct tabulation, never inferred from equality of total exclusion
counts:

| Population | Eligible | irr_meas missing | power_W missing | Both | Excluded |
|---|---|---|---|---|---|
| clean-sensor 2023 (Jan/Feb/Nov/Dec) | 4,749 | 0 | 4 | 0 | 4 |
| March-October 2023 | 11,126 | 186 | 1 | 0 | 187 |
| full 2023 | 15,875 | 186 | 5 | 0 | 191 |
| Saola proxy window (1-15 Sept 2023) | 662 | 186 | 1 | 0 | 187 |

## Conclusion

The two-feature Brazil configuration demonstrated substantial grouped
source-domain Normal/non-Normal discrimination, but direct transfer to HK
produced a high, Shadowing-concentrated classifier-screening rate during the
clean-sensor/no-known-fault population. The pattern was stable across
source-fold models and occurred predominantly within broad source support. These
findings provide evidence of limited direct cross-domain transferability under
the documented semantic, sensing and operating-distribution differences. HK
fault-type predictions are diagnostic outputs only and are not validated
physical fault assignments.
