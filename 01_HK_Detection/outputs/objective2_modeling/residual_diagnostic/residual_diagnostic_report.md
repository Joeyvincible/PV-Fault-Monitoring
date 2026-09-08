# Objective 2 — Residual Trend Diagnostic

Generated: 2026-09-06T22:44:49

Read-only. Nothing retrained, no threshold re-fitted, no model or dataset modified.

> **Sept 1-15 2023 is a PROXY EVENT WINDOW, not fault ground truth.**
>
> **2021-22 residuals are IN-SAMPLE (training) predictions; 2023 residuals are OUT-OF-SAMPLE. Part of any difference between them is prediction-error type, not physical change.**

Loaded full-period residuals: **40,783** rows per arm, 2021-06-01 to 2023-12-31
- training-period rows: 25,076 (in-sample) | test-period rows: 15,707 (out-of-sample)

## Part 1 — Characterising the shift

### 1.1 Monthly residual, 2021-2023

Full table in `residual_monthly_stats.csv` (124 arm-months). Yearly condensation below.

### 1.2 Residual distribution by year

| arm | year | sample | n | mean | median | SD | p5 | p25 | p75 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| Reference (measured irradiance) | 2021 | in-sample | 9,331 | 192 | 213 | 2,237 | -3,379 | -354 | 798 | 3,635 |
| Reference (measured irradiance) | 2022 | in-sample | 15,783 | -115 | -41 | 1,886 | -2,894 | -540 | 405 | 2,435 |
| Reference (measured irradiance) | 2023 | OUT-of-sample | 15,669 | -3,716 | -2,635 | 3,932 | -11,665 | -5,258 | -1,143 | 169 |
| Arm A (pvlib only) | 2021 | in-sample | 9,331 | -2 | -12 | 3,272 | -5,708 | -1,404 | 1,552 | 5,407 |
| Arm A (pvlib only) | 2022 | in-sample | 15,783 | -114 | -62 | 2,666 | -4,700 | -1,197 | 1,150 | 4,035 |
| Arm A (pvlib only) | 2023 | OUT-of-sample | 15,669 | -1,477 | -892 | 3,927 | -8,867 | -3,025 | 511 | 4,196 |
| Arm B (ML estimate) | 2021 | in-sample | 9,331 | 236 | 71 | 3,167 | -4,933 | -1,165 | 1,643 | 5,822 |
| Arm B (ML estimate) | 2022 | in-sample | 15,783 | 126 | 113 | 2,169 | -3,199 | -654 | 986 | 3,412 |
| Arm B (ML estimate) | 2023 | OUT-of-sample | 15,669 | -1,198 | -679 | 3,512 | -7,759 | -2,542 | 472 | 3,761 |
| Arm C (pvlib + ML estimate) | 2021 | in-sample | 9,331 | 89 | 32 | 3,088 | -5,051 | -1,189 | 1,415 | 5,359 |
| Arm C (pvlib + ML estimate) | 2022 | in-sample | 15,783 | 47 | 42 | 2,159 | -3,336 | -714 | 956 | 3,268 |
| Arm C (pvlib + ML estimate) | 2023 | OUT-of-sample | 15,669 | -1,327 | -741 | 3,499 | -7,925 | -2,570 | 347 | 3,448 |

**2021-22 residuals are IN-SAMPLE (training) predictions; 2023 residuals are OUT-OF-SAMPLE. Part of any difference between them is prediction-error type, not physical change.**

### 1.3 Within-2023 breakdown

| arm | Jan-Feb (clean) | Mar-Oct (sensor fault) | Nov-Dec (clean) | Sept 1-15 proxy window |
|---|---|---|---|---|
| Reference (measured irradiance) | -866 (n=2,364) | -4,742 (n=10,924) | -1,837 (n=2,381) | -3,339 (n=416) |
| Arm A (pvlib only) | -50 (n=2,364) | -1,868 (n=10,924) | -1,101 (n=2,381) | -1,524 (n=416) |
| Arm B (ML estimate) | -601 (n=2,364) | -1,292 (n=10,924) | -1,355 (n=2,381) | -1,215 (n=416) |
| Arm C (pvlib + ML estimate) | -598 (n=2,364) | -1,488 (n=10,924) | -1,311 (n=2,381) | -986 (n=416) |

Reference-arm residuals during Mar-Oct are affected by its corrupted sensor input and should be read accordingly.

### 1.4 Pre-Saola / Saola / post-Saola (8 weeks either side)

| arm | pre (8 wk) | proxy window | post (8 wk) |
|---|---|---|---|
| Reference (measured irradiance) | -5,098 (n=2,623) | -3,339 (n=416) | -2,721 (n=2,347) |
| Arm A (pvlib only) | -1,358 (n=2,623) | -1,524 (n=416) | -1,175 (n=2,347) |
| Arm B (ML estimate) | -768 (n=2,623) | -1,215 (n=416) | -1,034 (n=2,347) |
| Arm C (pvlib + ML estimate) | -1,159 (n=2,623) | -986 (n=416) | -1,226 (n=2,347) |

### 1.5 Do the arms shift together?

Correlation of monthly mean residual between arms:

| | Reference | Arm A | Arm B | Arm C |
|---|---|---|---|---|
| Reference | 1.000 | 0.878 | 0.833 | 0.881 |
| Arm A | 0.878 | 1.000 | 0.862 | 0.888 |
| Arm B | 0.833 | 0.862 | 1.000 | 0.976 |
| Arm C | 0.881 | 0.888 | 0.976 | 1.000 |

Mean off-diagonal correlation: **0.886**. Arms moving together points to a system- or site-level cause rather than to any one arm's irradiance representation.

## Part 2 — Candidate causes

### 2.1 Internal trend WITHIN the training period (both ends in-sample)

| arm | slope (W/month) | total drift over 2021-06 to 2022-12 | direction |
|---|---|---|---|
| Reference (measured irradiance) | -33.8 | -574 W | downward |
| Arm A (pvlib only) | -4.5 | -77 W | downward |
| Arm B (ML estimate) | -16.1 | -273 W | downward |
| Arm C (pvlib + ML estimate) | -14.0 | -238 W | downward |

This is the cleanest available trend test: both ends are in-sample, so the in-sample/out-of-sample confound applies equally across it.

### 2.2 Proportional or absolute?

| arm | 2023 mean residual (W) | 2023 mean residual / mean predicted (%) | training mean residual/pred (%) |
|---|---|---|---|
| Reference (measured irradiance) | -3,716 | -30.8% | -0.0% |
| Arm A (pvlib only) | -1,477 | -15.0% | -0.8% |
| Arm B (ML estimate) | -1,198 | -12.5% | +1.8% |
| Arm C (pvlib + ML estimate) | -1,327 | -13.7% | +0.7% |

### 2.3 Residual by clear-sky irradiance decile, 2022 vs 2023

**2021-22 residuals are IN-SAMPLE (training) predictions; 2023 residuals are OUT-OF-SAMPLE. Part of any difference between them is prediction-error type, not physical change.** 2022 is in-sample and 2023 out-of-sample, so a uniform vertical offset between the two curves is expected even with no physical change. The informative feature is the SHAPE — whether the gap widens with irradiance (proportional, consistent with degradation) or stays flat (absolute offset).

| arm | decile 0-2 gap (W) | decile 7-9 gap (W) | widens with irradiance? |
|---|---|---|---|
| Reference (measured irradiance) | -1,575 | -6,015 | yes |
| Arm A (pvlib only) | -460 | -2,305 | yes |
| Arm B (ML estimate) | -646 | -1,838 | yes |
| Arm C (pvlib + ML estimate) | -655 | -1,953 | yes |

## Part 3 — Causal detrended residual (DIAGNOSTIC ONLY)

A trailing median over the preceding N days, using only timestamps strictly before the current one. This is **not leakage** — it consults no future information — but it changes what the detector means. It answers *is this unusual relative to recent behaviour* rather than *is this unusual relative to healthy operation*. A detector that continually re-centres on recent test-period behaviour can absorb sustained degradation as the new normal, the same failure mode that motivated removing lagged power.

**These numbers are diagnostic. They are not adopted as the primary detector and are not Objective 2 results.**

The same numeric training-fitted z threshold is reused; no threshold is re-fitted.

### Trailing window N = 7 days

| arm | mean z pre (8wk) | mean z proxy window | mean z post (8wk) | contrast vs pre | event-window rate | alert rate outside window |
|---|---|---|---|---|---|---|
| Reference (measured irradiance) | -0.373 | +0.073 | -0.219 | +0.446 | 1.0% | 2.1% |
| Arm A (pvlib only) | -0.161 | -0.246 | -0.224 | -0.085 | 2.6% | 1.4% |
| Arm B (ML estimate) | -0.172 | -0.337 | -0.241 | -0.165 | 3.6% | 1.3% |
| Arm C (pvlib + ML estimate) | -0.218 | -0.257 | -0.254 | -0.039 | 3.8% | 1.3% |

### Trailing window N = 30 days

| arm | mean z pre (8wk) | mean z proxy window | mean z post (8wk) | contrast vs pre | event-window rate | alert rate outside window |
|---|---|---|---|---|---|---|
| Reference (measured irradiance) | -0.341 | +0.382 | -0.015 | +0.723 | 0.0% | 2.9% |
| Arm A (pvlib only) | -0.175 | -0.121 | -0.227 | +0.054 | 1.4% | 1.7% |
| Arm B (ML estimate) | -0.206 | -0.200 | -0.235 | +0.006 | 3.4% | 1.5% |
| Arm C (pvlib + ML estimate) | -0.276 | -0.048 | -0.252 | +0.228 | 3.4% | 1.4% |

### Trailing window N = 90 days

| arm | mean z pre (8wk) | mean z proxy window | mean z post (8wk) | contrast vs pre | event-window rate | alert rate outside window |
|---|---|---|---|---|---|---|
| Reference (measured irradiance) | -0.217 | +0.476 | +0.417 | +0.693 | 0.0% | 4.1% |
| Arm A (pvlib only) | +0.015 | -0.190 | -0.161 | -0.204 | 1.4% | 2.0% |
| Arm B (ML estimate) | +0.049 | -0.309 | -0.249 | -0.358 | 3.6% | 1.6% |
| Arm C (pvlib + ML estimate) | -0.027 | -0.155 | -0.266 | -0.128 | 3.6% | 1.6% |

## Part 4 — Implication for Level 3 as currently reported

| arm | training-fitted z threshold | 2023 median z | mis-centring (median z, in σ) | 2023 alert slots | slots if 2023 median removed | attributable to year-level offset |
|---|---|---|---|---|---|---|
| Reference (measured irradiance) | -3.576 | -1.298 | -1.298σ | 1,469 | 626 | 57% |
| Arm A (pvlib only) | -3.030 | -0.307 | -0.307σ | 431 | 335 | 22% |
| Arm B (ML estimate) | -3.025 | -0.262 | -0.262σ | 350 | 277 | 21% |
| Arm C (pvlib + ML estimate) | -3.112 | -0.291 | -0.291σ | 342 | 260 | 24% |

The decomposition subtracts the 2023 median z before applying the same unchanged threshold. It is a diagnostic attribution, not a re-fit.

## Figures

- saved `residual_monthly_series.png` / `.pdf`
- saved `residual_by_year_distribution.png` / `.pdf`
- saved `residual_by_irradiance_bin.png` / `.pdf`
- saved `residual_detrended_saola.png` / `.pdf`
