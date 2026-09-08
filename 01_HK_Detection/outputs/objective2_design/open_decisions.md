# Objective 3 — Open Decisions Requiring Approval

Six decisions from Part 4 of the design specification, plus one raised by
correction **[C3]**. Each states the options, the trade-off, and a
recommendation. Nothing here is implemented.

---

## 1. Which non-irradiance meteorological variables to admit?

Available and unused: Rainfall, Relative Humidity, Sea Level Pressure,
Visibility, Wind (speed and direction). Ambient temperature is already used.

**The research-question drift concern, addressed directly.** Admitting a
broad set of extra sensors does change what is being asked. "Can pvlib
replace the irradiance sensor?" and "can a suite of other sensors compensate
for its absence?" are different questions with different engineering
implications — the second implies a comparably instrumented site, which
undercuts the cost argument that motivates sensor removal.

**Recommendation — split the question rather than blur it:**

| Arm | Environmental inputs | Question it answers |
|---|---|---|
| **Arm A** | ambient temperature **only** | Can physics plus power alone replace the sensor? |
| **Arm B / C estimator** | temperature, humidity, pressure, wind, rainfall | Can a learned estimate from commonly available weather replace the sensor? |

Ambient temperature is retained in Arm A because it is near-universal, is
already in the historical baseline, and is not a proxy for irradiance. This
keeps Arm A a clean test of the original question while letting Arm B be what
it is meant to be — an ML estimate from weather context.

**Visibility: recommend EXCLUDE by default.** It is an optical atmospheric
transparency measurement, which is closer to the estimated quantity than any
other candidate. Including it risks the estimator partly reading a
transparency instrument rather than inferring irradiance from weather state.
Retain it as a labelled sensitivity test only, reported separately.

**Approval needed on:** whether Arm A stays temperature-only, and whether
Visibility is excluded by default.

---

## 2. What model estimates irradiance in Arm B?

| Option | For | Against |
|---|---|---|
| GradientBoostingRegressor (historical choice) | precedent | slow; superseded |
| **HistGradientBoostingRegressor** | fast, handles non-linearity, needs no scaling, native NaN handling | less interpretable than linear |
| Linear / physics-corrected | interpretable | cannot capture cloud non-linearity |
| Neural network | flexible | more tuning, more variance, no clear gain at this size |

**Recommendation: `HistGradientBoostingRegressor`.** It was the strongest
tabular learner in Objective 1, trains in tens of seconds at this scale, and
tolerates missing meteorological values without a separate imputation step —
relevant given 252 missing-sensor rows and gaps in the met streams.

**Training bound.** Fitted on eligible rows with timestamps **≤ 2022-12-31**
only. The test period is never seen. Target: measured irradiance. Predictors:
per decision 1, and **excluding `power(W)` at time *t* per [C2]**.

**Approval needed on:** model choice, and confirmation that the estimator's
predictor set excludes power entirely.

---

## 3. What is the downstream detection model?

With `p_exp` removed there is no physics-derived expected power. The
replacement is a learned expectation.

**Recommendation.** An LSTM predicting AC power at *t*, with the window
layout fixed by **[C3]**:

- **inputs:** lagged power and lagged exogenous variables over *t−16 … t−1*
  (4 hours at 15-minute resolution), plus contemporaneous *non-power*
  exogenous variables at *t* — clear-sky GHI, solar zenith and azimuth,
  ambient temperature, and for Arms B and C the irradiance estimate at *t*
- **target:** `power(t)`, which never appears in any input
- **architecture:** retain the historical 2-layer LSTM so the comparison with
  prior work is meaningful; this is a pipeline-correctness study, not an
  architecture search

**Approval needed on:** retaining the LSTM rather than substituting a
simpler learner.

---

## 4. How is the fault signal computed without `ratio_cal`?

**Recommendation.** Work directly on the prediction residual:

```
residual(t)  = power_actual(t) − power_predicted(t)
z(t)         = residual(t) / σ_train        (σ from training-period residuals)
fault_flag(t) = 1  iff  z(t) < −k  for  ≥ m consecutive eligible slots
```

Both `σ_train` and the threshold `k` are set from the **training-period
residual distribution only** — for example `k` at the training 1st percentile
— and applied unchanged to 2023. The persistence requirement `m` carries over
the historical logic (4 consecutive slots = 1 hour) and suppresses
single-slot noise.

Normalising by a training-period scale rather than by predicted power avoids
the instability of a ratio when the denominator approaches zero at low sun,
which was a weakness of `ratio_cal`.

**Approval needed on:** the residual-z formulation and on setting `k` from a
training quantile rather than a fixed value.

---

## 5. Train/test split, and the sensor fault window **[C4]**

**Factual correction applied.** The earlier statement that the fault spans
March–October 2023 "and therefore the entire test period" is **wrong**.
Verified from monthly maxima:

| 2023 month | max measured irradiance | sensor fault |
|---|---|---|
| Jan | 685.8 | no |
| Feb | 1196.7 | no |
| Mar–Oct | 1208.7 – 1529.1 | **yes** |
| Nov | 972.7 | no |
| Dec | 987.4 | no |

The fault covers **8 of 12 months** — a substantial portion of the test
period, **not** all of it. January–February and November–December 2023
constitute a **clean-sensor subset**: 11,520 grid rows, of which **4,749**
survive the common eligibility mask, against 11,126 eligible rows in the
faulty window.

**Recommendation: confirm the split, and report the clean-sensor subset as a
separate evaluation slice.** Train 2021-06 → 2022-12, test 2023.

The clean subset should be reported because it is **the only window in which
the Reference arm is trustworthy**. Across the full test year the Reference
arm is handicapped by its own instrument, so a no-sensor arm beating it there
proves little. On the clean subset the comparison is fair, and a no-sensor
arm performing comparably there is the meaningful result.

Two caveats to state alongside it: the clean subset is winter-weighted
(Jan, Feb, Nov, Dec), so it is not seasonally representative; and it excludes
the Typhoon Saola window entirely, so the fault-detection evaluation and the
fair-Reference evaluation cannot be performed on the same rows.

**Approval needed on:** the split, and on reporting three slices — full 2023,
fault window, clean-sensor subset.

---

## 6. Evaluation protocol

The Typhoon Saola window (Sept 1–15 2023) is the only approximate ground
truth, and it is approximate: the window is a calendar bracket around a storm,
not a per-timestep fault label.

**On the historical precision figures (~0.04).** They should **not** be
reported as a headline. With 618 eligible rows in the typhoon window against
15,875 eligible rows in 2023, the positive base rate is about 3.9%, so a
detector flagging generously is arithmetically forced to low precision. A
precision of 0.04 at that base rate is close to what random flagging would
give and says almost nothing about detector quality.

**Recommendation — report instead:**

- **recall on the typhoon window**, with the caveat that the window is a
  calendar bracket
- **false-alarm rate on a designated quiet period** — eligible 2023 rows
  outside the typhoon window and outside the fault window — which is the
  operationally meaningful cost
- **flagged hours per month across 2023**, as a time series, so the reader
  sees where alarms fall rather than a single collapsed number
- **the residual series itself** around the typhoon, which is the most honest
  evidence and does not depend on a threshold
- precision **only** with the base rate printed beside it

Uncertainty: report per-arm results with the three slices from decision 5, and
state that no confidence interval is meaningful over a single storm event.

**Approval needed on:** dropping precision as a headline metric in favour of
recall, false-alarm rate, and the residual series.

---

## 7. Should lagged power be an input at all? *(raised by [C3])*

**[C3]** permits lagged power provided it strictly precedes the target. But a
fault persisting beyond the 4-hour input window would be partly absorbed into
the lagged history: the model learns that low recent power predicts low
current power, and the residual shrinks even though the fault continues. The
Typhoon Saola damage was sustained for days, which is exactly this case.

| Option | Effect |
|---|---|
| Keep lagged power (historical behaviour) | Best short-horizon accuracy; risks attenuating sustained faults |
| Exogenous-only (no power in inputs at all) | Residual reflects the full deviation from environmental expectation; less accurate under normal operation |
| Run both as a labelled variant | Quantifies the attenuation directly |

**Recommendation: run both, as a labelled variant on Arm A only.** It costs
one extra training run and directly measures how much sustained-fault signal
the lagged-power channel absorbs. If the difference is small, keep lagged
power for all arms; if large, the exogenous-only formulation is the
defensible choice for fault detection.

**Approval needed on:** whether to run this variant.

---

## Summary of approvals sought

1. Arm A temperature-only; Visibility excluded by default
2. `HistGradientBoostingRegressor` estimator, no power in its predictors
3. LSTM retained as the downstream model
4. Residual-z fault signal with a training-quantile threshold
5. Split confirmed; three reporting slices including the clean-sensor subset
6. Precision demoted; recall, false-alarm rate and residual series reported
7. Lagged-power vs exogenous-only variant on Arm A
