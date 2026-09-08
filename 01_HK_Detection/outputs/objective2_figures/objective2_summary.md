# Objective 2 — Same-Time Irradiance Estimation for Sensor-Free PV Anomaly Screening

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
| ML same-time estimate | 109.3 | 152.1 | 0.735 |
| pvlib clear-sky | 214.1 | 275.0 | 0.132 |

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
| Reference (measured irradiance) | 20.8 | 46.0 | 61.2 |
| Arm A (pvlib clear-sky) | 28.4 | 34.2 | 41.8 |
| Arm B (ML same-time estimate) | 22.9 | 29.7 | 40.5 |
| Arm C (clear-sky + ML estimate) | 22.2 | 29.6 | 38.0 |

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
| Reference (measured irradiance) | 4.6% | 9.5% | 1,469 |
| Arm A (pvlib clear-sky) | 2.6% | 2.8% | 431 |
| Arm B (ML same-time estimate) | 3.6% | 2.2% | 350 |
| Arm C (clear-sky + ML estimate) | 3.6% | 2.1% | 342 |

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
