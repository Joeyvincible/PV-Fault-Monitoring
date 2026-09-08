# Objective 2 — Stage 2: four-arm evaluation

Generated: 2026-09-06T16:20:09

Grid: **90,624** rows, 2021-06-01 00:00:00+08:00 to 2023-12-31 23:45:00+08:00

## Feature sets — exogenous only [1]

| arm | features | count |
|---|---|---|
| Reference (measured irradiance) | zenith, azimuth, Temp (Degree Celsius), hour_sin, hour_cos, month_sin, month_cos, day_of_year, irr_meas | 9 |
| Arm A (pvlib only) | zenith, azimuth, Temp (Degree Celsius), hour_sin, hour_cos, month_sin, month_cos, day_of_year, ghi_clear | 9 |
| Arm B (ML estimate) | zenith, azimuth, Temp (Degree Celsius), hour_sin, hour_cos, month_sin, month_cos, day_of_year, irr_est | 9 |
| Arm C (pvlib + ML estimate) | zenith, azimuth, Temp (Degree Celsius), hour_sin, hour_cos, month_sin, month_cos, day_of_year, ghi_clear, irr_est | 10 |

**[1] assertion passed for all four arms — no power-derived column is an input anywhere.** `power_W` is the target only.

Current-time exogenous values are inputs by design [4]: `ghi_clear(t)`, solar geometry, ambient temperature and `irr_est(t)` all carry the target timestamp. Only `power(t)` is barred. Excluding `irr_est(t)` would defeat the objective, which is precisely to test whether a same-time estimate can replace the physical measurement.

## Sequence construction on the continuous grid [4]

- Grid uniformly spaced at 15 min: **True**
- Window: `t-15 … t` **inclusive of t** (exogenous only), target `power(t)`
- Device: cpu

## Training

- **Reference (measured irradiance)**: 40,783 sequences (train 25,076 / test 15,707); windows rejected for non-contiguous spacing: 0
  trained in 87s | training residual σ = 2,030 W | alert threshold z < -3.58 (training 1% quantile)
- **Arm A (pvlib only)**: 40,783 sequences (train 25,076 / test 15,707); windows rejected for non-contiguous spacing: 0
  trained in 164s | training residual σ = 2,908 W | alert threshold z < -3.03 (training 1% quantile)
- **Arm B (ML estimate)**: 40,783 sequences (train 25,076 / test 15,707); windows rejected for non-contiguous spacing: 0
  trained in 3736s | training residual σ = 2,587 W | alert threshold z < -3.02 (training 1% quantile)
- **Arm C (pvlib + ML estimate)**: 40,783 sequences (train 25,076 / test 15,707); windows rejected for non-contiguous spacing: 0
  trained in 2780s | training residual σ = 2,546 W | alert threshold z < -3.11 (training 1% quantile)

**[C1] assertion passed — all four arms share one identical evaluation population of 40,783 sequences**, so the head-to-head is like-for-like. The Reference arm consumes the sensor as a feature but does not get its own row set.

## Level 2 — expected-AC-power prediction accuracy [2]

### Slice 1 — Jan/Feb/Nov/Dec 2023 (clean sensor; NO known fault event; compares prediction accuracy, NOT fault-detection ability)

| arm | n | MAE (W) | RMSE (W) | nMAE (%) | mean residual (W) |
|---|---|---|---|---|---|
| Reference (measured irradiance) | 4,783 | 1,698 | 2,387 | 20.8 | -1,343 |
| Arm A (pvlib only) | 4,783 | 2,317 | 3,334 | 28.4 | -569 |
| Arm B (ML estimate) | 4,783 | 1,866 | 2,765 | 22.9 | -970 |
| Arm C (pvlib + ML estimate) | 4,783 | 1,811 | 2,722 | 22.2 | -942 |

### Slice 2 — full 2023 (no-sensor deployment evaluation; Reference arm UNRELIABLE Mar–Oct)

| arm | n | MAE (W) | RMSE (W) | nMAE (%) | mean residual (W) |
|---|---|---|---|---|---|
| Reference (measured irradiance) | 15,707 | 3,846 | 5,403 | 46.0 | -3,707 |
| Arm A (pvlib only) | 15,707 | 2,861 | 4,190 | 34.3 | -1,473 |
| Arm B (ML estimate) | 15,707 | 2,480 | 3,706 | 29.7 | -1,194 |
| Arm C (pvlib + ML estimate) | 15,707 | 2,470 | 3,738 | 29.6 | -1,322 |

### Slice 3 — Sept 1–15 2023 Saola proxy window (case study only; Reference arm's sensor faulty or absent throughout)

| arm | n | MAE (W) | RMSE (W) | nMAE (%) | mean residual (W) |
|---|---|---|---|---|---|
| Reference (measured irradiance) | 416 | 3,341 | 4,240 | 61.2 | -3,339 |
| Arm A (pvlib only) | 416 | 2,280 | 3,521 | 41.8 | -1,524 |
| Arm B (ML estimate) | 416 | 2,209 | 3,516 | 40.5 | -1,215 |
| Arm C (pvlib + ML estimate) | 416 | 2,071 | 3,392 | 37.9 | -986 |

## Level 3 — anomaly / potential-fault screening [2]

Reported as screening behaviour, not as labelled classification. The Saola window is a calendar proxy: timestamps outside it are **not** known to be fault-free, so metrics that presuppose labels are not computed.

| arm | anomalies in proxy window | proxy-window slots | event-window detection rate | alert rate outside proxy window (2023) | total 2023 anomaly slots |
|---|---|---|---|---|---|
| Reference (measured irradiance) | 19 | 416 | 4.6% | 9.5% | 1,469 |
| Arm A (pvlib only) | 11 | 416 | 2.6% | 2.7% | 431 |
| Arm B (ML estimate) | 15 | 416 | 3.6% | 2.2% | 350 |
| Arm C (pvlib + ML estimate) | 15 | 416 | 3.6% | 2.1% | 342 |

### Residual behaviour around the proxy window

| arm | mean z, Aug 2023 | mean z, proxy window | mean z, Oct 2023 |
|---|---|---|---|
| Reference (measured irradiance) | -2.536 | -1.645 | -1.186 |
| Arm A (pvlib only) | -0.508 | -0.524 | -0.497 |
| Arm B (ML estimate) | -0.323 | -0.470 | -0.395 |
| Arm C (pvlib + ML estimate) | -0.482 | -0.387 | -0.448 |

A more negative mean z inside the proxy window than either side indicates the arm's expected-power model saw output fall below environmental expectation during the storm period.
