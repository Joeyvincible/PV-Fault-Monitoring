# Hong Kong Read-Only Audit — Objective 2

Generated: 2026-09-07T19:33:40

Read-only investigation. No script, dataset or output was modified and no model was trained. Every finding carries a classification of CONFIRMED, SUPPORTED ASSUMPTION or UNRESOLVED.

## Part 1 — `power(W)` semantics

`SQ1.csv` columns: ['Time', 'generation(kWh)', 'power(W)'] — 90,624 rows, 2021-06-01 00:00:00 to 2023-12-31 23:45:00

**Sampling interval (empirical, site level):**

| interval | count |
|---|---|
| 0 days 00:15:00 | 90,623 |

Modal spacing is **0 days 00:15:00**. The dataset README states PV generation was collected at 5-minute intervals; the site-level file is not at that resolution. The inverter-level file is checked below.

### 1.1 AC or DC?

`SQ1_Inverter.csv` exists. Columns:

```
Time, dcVoltage(V), totalActivePower(W), L1_acCurrent(A), L1_acFrequency(Hz), L1_acVoltage(V), L1_activePower(W), L1_reactivePower(W), L2_acCurrent(A), L2_acFrequency(Hz), L2_acVoltage(V), L2_activePower(W), L2_reactivePower(W), L3_acCurrent(A), L3_acFrequency(Hz), L3_acVoltage(V), L3_activePower(W), L3_reactivePower(W)
```

Across 4,855 inverter rows, `totalActivePower(W) − (L1+L2+L3 activePower)` has mean absolute value **0.0000 W** — the total is exactly the sum of the three AC phase active powers.

The inverter schema exposes AC quantities per phase (`acCurrent`, `acVoltage`, `acFrequency`, `activePower`, `reactivePower`) and only a single DC quantity, `dcVoltage(V)`. No DC current is recorded, so DC power is not derivable from this dataset at all.

**Site level vs inverter level, aligned on 15-minute means (40,750 overlapping intervals):** correlation 0.98290, mean absolute difference 726.9 W (8.36% of mean inverter power).

The site-level series is therefore the same physical quantity as the inverter's AC active power.

### 1.2 Instantaneous or aggregated? `power(W)` vs `generation(kWh)`

- `generation(kWh)` range 0.000 to 7.112; monotonic non-decreasing: **False**
- corr(generation, power) on daylight rows: **0.9883**
- corr(Δgeneration, power×0.25h/1000): **0.1832**

`generation(kWh)` tracks `power(W)` in level rather than accumulating, so it is an interval energy figure, not a cumulative meter reading. Consistent with `power(W)` being the mean power over the 15-minute interval and `generation(kWh)` the corresponding interval energy.

Median `generation(kWh) ÷ (power(W)×0.25/1000)` on daylight rows: **1.0306** (1.0 would mean exact interval-energy correspondence).

### 1.4 Empirical cross-check against the 27.6 kW rating

| statistic | value (W) | as % of 27.6 kW rating |
|---|---|---|
| 99th percentile | 26,726 | 96.8% |
| 99.9th percentile | 27,612 | 100.0% |
| maximum | 27,735 | 100.5% |

Rows within 2% of the observed maximum: **282** (0.647% of positive-power rows).

The maximum never exceeds the rating. The distribution does **not** show a dense hard ceiling (only 282 rows sit near the maximum), so there is no strong evidence of sustained inverter clipping; the observed peak simply falls below the AC rating.

### 1.5 Consequence: PVWatts models DC, the measurement is AC

`pvlib.pvsystem.pvwatts_dc` returns **DC** array output. The existing `p_exp` therefore compares modelled DC against measured AC, and the calibration factor α silently absorbs the inverter conversion efficiency together with every other systematic bias.

- **(a) Expected-power baseline.** α is not a pure irradiance-model correction. Any statement that 'pvlib under-predicts by 2.6%' conflates model bias with DC→AC conversion loss, which typically runs 2–5%. The quantities are not like-for-like without an explicit inverter model.
- **(b) Objective 3 transfer.** Brazil `p_dc_computed` = vdc1·idc1 + vdc2·idc2 is definitively **DC**. Mapping it onto HK AC power compares different physical quantities separated by inverter efficiency and clipping behaviour. Capacity normalisation does not fix this: it rescales magnitude, not the AC/DC distinction.

## Part 2 — Environmental semantics

### 2.2 SQ1 geometry — TTL metadata

```turtle
pvsystem:SQ1 a brick:PV_Generation_System ;
    brick:coordinates [ brick:latitude 22.33 ;
            brick:longitude 114.18 ] ;
    brick:hasLocation [ brick:value "Staff_Quarter_Tower_1"^^xsd:string ] ;
    brick:hasPart pvsystem:SQ1_Inverter ;
    ext:altitude [ brick:hasUnit unit:M ;
            brick:value 94.0 ] ;
    ext:connectionDate [ brick:value "2021-04-30T00:00:00"^^xsd:string ] ;
    ext:contractor [ brick:value "SolarEdge"^^xsd:string ] ;
    ext:ratedPowerOutput [ brick:hasUnit unit:KW ;
            brick:value 27.6 ] ;
    ext:azimuth [ brick:value "0deg"^^xsd:string ] ;
    ext:tiltAngle [ brick:value "0deg"^^xsd:string ] .
```

| property | TTL value | value used in the pipeline | agree? |
|---|---|---|---|
| tilt | 0deg | 0° | yes |
| azimuth | 0deg | 180° | **no** — see note |
| latitude | 22.33 | 22.3363 | approximately |
| longitude | 114.18 | 114.2634 | **no** |
| altitude | 94.0 m | 30 m | **no** |
| rated power | 27.6 kW | 27,606 W | yes |

Tilt = 0deg is confirmed. At zero tilt the azimuth is geometrically irrelevant, so the 180° used in the pipeline is harmless here, but it does not match the metadata value of 0deg.

Two genuine discrepancies: the TTL places SQ1 at longitude 114.18 and altitude 94.0 m, while the pipeline uses 114.2634 and 30 m. The pipeline's coordinates are the campus centroid quoted in the dataset README (22.3363°N, 114.2634°E), not SQ1's own entry. The TTL coordinates are coarse (2 decimal places, ~1 km) and several stations share identical values, so neither source is precise; the altitude gap of 94.0 m versus 30 m is the more material of the two for clear-sky modelling.

### 2.1 What plane does the measured irradiance represent?

> 1. Description: This dataset includes measured photovoltaic (PV) power generation data and on-site weather data collected from 60 grid-connected rooftop PV stations in Hong Kong over a three-year period (2021-2023). The PV power generation data was collected at 5-minute intervals. The meteorological data was collected at 1-minute intervals from an on-site weather station. The metadata was represented using Brick schema was developed, which simplifies the data comprehension and the development of smart analytics applications.

> 1. Data collection: For stations without panel level optimizers (comprising 23 stations, accounting for 38.3% of the total), the data was individually measured and transferred by the inverter. For stations equipped with panel level optimizers (comprising 37 stations, accounting for 61.7% of the total), the PV generation data was measured and transferred by both the inverter and the panel level optimizer. Meteorological data is collected from the weather station located on the eastern side of the campus. The station comprises a 10-meter-high automatic weather tower and an outdoor plinth area that housing 6 samplers that measure meteorological data at 1-minute intervals.

> 4. Environmental/experimental conditions: The data were collected from 60 grid-connected rooftop PV stations and 1 weather station from a subtropical university campus under real operating conditions.

The README describes a 10-metre automatic weather tower with six samplers and names the variable only as `Irradiance (W/m2)`. Neither the README nor the TTL states the sensor plane, tilt, or instrument type. A mast-mounted campus weather station conventionally measures **global horizontal** irradiance, but this is an inference from installation type, not a documented fact.

For SQ1 specifically the distinction is inert — the array is horizontal (tilt = 0°), so GHI and plane-of-array coincide. It matters for any cross-station comparison and for how the column should be described.

### 2.3 Weather station pairing

- The campus has **60 PV stations and one weather station** (README).
- The weather station appears in the Brick model: **False**. A search for weather-related entities in the TTL returns nothing, so the station is not represented in the metadata at all.
- There is therefore **no documented link** between SQ1 and the weather station. The README places the station 'on the eastern side of the campus'; SQ1 is Staff Quarter Tower 1. No distance is stated.

Pairing SQ1 with this station is an **assumption of spatial representativeness**, not a documented relationship. It is the only meteorological source available, so the assumption is unavoidable, but it should be stated rather than implied — cloud fields decorrelate over campus-scale distances, which bears directly on how well any irradiance estimate can track this particular array.

### 2.4 Timezone

| file | first timestamp | tz-aware? |
|---|---|---|
| raw SQ1.csv | `2021-06-01 00:00:00` | False |
| raw Irradiance_2021.csv | `2021/1/1 0:00` | False |
| sq1_weather_15min_cleaned.csv | (missing) | – |
| sq1_pvlib_features.csv | (missing) | – |
| sq1_calibrated_baseline.csv | (missing) | – |

Every file carries **naive** timestamps — no offset, no zone. The raw data is presumed to be Hong Kong local time (UTC+8), which is what the pipeline assumes when it constructs a `Location` with `tz='Asia/Hong_Kong'`. Hong Kong has observed no daylight saving since 1979, so a fixed +8 offset is unambiguous for 2021–2023 — but the assumption is undocumented.

Note that `pvlib_features_SQ1.py` calls `site.get_solarposition(df.index)` on a **naive** index. pvlib then treats those timestamps as being in the site timezone. If the raw data were actually UTC, every solar position in the pipeline would be wrong by eight hours.

### 2.5 Sampling intervals (empirical)

| source | modal interval | README claim |
|---|---|---|
| PV site level (SQ1.csv) | 0 days 00:15:00 | 5 minutes |
| PV inverter level (SQ1_Inverter.csv) | 0 days 00:05:00 | 5 minutes |
| Meteorological (Irradiance_2021.csv) | 0 days 00:01:00 | 1 minute |

Meteorological data matches its documentation at 0 days 00:01:00. The site-level PV series is at 0 days 00:15:00, not the 5 minutes the README claims; the inverter-level series is at the documented 5 minutes. **Where the documentation and the data disagree, the data is the more reliable source** — the README statement is a generalisation across 60 stations and two product levels.

`Data_preprocessing_SQ1.py` resamples both PV and meteorological streams to 15 minutes with `.mean()`, so the 1-minute meteorological data is averaged into 15-minute means, and the already-15-minute PV series is effectively passed through.

### 2.6 Meteorological variables available

| variable | folder | columns | used in the current pipeline? |
|---|---|---|---|
| Irradiance | 3 csv, 0 xlsx | `Irradiance (W/m2)` | **yes** |
| Rainfall | 2 csv, 1 xlsx | `Rainfall(mm)` | no — available, unused |
| Relative Humidity | 3 csv, 0 xlsx | `RH (%)` | no — available, unused |
| Sea Level Pressure | 3 csv, 0 xlsx | `SLP (hPa)` | no — available, unused |
| Temperature | 3 csv, 0 xlsx | `Temp (Degree Celsius)` | **yes** |
| Visibility | 3 csv, 0 xlsx | `Vis (km)` | no — available, unused |
| Wind | 3 csv, 0 xlsx | `Wind Speed (m/s)`, `Wind Direction (degree)` | no — available, unused |

**Available but unused: Rainfall, Relative Humidity, Sea Level Pressure, Visibility, Wind.** These are the candidate non-irradiance environmental inputs for Arm A and Arm B. Relative humidity, sea-level pressure, visibility and wind are all plausibly informative about cloud state, and none of them is an irradiance measurement — so none violates a no-irradiance-sensor contract.

One caution: `Visibility` is an atmospheric-transparency measurement. It is not irradiance and does not come from the pyranometer, so it is admissible under a no-irradiance-sensor rule, but it is closer to the quantity being estimated than, say, pressure. Whether to admit it is a contract decision, not a technical constraint.

## Part 3 — Measured-irradiance dependency trace

Governing question: *could measured irradiance influence whether a row exists, how a feature is calculated, how a model or constant is calibrated, or which sequence reaches the model?*

| column / stage | depends on measured irradiance | derivation path |
|---|---|---|
| `zenith` | **NO** | pvlib solar position from timestamp + site location only |
| `azimuth` | **NO** | pvlib solar position from timestamp + site location only |
| `ghi_clear` | **NO** | pvlib Ineichen clear-sky from timestamp + location + altitude; no measurement |
| `k_clear` | **YES** | measured Irradiance / ghi_clear — clearness index, numerator is the sensor |
| `dni_est` | **YES** | DISC decomposition with ghi = measured Irradiance |
| `dhi_est` | **YES** | measured Irradiance − dni_est·cos(zenith) |
| `poa_global` | **YES** | get_total_irradiance with ghi = measured Irradiance and dni/dhi derived from it |
| `temp_cell` | **YES** | Faiman model driven by poa_global, which is a function of measured Irradiance |
| `p_exp` | **YES** | pvwatts_dc(poa_global, temp_cell) — both inputs trace to measured Irradiance |
| `residual` | **YES** | power(W) − p_exp |
| `ratio` | **YES** | power(W) / p_exp |
| `p_exp_cal` | **YES** | p_exp × α, and α itself is fitted on rows selected by measured Irradiance |
| `residual_cal` | **YES** | power(W) − p_exp_cal |
| `ratio_cal` | **YES** | power(W) / p_exp_cal |
| `power(W)` | **NO** | measured AC active power; independent of the irradiance sensor |
| `Temp (Degree Celsius)` | **NO** | measured ambient temperature; separate sensor |
| `month_sin / month_cos` | **NO** | calendar encoding of the timestamp |
| `ROW EXISTENCE (all downstream data)` | **YES** | Data_preprocessing_SQ1.py drops rows with Irradiance < 20 or > 1200 W/m², so every row in every derived CSV survived a measured-irradiance test |
| `SEQUENCE RETENTION (both LSTM scenarios)` | **YES** | build_sequences skips any window whose target row has measured Irradiance < MIN_IRR, in the no-sensor scenario as well as the with-sensor one |
| `alpha (calibration constant)` | **YES** | OLS fit over rows selected by measured Irradiance >= 50, using p_exp which is itself irradiance-derived |

### The current 'no-sensor' LSTM feature set

| feature | contaminated? | why |
|---|---|---|
| `power(W)` | no | measured AC power, not the irradiance sensor |
| `ghi_clear` | no | clear-sky model from time and location only |
| `Temp (Degree Celsius)` | no | ambient temperature sensor, not irradiance |
| `zenith` | no | solar geometry from timestamp only |
| `temp_cell` | **YES** | **Faiman model driven by poa_global → measured irradiance** |
| `p_exp_cal` | **YES** | **p_exp × α; p_exp is irradiance-derived and α is fitted on irradiance-selected rows** |
| `month_sin` | no | calendar encoding |
| `month_cos` | no | calendar encoding |

**2 of the 8 features are directly contaminated: `temp_cell` and `p_exp_cal`.**

But the feature list understates the problem. Two pipeline-level dependencies contaminate the scenario regardless of which features are chosen:

1. **Row existence.** Every row in `sq1_calibrated_baseline.csv` survived the `20 ≤ Irradiance ≤ 1200` filter in preprocessing. A deployment without the sensor could not have constructed this row set.
2. **Sequence retention.** `build_sequences` tests measured irradiance against `MIN_IRR` for every candidate window and skips those below it — in the no-sensor scenario too. The daylight *mask* was switched to `ghi_clear`, but the sequence builder was not, so measured irradiance still decides which samples the no-sensor model ever sees.

The consequence is that the historical 'no-sensor' result is **not a sensor-free result**, and this holds even if `temp_cell` and `p_exp_cal` were removed from the feature list.

## Part 4 — Calibration and fitting leakage

| learned quantity | fitted on | spans the test period? |
|---|---|---|
| α (calibration factor) | all retained rows with measured Irradiance ≥ 50, 2021-06 to 2023-12 | **YES — leakage** |
| monthly α (diagnostic) | same, grouped by month | **YES — leakage** |
| PDC0 estimate | 99.5th percentile of `power(W)` over the whole file | **YES — leakage** |
| StandardScaler (LSTM) | `df.index <= TRAIN_END` (2022-12-31) | no — correct |
| Imputer | none used in the HK pipeline | – |

**Finding.** `pvlib_calibration_SQ1.py` line 41 selects `day = df[df[IRR] >= MIN_IRR]` across the entire record and line 102 fits `α = Σ(P_act·P_exp) / Σ(P_exp²)` on that selection. The 2023 test period is inside the fit. Every `p_exp_cal`, `residual_cal` and `ratio_cal` value in the test period therefore embeds a constant that saw the test period.

A second, subtler instance: `pvlib_features_SQ1.py` line 107 sets `pdc0_est` to the 99.5th percentile of measured power over the whole file, so the system-capacity constant feeding PVWatts is also fitted across train and test.

The LSTM `StandardScaler` is handled correctly — fitted on `df.index <= TRAIN_END` only. But it is fitted on columns that are themselves contaminated by α, so correct scaler discipline does not rescue the pipeline.

**Proposed rule for the final modelling pipeline:**

> Any learned calibration constant, scaler, imputer, or threshold must be fitted on training-period data only and applied unchanged to validation and test periods.

Applying this would mean: fit α on 2021–2022 only; derive the capacity constant from nameplate (27.6 kW, per the TTL) or from training-period data only; keep the existing scaler discipline.

## Part 5 — Same-time estimator scope

| source | horizon-related evidence |
|---|---|
| `hk_data_audit_report.md` | 15 mention(s); e.g. "The README describes a 10-metre automatic weather tower with six samplers and names the variable only as `Irradiance (W/m2)`. Neither the README nor t" |
| `information_contracts_draft.md` | 1 mention(s); e.g. "## Arm B — ML-estimated same-time irradiance + power" |
| `open_questions.md` | 3 mention(s); e.g. "Neither README nor TTL states the plane, tilt or model of the irradiance sensor. Assumed horizontal from the weather-tower description." |
| `hk_design_checks_report.md` | none |
| `objective2_design_spec.md` | 6 mention(s); e.g. "- **Same-time irradiance estimation.** No forecast horizon is documented" |
| `open_decisions.md` | 1 mention(s); e.g. "/ Keep lagged power (historical behaviour) / Best short-horizon accuracy; risks attenuating sustained faults /" |
| `objective2_summary.md` | 12 mention(s); e.g. "# Objective 2 — Same-Time Irradiance Estimation for Sensor-Free PV Anomaly Screening" |
| `arms_report.md` | 1 mention(s); e.g. "Current-time exogenous values are inputs by design [4]: `ghi_clear(t)`, solar geometry, ambient temperature and `irr_est(t)` all carry the target time" |
| `prepare_report.md` | 3 mention(s); e.g. "## 3. Same-time irradiance estimator" |
| `residual_diagnostic_report.md` | none |
| `README.md` | none |
| `objective2_prepare.py` | 3 mention(s); e.g. "and the same-time irradiance estimator." |
| (docx sources) | python-docx unavailable |

**Current Objective 2 construction.** Its features are `ghi_clear, zenith, azimuth, Temp, hour_sin, hour_cos, month_sin, month_cos, day_of_year` and its target is `Irradiance (W/m2)` **at the same timestamp**. There is no lag, no shift, and no lead variable anywhere in the script. Whatever the intent, what is implemented is **same-time estimation**, not ahead-of-time prediction.

**Classification: SAME-TIME ESTIMATION.** No tutor material, brief, objective or protocol document states a future-prediction horizon.

| option | research question it answers | consequence |
|---|---|---|
| Same-time estimation (nowcast) | Can a model replace the physical irradiance sensor? | Legitimate sensor replacement **provided measured irradiance never enters at inference**. It is not circular: the sensor is absent at deployment and the estimate is built from time, geometry and non-irradiance weather. |
| Ahead-of-time prediction (t+h) | Can irradiance be estimated in advance for early fault warning? | A different and harder question. Requires choosing h, and makes the expected-power baseline a prediction about the future rather than an estimate of the present. |

**Conclusion.** The retained implementation is same-time estimation; no ahead-of-time task is defined here.

## Outputs written

- `hk_data_audit_report.md`
- `hk_semantics_findings.csv`
- `information_contracts_draft.md`
- `irradiance_dependency_table.csv`
- `open_questions.md`
