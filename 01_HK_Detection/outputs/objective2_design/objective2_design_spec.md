# Objective 3 — Hong Kong Rebuild: Design Specification

**Status: SPECIFICATION FOR APPROVAL. No pipeline code written, nothing trained.**

Generated from the HK audit plus the read-only computations in
`hk_design_checks.py`. Four corrections issued at commissioning are applied
and are marked **[C1]**–**[C4]** where they bind.

Two fixed decisions carried in from the audit:

- **Same-time irradiance estimation.** No forecast horizon is documented
  anywhere in the project and the historical script already targets the same
  timestamp. Objective 3 estimates irradiance at the current timestamp from
  information available without the sensor. Earlier project wording said
  "forecasting"; that is operationalised here as same-time estimation and
  **no future-prediction claim is made**. The corrected term is used
  throughout.
- **No DC→AC bridging parameter.** `power(W)` is confirmed AC active power
  and PVWatts models DC. No assumed inverter efficiency is introduced.
  Consequently `p_exp`, `p_exp_cal`, `residual`, `residual_cal`, `ratio`,
  `ratio_cal` and the calibration constant α are **not used in any arm**.

---

## Part 1 — The four arms

Each arm is defined by permitted *information sources*, not by a fixed
feature list. Where a choice remains genuinely open it is raised in
`open_decisions.md` rather than settled silently.

### Reference arm — measured irradiance + AC power

| | |
|---|---|
| **Permitted** | Measured irradiance at every stage as a *feature*; measured AC power; ambient temperature; pvlib clear-sky and solar geometry; time |
| **Prohibited** | Test-period data for fitting any learned quantity |
| **Measured irradiance** | Permitted at training and at inference |

A **reference benchmark, not an assumed ceiling.** A no-sensor arm may
legitimately beat it, most plausibly inside the March–October 2023 fault
window where the measured signal is corrupted.

**[C1]** The Reference arm is *evaluated on the common sensor-free
eligibility mask*, exactly like the other three. It may consume the sensor as
an input; it does not get to choose its own evaluation population.

### Arm A — pvlib + AC power

| | |
|---|---|
| **Permitted** | Time; SQ1 location and geometry from the TTL; pvlib clear-sky and solar-position quantities; measured AC power; ambient temperature |
| **Prohibited** | Measured irradiance at **every** stage — features, row selection, daylight rule, sequence formation and retention, scaler fitting, threshold selection, and any learned constant |
| **Measured irradiance** | Permitted at **neither** training nor inference |

Arm A **cannot inherit any existing derived CSV**: row existence in all of
them was decided by a measured-irradiance filter. It is built from raw
`SQ1.csv` and the raw meteorological files.

Prohibited columns follow automatically: `k_clear`, `dni_est`, `dhi_est`,
`poa_global`, `temp_cell` as currently derived, and every `p_exp*`,
`residual*`, `ratio*` quantity.

### Arm B — same-time ML irradiance estimate + AC power

| | |
|---|---|
| **Permitted at inference** | Everything Arm A permits, plus an irradiance estimate produced by a model consuming only permitted inputs |
| **Prohibited at inference** | Measured irradiance in any form, at any stage |
| **Measured irradiance at training** | **Permitted as the supervised target of the estimator, on training-period rows only** |

**Defence of the training-target position.** Training an estimator against
historical sensor readings does not require a sensor at deployment: the
sensor is needed once to build the model, not continuously to run it. This is
the standard justification for virtual sensing and it is legitimate
**provided** the deployment claim is stated as *"no irradiance sensor
required at inference"* rather than *"no sensor was ever involved"*. If the
stronger claim is wanted, Arm B does not support it and Arm A is the only
option. This position is stated explicitly here so it is never implicit in a
result.

**[C2] No power → irradiance → power circularity — binding.**

> `power(W)` at time *t* **must not** be an input to the irradiance
> estimator. Its predictors are drawn only from permitted non-irradiance
> environmental, time and location information.

The failure mode this prevents: a genuine fault depresses measured power; an
estimator fed that power infers low irradiance; expected power falls to
match; the residual stays near zero and the fault masks itself. Any proposal
to use *lagged* PV power in the estimator requires separate written
justification before implementation — it is not permitted by default.

### Arm C — pvlib + ML estimate + AC power

| | |
|---|---|
| **Permitted** | Arm A sources plus Arm B's estimate, as two distinct information sources |
| **Prohibited** | As Arm B; **[C2]** applies identically to the estimator |
| **Measured irradiance** | Training target of the estimator only, never at inference |

**Hypothesis to test, not an established result:** clear-sky gives a
theoretical environmental ceiling, the ML estimate a weather-adjusted
expectation, and the *gap between them* is itself a cloud-cover signal. Arm C
tests whether exposing both beats either alone.

---

## Part 2 — Sensor-free row and sequence rules

### 2.1 The common eligibility mask **[C1]**

**The "intersection of both rules" option is removed.** Intersecting with the
sensor rule would let measured irradiance decide which rows the no-sensor
arms are scored on — the same contamination the audit found at the
row-existence stage, merely relocated to evaluation.

> **Rule.** One common eligibility mask, computed from pvlib clear-sky and
> solar geometry alone, is applied identically to all four arms:
>
> ```
> eligible(t)  ⇔  ghi_clear(t) ≥ 50 W/m²  AND  solar_zenith(t) < 85°
> ```
>
> No measured quantity of any kind enters this mask.

All four arms train and are evaluated on this population. The historical
`20 ≤ Irradiance ≤ 1200` row set may be reported **as a labelled diagnostic
comparison only** and never as an evaluation population.

**Row counts on the raw 15-minute grid (90,624 rows, 2021-06 to 2023-12):**

| rule | rows | % of grid | 2023 rows | typhoon-window rows | sensor-free |
|---|---|---|---|---|---|
| historical `20 ≤ irr ≤ 1200` (diagnostic only) | 51,625 | 57.0% | 23,756 | 908 | no |
| `ghi_clear ≥ 20` | 42,636 | 47.0% | 16,475 | 644 | yes |
| **`ghi_clear ≥ 50` AND `zenith < 85` (proposed)** | **41,083** | **45.3%** | **15,875** | **618** | **yes** |
| `zenith < 85` | 42,517 | 46.9% | 16,424 | 645 | yes |
| `zenith < 80` | 39,767 | 43.9% | 15,358 | 612 | yes |

`ghi_clear ≥ 50` alone yields the identical 41,083 rows, so the zenith clause
is redundant in practice; it is retained because it makes the geometric
intent explicit and is robust if the clear-sky model or site parameters
change.

**Why the proposed rule retains fewer rows than the historical one — and why
that is a correction, not a loss.** The historical rule keeps 12,084 rows
that the geometric rule rejects. Their **median solar zenith is 116.4°** —
the sun is well below the horizon. These are *night* rows admitted because
the pyranometer has a non-zero night floor: mean measured irradiance with the
sun down is **9.85 W/m²**, and 8,893 rows read ≥ 20 W/m² at night. The
historical `MIN_IRR = 20` threshold was therefore not a daylight rule at all
in the small hours; it was partly reading sensor offset. A geometry-based
rule cannot make this mistake.

**What the sensor-free rule admits that the sensor rule excluded** — 1,542
rows grid-wide:

| reason the sensor rule excluded them | rows |
|---|---|
| measured irradiance < 20 W/m² (genuinely overcast) | 802 |
| measured irradiance > 1200 W/m² (physically impossible — sensor fault) | 488 |
| measured irradiance missing (NaN) | 252 |

All three categories are rows a sensor-free arm can legitimately evaluate and
the historical pipeline destroyed.

**The 1,200 W/m² ceiling** was a sensor-fault guard with no clear-sky
analogue. Under the new mask it is simply not needed: eligibility never
consults the sensor, so impossible readings cannot influence row existence.
For the Reference arm, which does consume the sensor as a feature, readings
above 1,200 W/m² are retained but **flagged** and reported separately rather
than deleted — deleting them is what tied row existence to the sensor.

**Typhoon window (Sept 1–15 2023), 1,345 grid rows.** The historical rule
retains 908, the proposed rule 618. The proposed rule admits 195 rows the
sensor rule excluded — and the breakdown corrects an assumption I had
expected to hold: **186 are missing sensor values and 9 are impossible
readings; none is an overcast row.** During the storm the sensor was largely
*absent*, not merely dark. That is a stronger argument for the sensor-free
mask than the one anticipated: the no-sensor arms can be evaluated across a
window in which the reference instrument frequently produced nothing at all.

### 2.2 Sequence construction

`build_sequences` must not consult measured irradiance in Arms A, B or C. The
replacement:

> A candidate sequence with target timestamp *t* is formed and retained
> **iff** `eligible(t)` under the common mask, and the input window is
> complete. Eligibility is evaluated on the target row. No measured
> irradiance value participates in formation, retention or target selection.

### 2.3 Target-time power leakage **[C3]**

The downstream model predicts AC power at *t*. Therefore:

> `power(t)` **must not** appear in the input features for target *t*. Where
> historical power is used it enters only as explicitly lagged values
> strictly preceding the target.

**Window layout, stated explicitly:**

```
input window:   t-L, t-L+1, ..., t-2, t-1      (L lags; L = 16 → 4 hours at 15-min)
                └── lagged power and lagged exogenous variables ──┘

exogenous at t:  ghi_clear(t), zenith(t), azimuth(t), temp(t),
                 irradiance_estimate(t)  [Arms B and C]
                 — none of these is power

target:          power(t)                       ← never present in any input
```

**Verification requirement.** The implementation must assert, for every
constructed sequence, that the maximum timestamp among power-derived inputs
is strictly less than the target timestamp, and report the count of
violations. **That count must be zero**, printed in the run log. A second
assertion must confirm that the target column is absent from the
contemporaneous exogenous block.

Note that lagged power remains an input under this rule. That is permitted,
but it carries a residual concern — a fault persisting longer than the input
window is partly absorbed into the lagged history, which would attenuate the
residual. This is raised as an open decision, not settled here.

---

## Part 3 — Corrected pvlib configuration

| Parameter | Value | Source |
|---|---|---|
| Latitude | 22.33 | SQ1 TTL entry |
| Longitude | **114.18** (not 114.2634) | SQ1 TTL entry, not campus centroid |
| Altitude | **94.0 m** (not 30 m) | SQ1 TTL entry |
| Tilt | 0° | TTL, confirmed |
| Azimuth | inert at tilt 0 | — |
| Timezone | Asia/Hong_Kong | no DST since 1979 |

### 3.1 Quantified impact of the correction

Over 45,766 rows with non-zero clear-sky GHI:

| quantity | mean difference | max absolute |
|---|---|---|
| `ghi_clear` (TTL − historical) | **−0.301 W/m²** | 3.61 W/m² |
| solar zenith | 0.063° | 0.077° |

Relative differences, restricted to avoid division artefacts:

| subset | mean % | max abs % |
|---|---|---|
| all rows with `ghi_clear > 0` | −0.55% | 549.8% |
| `ghi_clear > 50 W/m²` | −0.21% | 4.64% |
| `ghi_clear > 200 W/m²` | −0.09% | 1.71% |

**The 549.8% figure is a division artefact**, not a physical effect: it
occurs where clear-sky GHI is a fraction of a watt at sunrise or sunset, so a
sub-watt absolute difference divides to a large percentage. On the rows the
eligibility mask actually retains (`ghi_clear ≥ 50`), the correction is
**−0.21% on average and at most 4.6%**.

**Conclusion: the parameter correction is negligible in effect but is adopted
anyway**, because using SQ1's own metadata rather than a campus centroid is
correct on principle and costs nothing. It should not be presented as a
material improvement to results.

### 3.2 Timezone handling — now CONFIRMED, upgraded from the audit

The audit classified the HK-local assumption as SUPPORTED. A direct alignment
test settles it:

| interpretation | corr(measured irradiance, clear-sky GHI) | mean measured irradiance with sun down |
|---|---|---|
| **timestamps are HK local (+8)** | **0.739** | 9.85 W/m² |
| timestamps are UTC | −0.328 | 217.68 W/m² |

Mean measured irradiance by raw clock hour peaks at **hour 12 (497 W/m²)**,
consistent with HK solar noon near 12:30 local. Under a UTC reading the
correlation is *negative* and the sun-down mean is 218 W/m², which is
physically impossible. **The timestamps are Hong Kong local time —
CONFIRMED.**

**Handling.** Localise once, at load, to `Asia/Hong_Kong`; carry a
timezone-aware index throughout; never strip and re-localise. The historical
naive → aware → naive round-trip is eliminated. Verification: assert the
index is tz-aware after load and that the count of rows per calendar day is
unchanged by localisation.

---

## Part 4 — Open decisions

Presented with options, trade-offs and recommendations in
`open_decisions.md`. They are the decisions this approval confirms.

---

## Part 5 — Cross-cutting binding constraints

1. **Training-only fitting.** Every learned quantity — scaler, imputer,
   irradiance estimator, detection model, fault threshold — is fitted on
   training-period rows only and applied unchanged to the test period.
2. **Sensor-free decisions.** In Arms A, B and C no measured-irradiance value
   may influence row existence, sequence formation, sequence retention,
   feature computation, scaler fitting or threshold selection.
3. **Common evaluation population [C1].** All four arms are evaluated on the
   identical sensor-free eligibility mask. The historical row set is a
   labelled diagnostic only.
4. **No estimator circularity [C2].** `power(W)` at time *t* is not an input
   to the irradiance estimator; lagged power requires separate justification.
5. **No target-time leakage [C3].** `power(t)` never appears in the inputs
   for target *t*; strict-lag assertion required, violations must be zero.
6. **No DC/AC bridging parameter.** No assumed inverter efficiency; no
   DC-modelled expected power compared against measured AC power.
7. **Terminology.** "Same-time irradiance estimation", never "forecasting".
8. **Sensor fault window reported separately [C4].** March–October 2023
   affects **8 of 12 months** — a substantial portion of the test period, not
   the whole of it. January–February and November–December 2023 form a
   clean-sensor subset and are reported as a separate evaluation slice.
9. **Limitations carried forward.** Irradiance plane unresolved (inert at
   tilt 0 for SQ1); weather-station pairing to SQ1 undocumented; α and its
   leakage not used, but the historical result's interpretation is affected.

---

## Appendix — evidence files

- `daylight_rule_comparison.csv` — row counts for every candidate rule
- `daylight_rule_setdiff.csv` — set differences against the historical rule
- `pvlib_parameter_impact.csv` — effect of the TTL correction on `ghi_clear`
- `sensor_fault_months_2023.csv` — monthly maxima and fault classification
- `hk_design_checks_report.md` — full computation log
