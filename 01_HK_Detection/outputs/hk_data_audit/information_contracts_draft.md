# Draft Information Contracts — Four Provisional Arms

**Status: PROVISIONAL DESIGN HYPOTHESIS, AWAITING APPROVAL.** These four arms
were reconstructed from a tutor meeting. They are not a confirmed
specification. Each contract specifies *permitted information sources*, not a
fixed feature list — features follow after approval and after the open
questions in `open_questions.md` are settled.

A single rule governs all four:

> Measured irradiance is "used" if it could influence whether a row exists,
> how a feature is calculated, how a constant or model is calibrated, or which
> sequence reaches the model — not merely if it appears in the feature list.

The audit found that the historical no-sensor scenario failed this test at two
pipeline stages (row filtering and sequence retention) in addition to two
contaminated features.

---

## Reference arm — measured irradiance + power

**Permitted:** measured irradiance at every stage; measured AC power; ambient
temperature; other measured meteorological variables; pvlib quantities; time
and geometry.

**Prohibited:** nothing, other than test-period information used to fit
training constants.

**Measured irradiance:** permitted at training and at inference.

**Purpose.** A reference benchmark, *not* an assumed performance ceiling. A
no-sensor arm could legitimately beat it — most plausibly during the confirmed
March–October 2023 sensor fault window, when the measured signal is corrupted
and a model that never depended on it may be more reliable.

**If it performs well:** establishes what the sensor is worth when working.
**If it performs poorly:** points to sensor fault or weather-station
representativeness rather than to model inadequacy.

---

## Arm A — pvlib + power

**Permitted:** timestamp; site location and panel geometry; pvlib-derived
quantities computed from time and geometry alone (`zenith`, `azimuth`,
`ghi_clear`, clear-sky POA); measured AC power; permitted non-irradiance
meteorological variables (ambient temperature, relative humidity, sea-level
pressure, wind speed and direction, rainfall — with visibility flagged as a
contract decision, see below).

**Prohibited:** measured irradiance at every stage — as a feature, as a row
filter, as a sequence-retention test, as an input to any derived column, and
as data for fitting any constant. This prohibits `k_clear`, `dni_est`,
`dhi_est`, `poa_global`, `temp_cell` as currently derived, `p_exp`, `p_exp_cal`,
`residual*`, `ratio*`, and α as currently fitted.

**Measured irradiance:** permitted at **neither** training nor inference.

**Note.** Cell temperature must be re-derived from clear-sky POA or from
ambient temperature alone, and the expected-power baseline must be rebuilt on
clear-sky irradiance. This arm cannot inherit any existing derived column.

**If it performs well:** physics alone, with no irradiance sensing, suffices
for fault detection — the strongest possible sensor-elimination result.
**If it performs poorly:** clear-sky assumptions are too crude under real cloud
cover, motivating Arm B.

---

## Arm B — ML-estimated same-time irradiance + power

**Permitted at inference:** everything Arm A permits, plus an irradiance
estimate produced by a model that consumes only Arm A-permitted inputs.

**Prohibited at inference:** measured irradiance, in any form, at any stage.

**Measured irradiance as a supervised training target — the explicit position:**

> Measured irradiance **is permitted as the supervised target** when training
> the irradiance estimator, on training-period data only. It is **prohibited**
> as an input at inference, as a row filter, as a sequence-retention test, and
> as data for fitting any constant applied to the test period.

This is stated explicitly rather than left implicit because it is the crux of
the arm's defensibility. Training an estimator against historical sensor
readings does not require a sensor at deployment: the sensor is needed once, to
build the model, not continuously to run it. This is the standard justification
for virtual sensing, and it is legitimate **provided** the deployment claim is
stated as "no sensor required at inference" rather than "no sensor ever
involved". If the intended claim is the stronger one, this arm does not support
it and Arm A is the only option.

A caveat that must accompany any result: the estimator is trained against a
sensor that is confirmed faulty from March to October 2023. Training on
2021–2022 avoids the fault window, but the arm cannot be *evaluated* against
measured irradiance during it.

**If it performs well:** a learned virtual sensor can replace the physical one
at deployment.
**If it performs poorly:** irradiance is not recoverable from time, geometry and
non-irradiance weather at this site.

---

## Arm C — pvlib + ML-estimated irradiance + power

**Permitted:** Arm A sources and Arm B's estimate, as two distinct information
sources available simultaneously.

**Prohibited:** as Arm B — measured irradiance at inference, and the same
training-target position applies.

**Measured irradiance:** training target only, never at inference.

**Hypothesis (to test, not an established result).** Physics-derived and
learned estimates carry complementary information, and the *gap* between
clear-sky and estimated irradiance is itself a cloud-cover signal. Arm C tests
whether exposing both representations beats either alone.

**If it performs well:** the divergence between physical expectation and
learned estimate is informative in its own right.
**If it performs poorly:** the learned estimate subsumes the clear-sky signal
and the extra representation adds nothing.

---

## Cross-cutting requirements for all arms

1. Any learned constant, scaler, imputer or threshold is fitted on
   training-period data only and applied unchanged thereafter.
2. Row existence and sequence retention must be decidable without the
   irradiance sensor in Arms A, B and C. A clear-sky daylight criterion
   satisfies this; the current `build_sequences` test does not.
3. The AC/DC question from Part 1 applies to every arm: PVWatts models DC while
   `power(W)` is AC. The expected-power definition must be settled before any
   arm is built.
4. Report the March–October 2023 sensor-fault window separately in every arm.
