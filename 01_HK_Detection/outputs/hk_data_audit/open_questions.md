# Open Questions — HK Audit

Items this audit could not resolve from available evidence.

## 1. AC/DC mismatch in expected power

PVWatts models DC; HK power(W) is AC. Whether to add an inverter efficiency model, or compare AC-to-AC by other means, is a design decision that affects both the baseline and any transfer claim.

**What would resolve it:** A decision on the expected-power definition, and whether Objective 3 transfer remains defensible across AC/DC.

## 2. Irradiance sensor plane and instrument

Neither README nor TTL states the plane, tilt or model of the irradiance sensor. Assumed horizontal from the weather-tower description.

**What would resolve it:** Dataset authors' confirmation, or an instrument specification.

## 3. Weather-station representativeness for SQ1

No documented pairing between SQ1 and the single campus weather station, and no stated separation distance.

**What would resolve it:** Station coordinates and a statement of intended coverage from the dataset authors.

## 4. Timestamp timezone

No file states a timezone. Local HK time is assumed; if the data were UTC every solar-position feature would be shifted by eight hours.

**What would resolve it:** A statement from the dataset authors, or a solar-noon alignment test against computed solar position.

## 5. Future-prediction horizon

No source states a horizon. The implemented script is a same-time estimator, with no ahead-of-time task defined.

**What would resolve it:** A decision from the tutor, or an explicit project choice recorded with its rationale.
