# Objective 2 — Design Checks (read-only)

Generated: 2026-09-06T16:18:22

## 1. Raw 15-minute grid (no irradiance filter applied)

- Raw PV rows: **90,624** (2021-06-01 00:00:00 to 2023-12-31 23:45:00)
- Grid rows after joining meteorological means: **90,624**
- Rows with a measured-irradiance value: **90,027**

## 2. pvlib parameter correction: effect on `ghi_clear`

| quantity | mean | max abs | mean % | max % |
|---|---|---|---|---|
| ghi_clear difference (TTL − historical), W/m² | -0.301 | 3.610 | -0.4964% | 549.8367% |
| solar zenith difference, degrees | 0.0631 | 0.0774 | – | – |

Evaluated on 45,766 rows with non-zero clear-sky GHI.

## 3. Candidate eligibility rules

| rule | rows retained | % of grid | 2023 rows | typhoon-window rows | sensor-free? |
|---|---|---|---|---|---|
| HISTORICAL sensor rule (20<=irr<=1200) [contaminated] | 51,625 | 57.0% | 23,756 | 908 | no — uses the sensor |
| ghi_clear >= 20 | 42,636 | 47.0% | 16,475 | 644 | **yes** |
| ghi_clear >= 50 | 41,083 | 45.3% | 15,875 | 618 | **yes** |
| zenith < 85 | 42,517 | 46.9% | 16,424 | 645 | **yes** |
| zenith < 80 | 39,767 | 43.9% | 15,358 | 612 | **yes** |
| ghi_clear >= 50 AND zenith < 85 | 41,083 | 45.3% | 15,875 | 618 | **yes** |

Typhoon window = 2023-09-01 to 2023-09-15; total grid rows in that window (unfiltered): **1,345**

### 3b. Set differences against the historical sensor rule

| candidate | in candidate only | in historical only | in both | candidate-only rows that are overcast (irr<20) | candidate-only rows with no sensor value |
|---|---|---|---|---|---|
| ghi_clear >= 20 | 1,974 | 10,963 | 40,662 | 1,224 | 262 |
| ghi_clear >= 50 | 1,542 | 12,084 | 39,541 | 802 | 252 |
| zenith < 85 | 1,903 | 11,011 | 40,614 | 1,155 | 260 |
| zenith < 80 | 1,336 | 13,194 | 38,431 | 602 | 246 |
| ghi_clear >= 50 AND zenith < 85 | 1,542 | 12,084 | 39,541 | 802 | 252 |

## 4. Sensor fault window and the clean-sensor 2023 subset

| month 2023 | max measured irradiance (W/m²) | exceeds 1200? | rows |
|---|---|---|---|
| 01 | 685.8 | no | 2,976 |
| 02 | 1196.7 | no | 2,688 |
| 03 | 1208.7 | **YES** | 2,976 |
| 04 | 1328.7 | **YES** | 2,880 |
| 05 | 1412.5 | **YES** | 2,976 |
| 06 | 1507.5 | **YES** | 2,880 |
| 07 | 1467.5 | **YES** | 2,976 |
| 08 | 1529.1 | **YES** | 2,976 |
| 09 | 1415.6 | **YES** | 2,486 |
| 10 | 1299.5 | **YES** | 2,976 |
| 11 | 972.7 | no | 2,880 |
| 12 | 987.4 | no | 2,976 |

- Months with physically impossible readings: **[3, 4, 5, 6, 7, 8, 9, 10]**
- Clean-sensor months in 2023: **[1, 2, 11, 12]**
- Clean-sensor 2023 rows (all-hours grid): **11,520** of 35,040 (32.9%)
- Clean-sensor 2023 rows under the proposed sensor-free rule (ghi_clear>=50 & zenith<85): **4,749**
- Faulty-window 2023 rows under the same rule: **11,126**

The fault window covers 8 of 12 months of 2023 — a substantial portion of the test period, but **not** the whole of it.

## 5. Typhoon window composition

- Grid rows in the window: **1,345**
- Retained by the historical sensor rule: **908**
- Retained by the proposed sensor-free rule: **618**
- Admitted by the sensor-free rule but excluded by the sensor rule: **195**
- Of those, rows whose measured irradiance was below 20 W/m² (overcast or storm-darkened): **0**

These are exactly the rows a storm suppresses. Under the historical rule they were deleted before the model ever saw them; under a sensor-free rule they are retained, which is the point of the correction.
