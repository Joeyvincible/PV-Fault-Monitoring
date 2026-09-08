# Objective 2 — Stage 1: prepare

Generated: 2026-09-06T16:17:53

## 1. Continuous grid from raw sources

- Raw PV rows: **90,624**
- Continuous grid: **90,624** rows, 2021-06-01 00:00:00 to 2023-12-31 23:45:00
- Grid spacing check: 1 distinct interval(s), modal 0 days 00:15:00
- **Grid is uniformly spaced with no gaps** — sequences can be built positionally

- Localised once to Asia/Hong_Kong; index tz-aware: **True**

## 2. pvlib quantities (SQ1 TTL parameters) and the eligibility mask

- Parameters: lat 22.33, lon 114.18, altitude 94.0 m (SQ1 TTL entry)
- Eligibility: `ghi_clear >= 50.0` AND `zenith < 85.0` — **no measured quantity participates**
- Eligible rows: **41,083** of 90,624 (45.3%)

## 3. Same-time irradiance estimator

- Predictors (14): ghi_clear, zenith, azimuth, Temp (Degree Celsius), RH (%), SLP (hPa), Wind Speed (m/s), Wind Direction (degree), hour_sin, hour_cos, month_sin, month_cos, day_of_year, Rainfall (mm)
- **[C2] assertion passed — `power` is not among them.** A fault depresses
  power; an estimator fed power would infer low irradiance and the fault
  would mask itself.
- Excluded by decision: `Vis (km)` (optical transparency — too close to the
  estimated quantity; retained in the grid for a later sensitivity test)

- Training rows (eligible, <= 2022-12-31, sensor valid): **15,773**
- Training-period rows excluded for impossible readings (>1200 W/m²): **0**

**2023 clean-sensor months (Jan,Feb,Nov,Dec)** (n=4,787)

| estimate | MAE (W/m²) | RMSE (W/m²) | R² |
|---|---|---|---|
| ML same-time estimate | 109.3 | 152.1 | 0.7345 |
| pvlib clear-sky | 214.1 | 275.0 | 0.1321 |

**2023 all eligible rows with a valid reading** (n=15,239)

| estimate | MAE (W/m²) | RMSE (W/m²) | R² |
|---|---|---|---|
| ML same-time estimate | 145.2 | 206.4 | 0.6027 |
| pvlib clear-sky | 249.2 | 323.3 | 0.0256 |

Evaluated only on periods where the measured signal is trustworthy — the March–October 2023 window is excluded from the clean-sensor row, since the reference instrument there is faulty.

## 4. Output

- `objective2_grid.csv` — 90,624 rows x 19 columns
- Columns: power_W, irr_meas, Temp (Degree Celsius), RH (%), SLP (hPa), Vis (km), Wind Speed (m/s), Wind Direction (degree), Rainfall (mm), zenith, azimuth, ghi_clear, hour_sin, hour_cos, month_sin, month_cos, day_of_year, eligible, irr_est
- No existing dataset or script was read or modified; built from raw only.
