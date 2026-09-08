"""
objective2_prepare.py
=====================
Objective 2 — Stage 1: continuous grid, pvlib, eligibility mask,
and the same-time irradiance estimator.

Built from RAW sources only. No existing derived CSV is read: row existence
in all of them was decided by a measured-irradiance filter, so a sensor-free
arm cannot inherit that row set.

BINDING CONSTRAINTS IMPLEMENTED HERE
------------------------------------
[C2] power(W) is NOT a predictor of the irradiance estimator. Its predictors
     are time, solar geometry, clear-sky and non-irradiance weather only.
     A fault depresses power; an estimator fed power would infer low
     irradiance, expected power would fall to match, and the fault would
     mask itself.
[C1] One common sensor-free eligibility mask (clear-sky + geometry only),
     used later by all four arms including Reference.
     Rule: ghi_clear >= 50 W/m2 AND solar zenith < 85 deg.
Training-only fitting: the estimator sees timestamps <= TRAIN_END only.
Timezone: localised once to Asia/Hong_Kong, never stripped.

Outputs:
  outputs/objective2_modeling/objective2_grid.csv
  outputs/objective2_modeling/irradiance_estimator_metrics.csv
  outputs/objective2_modeling/prepare_report.md
"""

import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib.location import Location
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_ROOT    = PROJECT_DIR / "raw_data"
RAW_DATA_DIR = RAW_ROOT / "dryad_dataset"
TS          = RAW_DATA_DIR / "Time series dataset"
MET         = TS / "Meteorological dataset"
SQ1_RAW     = (TS / "PV generation dataset" / "PV stations with panel level optimizer"
               / "Site level dataset" / "SQ1.csv")
OUT_DIR     = PROJECT_DIR / "outputs" / "objective2_modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG     = PROJECT_DIR / "outputs" / "run_log.txt"

# SQ1's own TTL entry, not the campus centroid
LAT, LON, ALT, TZ = 22.33, 114.18, 94.0, "Asia/Hong_Kong"
FREQ = "15min"
ELIG_GHI_CLEAR, ELIG_ZENITH = 50.0, 85.0
TRAIN_END = "2022-12-31"
SENSOR_MAX = 1200.0          # physically impossible above this — estimator target only
RANDOM_SEED = 42

MET_VARS = {
    "Temperature":       "Temp (Degree Celsius)",
    "Relative Humidity": "RH (%)",
    "Sea Level Pressure": "SLP (hPa)",
    "Visibility":        "Vis (km)",
}
WIND_DIR = "Wind"


def require_repository_raw_sources():
    """Fail clearly when the submitted raw source subset is not local."""
    expected = [SQ1_RAW]
    for folder, names in {
        "Irradiance": ["Irradiance_2021.csv", "Irradiance_2022.csv", "Irradiance_2023.csv"],
        "Temperature": ["Temperature_2021.csv", "Temperature_2022.csv", "Temperature_2023.csv"],
        "Relative Humidity": ["Relative Humidity_2021.csv", "Relative Humidity_2022.csv", "Relative Humidity_2023.csv"],
        "Sea Level Pressure": ["Sea Level Pressure_2021.csv", "Sea Level Pressure_2022.csv", "Sea Level Pressure_2023.csv"],
        "Visibility": ["Visibility_2021.csv", "Visibility_2022.csv", "Visibility_2023.csv"],
        "Wind": ["Wind_2021.csv", "Wind_2022.csv", "Wind_2023.csv"],
        "Rainfall": ["Rainfall_2022.csv", "Rainfall_2023.csv"],
    }.items():
        expected.extend(MET / folder / name for name in names)
    missing = [str(p.relative_to(PROJECT_DIR)) for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing repository-local Hong Kong raw data. Expected files under "
            f"{RAW_ROOT.relative_to(PROJECT_DIR)}/; no external fallback is configured: "
            + ", ".join(missing)
        )


require_repository_raw_sources()

_rep = []


def emit(s=""):
    print(s)
    _rep.append(s)


emit("# Objective 2 — Stage 1: prepare")
emit("")
emit(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONTINUOUS 15-MINUTE GRID FROM RAW
# ─────────────────────────────────────────────────────────────────────────────
emit("## 1. Continuous grid from raw sources")
emit("")
pv = pd.read_csv(SQ1_RAW)
pv["Time"] = pd.to_datetime(pv["Time"], errors="coerce")
pv = pv.dropna(subset=["Time"]).sort_values("Time").set_index("Time")
pv = pv[~pv.index.duplicated(keep="first")]


def load_met(folder, cols):
    frames = []
    for f in sorted((MET / folder).glob("*.csv")):
        d = pd.read_csv(f)
        d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
        d = d.dropna(subset=["Time"])
        for c in cols:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        keep = ["Time"] + [c for c in cols if c in d.columns]
        frames.append(d[keep])
    m = pd.concat(frames, ignore_index=True).sort_values("Time").set_index("Time")
    return m.resample(FREQ).mean()


irr = load_met("Irradiance", ["Irradiance (W/m2)"])["Irradiance (W/m2)"]
met = pd.DataFrame(index=irr.index)
for folder, col in MET_VARS.items():
    met[col] = load_met(folder, [col])[col]
wind = load_met(WIND_DIR, ["Wind Speed (m/s)", "Wind Direction (degree)"])
met["Wind Speed (m/s)"] = wind["Wind Speed (m/s)"]
met["Wind Direction (degree)"] = wind["Wind Direction (degree)"]

# Rainfall: 2021 is .xlsx, later years CSV — take whatever is machine-readable
rain_frames = []
for f in sorted((MET / "Rainfall").glob("*.csv")):
    d = pd.read_csv(f)
    d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
    d = d.dropna(subset=["Time"])
    rc = [c for c in d.columns if c != "Time"][0]
    d[rc] = pd.to_numeric(d[rc], errors="coerce")
    rain_frames.append(d[["Time", rc]].rename(columns={rc: "Rainfall (mm)"}))
if rain_frames:
    rain = (pd.concat(rain_frames).sort_values("Time").set_index("Time")
            .resample(FREQ).mean()["Rainfall (mm)"])
    met["Rainfall (mm)"] = rain

# Complete, gap-free 15-minute index — the continuous grid [C4]
full_idx = pd.date_range(pv.index.min(), pv.index.max(), freq=FREQ)
grid = pd.DataFrame(index=full_idx)
grid["power_W"] = pv["power(W)"].reindex(full_idx)
grid["irr_meas"] = irr.reindex(full_idx)
for c in met.columns:
    grid[c] = met[c].reindex(full_idx)

emit(f"- Raw PV rows: **{len(pv):,}**")
emit(f"- Continuous grid: **{len(grid):,}** rows, "
     f"{grid.index.min()} to {grid.index.max()}")
gaps = pd.Series(grid.index).diff().dropna().value_counts()
emit(f"- Grid spacing check: {len(gaps)} distinct interval(s), modal {gaps.index[0]}")
assert len(gaps) == 1 and gaps.index[0] == pd.Timedelta(FREQ), \
    "Grid is not uniformly spaced"
emit("- **Grid is uniformly spaced with no gaps** — sequences can be built positionally")
emit("")

# Localise once, never strip
grid.index = grid.index.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
grid = grid[~grid.index.isna()]
assert grid.index.tz is not None, "Index must remain timezone-aware"
emit(f"- Localised once to {TZ}; index tz-aware: **{grid.index.tz is not None}**")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PVLIB + ELIGIBILITY
# ─────────────────────────────────────────────────────────────────────────────
emit("## 2. pvlib quantities (SQ1 TTL parameters) and the eligibility mask")
emit("")
site = Location(latitude=LAT, longitude=LON, altitude=ALT, tz=TZ)
sp = site.get_solarposition(grid.index)
cs = site.get_clearsky(grid.index, model="ineichen")
grid["zenith"] = sp["zenith"].to_numpy()
grid["azimuth"] = sp["azimuth"].to_numpy()
grid["ghi_clear"] = cs["ghi"].to_numpy()

h = grid.index.hour + grid.index.minute / 60
grid["hour_sin"] = np.sin(2 * np.pi * h / 24)
grid["hour_cos"] = np.cos(2 * np.pi * h / 24)
grid["month_sin"] = np.sin(2 * np.pi * grid.index.month / 12)
grid["month_cos"] = np.cos(2 * np.pi * grid.index.month / 12)
grid["day_of_year"] = grid.index.dayofyear

grid["eligible"] = ((grid["ghi_clear"] >= ELIG_GHI_CLEAR)
                    & (grid["zenith"] < ELIG_ZENITH))
emit(f"- Parameters: lat {LAT}, lon {LON}, altitude {ALT} m (SQ1 TTL entry)")
emit(f"- Eligibility: `ghi_clear >= {ELIG_GHI_CLEAR}` AND `zenith < {ELIG_ZENITH}` "
     f"— **no measured quantity participates**")
emit(f"- Eligible rows: **{int(grid['eligible'].sum()):,}** of {len(grid):,} "
     f"({100*grid['eligible'].mean():.1f}%)")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# 3. IRRADIANCE ESTIMATOR  [C2]
# ─────────────────────────────────────────────────────────────────────────────
emit("## 3. Same-time irradiance estimator")
emit("")
EST_FEATURES = ["ghi_clear", "zenith", "azimuth",
                "Temp (Degree Celsius)", "RH (%)", "SLP (hPa)",
                "Wind Speed (m/s)", "Wind Direction (degree)",
                "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year"]
if "Rainfall (mm)" in grid.columns:
    EST_FEATURES.append("Rainfall (mm)")
EST_FEATURES = [c for c in EST_FEATURES if c in grid.columns]

# [C2] hard guarantee: no power-derived column may be a predictor
POWER_COLS = {"power_W", "power(W)"}
assert not (set(EST_FEATURES) & POWER_COLS), \
    "[C2] VIOLATION: power is present in the estimator predictors"
emit(f"- Predictors ({len(EST_FEATURES)}): {', '.join(EST_FEATURES)}")
emit("- **[C2] assertion passed — `power` is not among them.** A fault depresses")
emit("  power; an estimator fed power would infer low irradiance and the fault")
emit("  would mask itself.")
emit("- Excluded by decision: `Vis (km)` (optical transparency — too close to the")
emit("  estimated quantity; retained in the grid for a later sensitivity test)")
emit("")

train_end_ts = pd.Timestamp(TRAIN_END, tz=TZ)
est_train = grid[grid["eligible"] & (grid.index <= train_end_ts)
                 & grid["irr_meas"].notna()
                 & (grid["irr_meas"] <= SENSOR_MAX)].copy()
est_train = est_train.dropna(subset=EST_FEATURES)
emit(f"- Training rows (eligible, <= {TRAIN_END}, sensor valid): "
     f"**{len(est_train):,}**")
n_bad_train = int((grid[grid.index <= train_end_ts]["irr_meas"] > SENSOR_MAX).sum())
emit(f"- Training-period rows excluded for impossible readings (>{SENSOR_MAX:.0f} W/m²): "
     f"**{n_bad_train:,}**")

est = HistGradientBoostingRegressor(max_iter=300, random_state=RANDOM_SEED)
est.fit(est_train[EST_FEATURES].to_numpy(), est_train["irr_meas"].to_numpy())

X_all = grid[EST_FEATURES].to_numpy()
grid["irr_est"] = np.clip(est.predict(X_all), 0, None)
# NOTE: the estimate is produced for EVERY row, including ineligible ones.
# Eligibility governs which timestamps may be prediction TARGETS [C1]; it must
# not blank the input window, or sequences whose 16-step context reaches back
# into pre-dawn slots would be dropped for Arms B and C only — leaving the
# arms on different evaluation populations, which [C1] forbids.

# Evaluate on the trustworthy 2023 subset (clean-sensor months)
CLEAN_MONTHS = [1, 2, 11, 12]
te = grid[grid["eligible"] & (grid.index > train_end_ts)
          & grid["irr_meas"].notna() & (grid["irr_meas"] <= SENSOR_MAX)]
clean = te[te.index.month.isin(CLEAN_MONTHS)]
rows = []
for label, sub in [("2023 clean-sensor months (Jan,Feb,Nov,Dec)", clean),
                   ("2023 all eligible rows with a valid reading", te)]:
    if len(sub) < 50:
        continue
    y, p = sub["irr_meas"].to_numpy(), sub["irr_est"].to_numpy()
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    mae = mean_absolute_error(y, p)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    r2 = r2_score(y, p)
    # clear-sky as the physics comparator
    c = sub["ghi_clear"].to_numpy()[ok]
    rows.append({"slice": label, "n": int(len(y)),
                 "est_MAE": round(mae, 2), "est_RMSE": round(rmse, 2),
                 "est_R2": round(r2, 4),
                 "clearsky_MAE": round(mean_absolute_error(y, c), 2),
                 "clearsky_RMSE": round(float(np.sqrt(mean_squared_error(y, c))), 2),
                 "clearsky_R2": round(r2_score(y, c), 4)})
    emit("")
    emit(f"**{label}** (n={len(y):,})")
    emit("")
    emit("| estimate | MAE (W/m²) | RMSE (W/m²) | R² |")
    emit("|---|---|---|---|")
    emit(f"| ML same-time estimate | {mae:.1f} | {rmse:.1f} | {r2:.4f} |")
    emit(f"| pvlib clear-sky | {mean_absolute_error(y, c):.1f} | "
         f"{float(np.sqrt(mean_squared_error(y, c))):.1f} | {r2_score(y, c):.4f} |")
pd.DataFrame(rows).to_csv(OUT_DIR / "irradiance_estimator_metrics.csv", index=False)
emit("")
emit("Evaluated only on periods where the measured signal is trustworthy — the "
     "March–October 2023 window is excluded from the clean-sensor row, since "
     "the reference instrument there is faulty.")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE GRID
# ─────────────────────────────────────────────────────────────────────────────
out = OUT_DIR / "objective2_grid.csv"
grid.to_csv(out)
emit("## 4. Output")
emit("")
emit(f"- `{out.name}` — {len(grid):,} rows x {grid.shape[1]} columns")
emit(f"- Columns: {', '.join(grid.columns)}")
emit(f"- No existing dataset or script was read or modified; built from raw only.")
(OUT_DIR / "prepare_report.md").write_text("\n".join(_rep) + "\n")
print(f"\nSaved to: {OUT_DIR}")
with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"objective2_prepare.py | input: raw SQ1.csv + raw meteorological CSVs | "
            f"output: objective2_grid.csv ({len(grid):,} rows), estimator metrics | "
            f"notes: continuous 15-min grid, TTL pvlib params, sensor-free "
            f"eligibility {int(grid['eligible'].sum()):,} rows; estimator excludes "
            f"power [C2]; trained <= {TRAIN_END} only\n")
print(f"Run log appended: {RUN_LOG}")
