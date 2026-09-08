"""
hk_design_checks.py
===================
Objective 2 — read-only computations supporting the design
specification. Writes nothing except its own outputs in outputs/objective2_design/.
Trains nothing. Modifies no script, dataset or existing output.

WHY THIS WORKS FROM RAW DATA
-----------------------------
Every derived CSV in data/ had its rows filtered by measured irradiance
(20 <= Irradiance <= 1200) during preprocessing. A sensor-free arm cannot
inherit that row set, so the candidate daylight rules must be evaluated on
the raw 15-minute grid reconstructed from SQ1.csv plus the raw meteorological
files — not on sq1_calibrated_baseline.csv.

Computes:
  1. Candidate sensor-free eligibility masks vs the historical sensor rule
  2. Set differences, including rows the sensor rule excluded but a
     clear-sky rule admits (the overcast question)
  3. Composition of the Typhoon Saola window under each rule
  4. Effect of the TTL parameter correction (longitude, altitude) on ghi_clear
  5. Sensor-fault window characterisation and the clean-sensor 2023 subset
"""

import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_ROOT    = PROJECT_DIR / "raw_data"
DS          = RAW_ROOT / "dryad_dataset"
TS          = DS / "Time series dataset"
MET         = TS / "Meteorological dataset"
SQ1_RAW     = (TS / "PV generation dataset" / "PV stations with panel level optimizer"
               / "Site level dataset" / "SQ1.csv")
OUT_DIR     = PROJECT_DIR / "outputs" / "objective2_design"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG     = PROJECT_DIR / "outputs" / "run_log.txt"

# Historical (contaminated) rule, for diagnostic comparison only
HIST_MIN_IRR, HIST_MAX_IRR = 20, 1200
TYPHOON = ("2023-09-01", "2023-09-15")
TRAIN_END, TEST_START = "2022-12-31", "2023-01-01"

# Parameter sets: campus centroid (historical) vs SQ1's own TTL entry
PARAMS_HIST = dict(latitude=22.3363, longitude=114.2634, altitude=30.0,
                   tz="Asia/Hong_Kong", label="historical (campus centroid)")
PARAMS_TTL  = dict(latitude=22.33, longitude=114.18, altitude=94.0,
                   tz="Asia/Hong_Kong", label="SQ1 TTL entry")


def require_repository_raw_sources():
    expected = [SQ1_RAW]
    for folder, names in {
        "Irradiance": ["Irradiance_2021.csv", "Irradiance_2022.csv", "Irradiance_2023.csv"],
        "Temperature": ["Temperature_2021.csv", "Temperature_2022.csv", "Temperature_2023.csv"],
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


emit("# Objective 2 — Design Checks (read-only)")
emit("")
emit(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# Reconstruct the raw 15-minute grid
# ─────────────────────────────────────────────────────────────────────────────
emit("## 1. Raw 15-minute grid (no irradiance filter applied)")
emit("")
pv = pd.read_csv(SQ1_RAW)
pv["Time"] = pd.to_datetime(pv["Time"], errors="coerce")
pv = pv.dropna(subset=["Time"]).sort_values("Time").set_index("Time")
pv = pv[~pv.index.duplicated(keep="first")]


def load_met(folder, col):
    frames = []
    for f in sorted((MET / folder).glob("*.csv")):
        d = pd.read_csv(f)
        d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
        d = d.dropna(subset=["Time"])
        d[col] = pd.to_numeric(d[col], errors="coerce")
        frames.append(d[["Time", col]])
    m = pd.concat(frames, ignore_index=True).sort_values("Time").set_index("Time")
    return m[col].resample("15min").mean()


irr = load_met("Irradiance", "Irradiance (W/m2)")
tmp = load_met("Temperature", "Temp (Degree Celsius)")

grid = pd.DataFrame({"power_W": pv["power(W)"]})
grid = grid.join(irr.rename("irr_meas"), how="left")
grid = grid.join(tmp.rename("temp_C"), how="left")
grid = grid.loc[pv.index.min():pv.index.max()]
emit(f"- Raw PV rows: **{len(pv):,}** ({pv.index.min()} to {pv.index.max()})")
emit(f"- Grid rows after joining meteorological means: **{len(grid):,}**")
emit(f"- Rows with a measured-irradiance value: **{int(grid['irr_meas'].notna().sum()):,}**")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# pvlib quantities under both parameter sets
# ─────────────────────────────────────────────────────────────────────────────
emit("## 2. pvlib parameter correction: effect on `ghi_clear`")
emit("")
idx_local = grid.index.tz_localize("Asia/Hong_Kong", nonexistent="shift_forward",
                                   ambiguous="NaT")
keep = ~idx_local.isna()
grid = grid[keep]
idx_local = idx_local[keep]

clear = {}
zen = {}
for key, P in [("hist", PARAMS_HIST), ("ttl", PARAMS_TTL)]:
    site = Location(latitude=P["latitude"], longitude=P["longitude"],
                    altitude=P["altitude"], tz=P["tz"])
    cs = site.get_clearsky(idx_local, model="ineichen")
    sp = site.get_solarposition(idx_local)
    clear[key] = cs["ghi"].to_numpy()
    zen[key] = sp["zenith"].to_numpy()

grid["ghi_clear_hist"] = clear["hist"]
grid["ghi_clear_ttl"] = clear["ttl"]
grid["zenith_ttl"] = zen["ttl"]
grid["zenith_hist"] = zen["hist"]

day = grid["ghi_clear_ttl"] > 0
d = grid.loc[day, "ghi_clear_ttl"] - grid.loc[day, "ghi_clear_hist"]
rel = 100 * d / grid.loc[day, "ghi_clear_hist"].replace(0, np.nan)
zd = (grid.loc[day, "zenith_ttl"] - grid.loc[day, "zenith_hist"]).abs()

emit("| quantity | mean | max abs | mean % | max % |")
emit("|---|---|---|---|---|")
emit(f"| ghi_clear difference (TTL − historical), W/m² | {d.mean():+.3f} | "
     f"{d.abs().max():.3f} | {rel.mean():+.4f}% | {rel.abs().max():.4f}% |")
emit(f"| solar zenith difference, degrees | {zd.mean():.4f} | {zd.max():.4f} | – | – |")
emit("")
emit(f"Evaluated on {int(day.sum()):,} rows with non-zero clear-sky GHI.")
emit("")
param_rows = [{
    "quantity": "ghi_clear (W/m2)",
    "mean_diff": round(float(d.mean()), 4),
    "max_abs_diff": round(float(d.abs().max()), 4),
    "mean_pct": round(float(rel.mean()), 5),
    "max_abs_pct": round(float(rel.abs().max()), 5),
    "n_rows": int(day.sum()),
}, {
    "quantity": "solar_zenith (deg)",
    "mean_diff": round(float(zd.mean()), 5),
    "max_abs_diff": round(float(zd.max()), 5),
    "mean_pct": None, "max_abs_pct": None, "n_rows": int(day.sum()),
}]
pd.DataFrame(param_rows).to_csv(OUT_DIR / "pvlib_parameter_impact.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Candidate eligibility masks
# ─────────────────────────────────────────────────────────────────────────────
emit("## 3. Candidate eligibility rules")
emit("")
g = grid
rules = {
    "HISTORICAL sensor rule (20<=irr<=1200) [contaminated]":
        (g["irr_meas"] >= HIST_MIN_IRR) & (g["irr_meas"] <= HIST_MAX_IRR),
    "ghi_clear >= 20":  g["ghi_clear_ttl"] >= 20,
    "ghi_clear >= 50":  g["ghi_clear_ttl"] >= 50,
    "zenith < 85":      g["zenith_ttl"] < 85,
    "zenith < 80":      g["zenith_ttl"] < 80,
    "ghi_clear >= 50 AND zenith < 85": (g["ghi_clear_ttl"] >= 50) & (g["zenith_ttl"] < 85),
}
t0, t1 = pd.Timestamp(TYPHOON[0], tz="Asia/Hong_Kong"), pd.Timestamp(TYPHOON[1], tz="Asia/Hong_Kong")
in_typhoon = (idx_local >= t0) & (idx_local <= t1)
in_2023 = idx_local.year == 2023

emit("| rule | rows retained | % of grid | 2023 rows | typhoon-window rows | sensor-free? |")
emit("|---|---|---|---|---|---|")
rule_rows = []
for name, m in rules.items():
    m = m.fillna(False).to_numpy()
    free = "no — uses the sensor" if name.startswith("HISTORICAL") else "**yes**"
    emit(f"| {name} | {m.sum():,} | {100*m.sum()/len(g):.1f}% | "
         f"{int((m & in_2023).sum()):,} | {int((m & in_typhoon).sum()):,} | {free} |")
    rule_rows.append({"rule": name, "rows": int(m.sum()),
                      "pct_of_grid": round(100*m.sum()/len(g), 2),
                      "rows_2023": int((m & in_2023).sum()),
                      "rows_typhoon_window": int((m & in_typhoon).sum()),
                      "sensor_free": not name.startswith("HISTORICAL")})
emit("")
emit(f"Typhoon window = {TYPHOON[0]} to {TYPHOON[1]}; total grid rows in that "
     f"window (unfiltered): **{int(in_typhoon.sum()):,}**")
emit("")

# Set differences vs the historical rule
emit("### 3b. Set differences against the historical sensor rule")
emit("")
hist_m = ((g["irr_meas"] >= HIST_MIN_IRR) & (g["irr_meas"] <= HIST_MAX_IRR)).fillna(False).to_numpy()
emit("| candidate | in candidate only | in historical only | in both | "
     "candidate-only rows that are overcast (irr<20) | candidate-only rows with no sensor value |")
emit("|---|---|---|---|---|---|")
diff_rows = []
for name, m in rules.items():
    if name.startswith("HISTORICAL"):
        continue
    m = m.fillna(False).to_numpy()
    only_c = m & ~hist_m
    only_h = hist_m & ~m
    both = m & hist_m
    irrv = g["irr_meas"].to_numpy()
    overcast = int((only_c & (irrv < HIST_MIN_IRR)).sum())
    nosensor = int((only_c & ~np.isfinite(irrv)).sum())
    emit(f"| {name} | {only_c.sum():,} | {only_h.sum():,} | {both.sum():,} | "
         f"{overcast:,} | {nosensor:,} |")
    diff_rows.append({"candidate_rule": name, "candidate_only": int(only_c.sum()),
                      "historical_only": int(only_h.sum()), "both": int(both.sum()),
                      "candidate_only_overcast_irr_lt_20": overcast,
                      "candidate_only_missing_sensor": nosensor})
pd.DataFrame(rule_rows).to_csv(OUT_DIR / "daylight_rule_comparison.csv", index=False)
pd.DataFrame(diff_rows).to_csv(OUT_DIR / "daylight_rule_setdiff.csv", index=False)
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# Sensor fault window and clean-sensor 2023 subset
# ─────────────────────────────────────────────────────────────────────────────
emit("## 4. Sensor fault window and the clean-sensor 2023 subset")
emit("")
g2 = g.copy()
g2.index = idx_local
m2023 = g2[g2.index.year == 2023]
monthly = m2023.groupby(m2023.index.month)["irr_meas"].agg(["max", "count"])
emit("| month 2023 | max measured irradiance (W/m²) | exceeds 1200? | rows |")
emit("|---|---|---|---|")
clean_months, faulty_months = [], []
for mth, r in monthly.iterrows():
    bad = r["max"] > HIST_MAX_IRR
    (faulty_months if bad else clean_months).append(mth)
    emit(f"| {mth:02d} | {r['max']:.1f} | {'**YES**' if bad else 'no'} | {int(r['count']):,} |")
emit("")
emit(f"- Months with physically impossible readings: **{faulty_months}**")
emit(f"- Clean-sensor months in 2023: **{clean_months}**")
clean_mask = m2023.index.month.isin(clean_months)
emit(f"- Clean-sensor 2023 rows (all-hours grid): **{int(clean_mask.sum()):,}** "
     f"of {len(m2023):,} ({100*clean_mask.sum()/len(m2023):.1f}%)")
cs_rule = (m2023["ghi_clear_ttl"] >= 50) & (m2023["zenith_ttl"] < 85)
emit(f"- Clean-sensor 2023 rows under the proposed sensor-free rule "
     f"(ghi_clear>=50 & zenith<85): **{int((clean_mask & cs_rule).sum()):,}**")
emit(f"- Faulty-window 2023 rows under the same rule: "
     f"**{int((~clean_mask & cs_rule).sum()):,}**")
emit("")
emit(f"The fault window covers {len(faulty_months)} of 12 months of 2023 — a "
     f"substantial portion of the test period, but **not** the whole of it.")
emit("")

pd.DataFrame({"month": monthly.index,
              "max_irr": monthly["max"].round(1).values,
              "rows": monthly["count"].values,
              "sensor_fault": [m in faulty_months for m in monthly.index]}
             ).to_csv(OUT_DIR / "sensor_fault_months_2023.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Typhoon window composition under the proposed rule
# ─────────────────────────────────────────────────────────────────────────────
emit("## 5. Typhoon window composition")
emit("")
tw = g2[(g2.index >= t0) & (g2.index <= t1)]
tw_hist = ((tw["irr_meas"] >= HIST_MIN_IRR) & (tw["irr_meas"] <= HIST_MAX_IRR)).fillna(False)
tw_free = (tw["ghi_clear_ttl"] >= 50) & (tw["zenith_ttl"] < 85)
emit(f"- Grid rows in the window: **{len(tw):,}**")
emit(f"- Retained by the historical sensor rule: **{int(tw_hist.sum()):,}**")
emit(f"- Retained by the proposed sensor-free rule: **{int(tw_free.sum()):,}**")
emit(f"- Admitted by the sensor-free rule but excluded by the sensor rule: "
     f"**{int((tw_free & ~tw_hist).sum()):,}**")
emit(f"- Of those, rows whose measured irradiance was below 20 W/m² (overcast "
     f"or storm-darkened): **{int((tw_free & ~tw_hist & (tw['irr_meas'] < 20)).sum()):,}**")
emit("")
emit("These are exactly the rows a storm suppresses. Under the historical rule "
     "they were deleted before the model ever saw them; under a sensor-free "
     "rule they are retained, which is the point of the correction.")

(OUT_DIR / "hk_design_checks_report.md").write_text("\n".join(_rep) + "\n")
print(f"\nSaved to: {OUT_DIR}")
with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"hk_design_checks.py | input: raw SQ1.csv + raw meteorological CSVs "
            f"(read-only) | output: 5 files in outputs/objective2_design/ | "
            f"notes: design computations only; no training, nothing modified\n")
print(f"Run log appended: {RUN_LOG}")
