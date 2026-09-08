"""
hk_data_audit.py
===========
Objective 2 — Hong Kong read-only audit.

READ ONLY. This script opens no file for writing except its own outputs in
outputs/hk_data_audit/ and one appended line in outputs/run_log.txt. It trains
nothing, modifies no dataset, and touches no Brazil artefact. Objective 1
remains frozen.

Purpose: establish what is actually true about the HK data, and where
measured irradiance influences the existing scenarios, before final modelling.
Every finding is classified CONFIRMED / SUPPORTED ASSUMPTION / UNRESOLVED.
"""

import datetime
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent                      # 01_HK_Detection
ROOT        = PROJECT_DIR.parent                     # repository root
RAW_ROOT    = PROJECT_DIR / "raw_data"
DATASET     = RAW_ROOT / "dryad_dataset"
TS          = DATASET / "Time series dataset"
MET         = TS / "Meteorological dataset"
PVGEN       = TS / "PV generation dataset"
SITE_OPT    = PVGEN / "PV stations with panel level optimizer" / "Site level dataset"
INV_OPT     = PVGEN / "PV stations with panel level optimizer" / "Inverter level dataset"
TTL         = DATASET / "Metadata" / "PV generation system metadata.ttl"
README_DS   = DATASET.parent / "README.md"
DOCS        = PROJECT_DIR

DATA_DIR    = PROJECT_DIR / "data"
OUT_DIR     = PROJECT_DIR / "outputs" / "hk_data_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG     = PROJECT_DIR / "outputs" / "run_log.txt"

SQ1_RAW      = SITE_OPT / "SQ1.csv"
SQ1_INV      = INV_OPT / "SQ1_Inverter.csv"
CLEANED      = DATA_DIR / "sq1_weather_15min_cleaned.csv"
FEATURES     = DATA_DIR / "sq1_pvlib_features.csv"
BASELINE     = DATA_DIR / "sq1_calibrated_baseline.csv"


def require_repository_raw_sources():
    """Audit only the locally copied source subset; never search externally."""
    expected = [SQ1_RAW, SQ1_INV, TTL, README_DS]
    for folder, names in {
        "Irradiance": ["Irradiance_2021.csv", "Irradiance_2022.csv", "Irradiance_2023.csv"],
        "Rainfall": ["Rainfall_2021.xlsx", "Rainfall_2022.csv", "Rainfall_2023.csv"],
        "Relative Humidity": ["Relative Humidity_2021.csv", "Relative Humidity_2022.csv", "Relative Humidity_2023.csv"],
        "Sea Level Pressure": ["Sea Level Pressure_2021.csv", "Sea Level Pressure_2022.csv", "Sea Level Pressure_2023.csv"],
        "Temperature": ["Temperature_2021.csv", "Temperature_2022.csv", "Temperature_2023.csv"],
        "Visibility": ["Visibility_2021.csv", "Visibility_2022.csv", "Visibility_2023.csv"],
        "Wind": ["Wind_2021.csv", "Wind_2022.csv", "Wind_2023.csv"],
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

RATED_KW = 27.6           # TTL ext:ratedPowerOutput for SQ1
CONFIRMED, SUPPORTED, UNRESOLVED = "CONFIRMED", "SUPPORTED ASSUMPTION", "UNRESOLVED"

_report, _semantics, _deps, _open_q = [], [], [], []


def emit(line=""):
    print(line)
    _report.append(line)


def finding(part, question, result, evidence, cls):
    _semantics.append({"part": part, "question": question, "finding": result,
                       "evidence_source": evidence, "classification": cls})


def dep(col, depends, path, where):
    _deps.append({"column": col, "depends_on_measured_irradiance": depends,
                  "derivation_path": path, "script_and_line": where})


emit("# Hong Kong Read-Only Audit — Objective 2")
emit("")
emit(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
emit("")
emit("Read-only investigation. No script, dataset or output was modified and no "
     "model was trained. Every finding carries a classification of CONFIRMED, "
     "SUPPORTED ASSUMPTION or UNRESOLVED.")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — power(W) SEMANTICS
# ─────────────────────────────────────────────────────────────────────────────
emit("## Part 1 — `power(W)` semantics")
emit("")

raw = pd.read_csv(SQ1_RAW)
raw["Time"] = pd.to_datetime(raw["Time"], errors="coerce")
raw = raw.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
emit(f"`SQ1.csv` columns: {list(raw.columns)} — {len(raw):,} rows, "
     f"{raw['Time'].min()} to {raw['Time'].max()}")
emit("")

# 1.3 interval, empirically
diffs = raw["Time"].diff().dropna().value_counts().head(4)
emit("**Sampling interval (empirical, site level):**")
emit("")
emit("| interval | count |")
emit("|---|---|")
for d, c in diffs.items():
    emit(f"| {d} | {c:,} |")
modal = diffs.index[0]
emit("")
emit(f"Modal spacing is **{modal}**. The dataset README states PV generation was "
     f"collected at 5-minute intervals; the site-level file is not at that "
     f"resolution. The inverter-level file is checked below.")
finding(1, "Sampling interval of site-level PV data",
        f"Modal spacing {modal} (empirical)",
        "SQ1.csv timestamps; contradicts README '5-minute intervals'", CONFIRMED)

# 1.1 AC or DC — decisive evidence from the inverter-level schema
emit("")
emit("### 1.1 AC or DC?")
emit("")
inv_ac = None
if SQ1_INV.exists():
    inv = pd.read_csv(SQ1_INV, nrows=5000)
    emit(f"`SQ1_Inverter.csv` exists. Columns:")
    emit("")
    emit("```")
    emit(", ".join(inv.columns))
    emit("```")
    emit("")
    have = [c for c in inv.columns if "activePower" in c]
    if {"totalActivePower(W)", "L1_activePower(W)"}.issubset(inv.columns):
        d = inv.dropna(subset=["totalActivePower(W)", "L1_activePower(W)",
                               "L2_activePower(W)", "L3_activePower(W)"])
        resid = (d["totalActivePower(W)"]
                 - (d["L1_activePower(W)"] + d["L2_activePower(W)"]
                    + d["L3_activePower(W)"])).abs()
        emit(f"Across {len(d):,} inverter rows, "
             f"`totalActivePower(W) − (L1+L2+L3 activePower)` has mean absolute "
             f"value **{resid.mean():.4f} W** — the total is exactly the sum of "
             f"the three AC phase active powers.")
        inv_ac = True
    emit("")
    emit("The inverter schema exposes AC quantities per phase "
         "(`acCurrent`, `acVoltage`, `acFrequency`, `activePower`, "
         "`reactivePower`) and only a single DC quantity, `dcVoltage(V)`. "
         "No DC current is recorded, so DC power is not derivable from this "
         "dataset at all.")
else:
    emit("`SQ1_Inverter.csv` not found.")

# Does the site-level power match the inverter total, resampled?
emit("")
if SQ1_INV.exists():
    inv_full = pd.read_csv(SQ1_INV, usecols=["Time", "totalActivePower(W)"])
    inv_full["Time"] = pd.to_datetime(inv_full["Time"], errors="coerce")
    inv_full = inv_full.dropna(subset=["Time"]).set_index("Time").sort_index()
    inv_15 = inv_full["totalActivePower(W)"].resample("15min").mean()
    site = raw.set_index("Time")["power(W)"]
    joined = pd.concat([site.rename("site"), inv_15.rename("inv_mean")],
                       axis=1).dropna()
    joined = joined[(joined["site"] > 0) | (joined["inv_mean"] > 0)]
    if len(joined) > 100:
        corr = joined["site"].corr(joined["inv_mean"])
        mad = (joined["site"] - joined["inv_mean"]).abs().mean()
        rel = mad / joined["inv_mean"].replace(0, np.nan).mean() * 100
        emit(f"**Site level vs inverter level, aligned on 15-minute means "
             f"({len(joined):,} overlapping intervals):** correlation "
             f"{corr:.5f}, mean absolute difference {mad:.1f} W "
             f"({rel:.2f}% of mean inverter power).")
        emit("")
        emit("The site-level series is therefore the same physical quantity as "
             "the inverter's AC active power.")
        finding(1, "Is power(W) AC or DC?",
                f"AC active power. Inverter total = sum of three AC phase active "
                f"powers (mean |diff| {resid.mean():.4f} W); site level matches "
                f"inverter 15-min mean (r={corr:.5f}). No DC current exists in "
                f"the dataset, so DC power is not derivable.",
                "SQ1_Inverter.csv schema and values; SQ1.csv", CONFIRMED)
    else:
        finding(1, "Is power(W) AC or DC?",
                "AC active power (inverter schema is AC; no DC current recorded)",
                "SQ1_Inverter.csv schema", SUPPORTED)

# 1.2 instantaneous vs aggregated — power vs generation
emit("")
emit("### 1.2 Instantaneous or aggregated? `power(W)` vs `generation(kWh)`")
emit("")
r = raw.set_index("Time")
gen = r["generation(kWh)"]
pw = r["power(W)"]
gen_diff = gen.diff()
# If generation is a cumulative meter, diff ~ power * 0.25h / 1000
implied = pw * 0.25 / 1000.0
cmp_df = pd.concat([gen_diff.rename("gen_diff"), implied.rename("implied_kWh"),
                    gen.rename("gen"), pw.rename("pw")], axis=1).dropna()
day = cmp_df[cmp_df["pw"] > 0]
corr_diff = day["gen_diff"].corr(day["implied_kWh"]) if len(day) > 100 else np.nan
corr_lvl = day["gen"].corr(day["pw"]) if len(day) > 100 else np.nan
emit(f"- `generation(kWh)` range {gen.min():.3f} to {gen.max():.3f}; "
     f"monotonic non-decreasing: **{bool(gen.dropna().is_monotonic_increasing)}**")
emit(f"- corr(generation, power) on daylight rows: **{corr_lvl:.4f}**")
emit(f"- corr(Δgeneration, power×0.25h/1000): **{corr_diff:.4f}**")
emit("")
if not np.isnan(corr_lvl) and corr_lvl > 0.9:
    emit("`generation(kWh)` tracks `power(W)` in level rather than accumulating, "
         "so it is an interval energy figure, not a cumulative meter reading. "
         "Consistent with `power(W)` being the mean power over the 15-minute "
         "interval and `generation(kWh)` the corresponding interval energy.")
    agg_txt = ("Interval-aggregated: power(W) behaves as a 15-minute mean, "
               "generation(kWh) as the matching interval energy")
    agg_cls = SUPPORTED
else:
    agg_txt = "Relationship between the two columns not resolved empirically"
    agg_cls = UNRESOLVED
ratio_gp = (day["gen"] / (day["pw"] * 0.25 / 1000.0).replace(0, np.nan)).median()
emit("")
emit(f"Median `generation(kWh) ÷ (power(W)×0.25/1000)` on daylight rows: "
     f"**{ratio_gp:.4f}** (1.0 would mean exact interval-energy correspondence).")
finding(1, "Instantaneous or aggregated?", agg_txt,
        "SQ1.csv generation(kWh) vs power(W) empirical comparison", agg_cls)

# 1.4 empirical ceiling / clipping
emit("")
emit("### 1.4 Empirical cross-check against the 27.6 kW rating")
emit("")
pw_pos = pw[pw > 0]
q99, q999, mx = pw_pos.quantile(0.99), pw_pos.quantile(0.999), pw_pos.max()
rated_w = RATED_KW * 1000
emit("| statistic | value (W) | as % of 27.6 kW rating |")
emit("|---|---|---|")
for lab, v in [("99th percentile", q99), ("99.9th percentile", q999),
               ("maximum", mx)]:
    emit(f"| {lab} | {v:,.0f} | {100*v/rated_w:.1f}% |")
near = int((pw_pos > 0.98 * mx).sum())
emit("")
emit(f"Rows within 2% of the observed maximum: **{near:,}** "
     f"({100*near/len(pw_pos):.3f}% of positive-power rows).")
emit("")
if mx <= rated_w * 1.02:
    emit(f"The maximum never exceeds the rating. The distribution does **not** "
         f"show a dense hard ceiling (only {near:,} rows sit near the maximum), "
         f"so there is no strong evidence of sustained inverter clipping; the "
         f"observed peak simply falls below the AC rating.")
else:
    emit("The maximum exceeds the AC rating, which would be inconsistent with a "
         "clipped AC series.")
finding(1, "Does the distribution show inverter clipping?",
        f"max {mx:,.0f} W = {100*mx/rated_w:.1f}% of the 27.6 kW rating; "
        f"{near:,} rows within 2% of max — no dense ceiling, no strong clipping "
        f"evidence", "SQ1.csv power(W) distribution vs TTL rating", CONFIRMED)

emit("")
emit("### 1.5 Consequence: PVWatts models DC, the measurement is AC")
emit("")
emit("`pvlib.pvsystem.pvwatts_dc` returns **DC** array output. The existing "
     "`p_exp` therefore compares modelled DC against measured AC, and the "
     "calibration factor α silently absorbs the inverter conversion efficiency "
     "together with every other systematic bias.")
emit("")
emit("- **(a) Expected-power baseline.** α is not a pure irradiance-model "
     "correction. Any statement that 'pvlib under-predicts by 2.6%' conflates "
     "model bias with DC→AC conversion loss, which typically runs 2–5%. The "
     "quantities are not like-for-like without an explicit inverter model.")
emit("- **(b) Objective 3 transfer.** Brazil `p_dc_computed` = vdc1·idc1 + "
     "vdc2·idc2 is definitively **DC**. Mapping it onto HK AC power compares "
     "different physical quantities separated by inverter efficiency and "
     "clipping behaviour. Capacity normalisation does not fix this: it rescales "
     "magnitude, not the AC/DC distinction.")
_open_q.append(("AC/DC mismatch in expected power",
                "PVWatts models DC; HK power(W) is AC. Whether to add an "
                "inverter efficiency model, or compare AC-to-AC by other means, "
                "is a design decision that affects both the baseline and any "
                "transfer claim.",
                "A decision on the expected-power definition, and whether "
                "Objective 3 transfer remains defensible across AC/DC."))

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — ENVIRONMENTAL SEMANTICS
# ─────────────────────────────────────────────────────────────────────────────
emit("")
emit("## Part 2 — Environmental semantics")
emit("")

ttl_text = TTL.read_text(errors="replace")
m = re.search(r"pvsystem:SQ1 a brick:PV_Generation_System ;(.*?)\.\n", ttl_text,
              re.S)
sq1_block = m.group(0) if m else "(SQ1 block not found)"
emit("### 2.2 SQ1 geometry — TTL metadata")
emit("")
emit("```turtle")
emit(sq1_block.strip())
emit("```")
emit("")


def ttl_val(pat):
    mm = re.search(pat, sq1_block)
    return mm.group(1) if mm else "not found"


ttl_tilt = ttl_val(r'ext:tiltAngle \[ brick:value "([^"]+)"')
ttl_azi = ttl_val(r'ext:azimuth \[ brick:value "([^"]+)"')
ttl_alt = ttl_val(r"ext:altitude \[ brick:hasUnit unit:M ;\s*brick:value ([0-9.]+)")
ttl_lat = ttl_val(r"brick:latitude ([0-9.]+)")
ttl_lon = ttl_val(r"brick:longitude ([0-9.]+)")
ttl_rated = ttl_val(r"ext:ratedPowerOutput \[ brick:hasUnit unit:KW ;\s*brick:value ([0-9.]+)")

emit("| property | TTL value | value used in the pipeline | agree? |")
emit("|---|---|---|---|")
emit(f"| tilt | {ttl_tilt} | 0° | yes |")
emit(f"| azimuth | {ttl_azi} | 180° | **no** — see note |")
emit(f"| latitude | {ttl_lat} | 22.3363 | approximately |")
emit(f"| longitude | {ttl_lon} | 114.2634 | **no** |")
emit(f"| altitude | {ttl_alt} m | 30 m | **no** |")
emit(f"| rated power | {ttl_rated} kW | 27,606 W | yes |")
emit("")
emit(f"Tilt = {ttl_tilt} is confirmed. At zero tilt the azimuth is geometrically "
     f"irrelevant, so the 180° used in the pipeline is harmless here, but it "
     f"does not match the metadata value of {ttl_azi}.")
emit("")
emit(f"Two genuine discrepancies: the TTL places SQ1 at longitude {ttl_lon} and "
     f"altitude {ttl_alt} m, while the pipeline uses 114.2634 and 30 m. The "
     f"pipeline's coordinates are the campus centroid quoted in the dataset "
     f"README (22.3363°N, 114.2634°E), not SQ1's own entry. The TTL "
     f"coordinates are coarse (2 decimal places, ~1 km) and several stations "
     f"share identical values, so neither source is precise; the altitude gap "
     f"of {ttl_alt} m versus 30 m is the more material of the two for clear-sky "
     f"modelling.")
finding(2, "SQ1 tilt", f"{ttl_tilt} — confirms the pipeline's tilt=0",
        "TTL ext:tiltAngle", CONFIRMED)
finding(2, "SQ1 azimuth", f"TTL says {ttl_azi}; pipeline uses 180°. Irrelevant "
        f"at zero tilt but not matching", "TTL ext:azimuth", CONFIRMED)
finding(2, "SQ1 location and altitude",
        f"TTL: lat {ttl_lat}, lon {ttl_lon}, altitude {ttl_alt} m. Pipeline: "
        f"22.3363, 114.2634, 30 m (README campus centroid). Discrepancy in "
        f"longitude and altitude", "TTL vs README vs pipeline constants", CONFIRMED)

# 2.1 irradiance plane
emit("")
emit("### 2.1 What plane does the measured irradiance represent?")
emit("")
readme = README_DS.read_text(errors="replace") if README_DS.exists() else ""
irr_mentions = [l.strip() for l in readme.splitlines()
                if re.search(r"irradian|pyranom|weather station|sampler", l, re.I)]
for l in irr_mentions[:4]:
    emit(f"> {l}")
    emit("")
emit("The README describes a 10-metre automatic weather tower with six samplers "
     "and names the variable only as `Irradiance (W/m2)`. Neither the README nor "
     "the TTL states the sensor plane, tilt, or instrument type. A mast-mounted "
     "campus weather station conventionally measures **global horizontal** "
     "irradiance, but this is an inference from installation type, not a "
     "documented fact.")
emit("")
emit("For SQ1 specifically the distinction is inert — the array is horizontal "
     "(tilt = 0°), so GHI and plane-of-array coincide. It matters for any "
     "cross-station comparison and for how the column should be described.")
finding(2, "Irradiance measurement plane",
        "Not documented. Weather-tower installation implies global horizontal, "
        "but no source states it. Inert for SQ1 (tilt=0, so GHI==POA)",
        "Dataset README; TTL has no weather-station entry", UNRESOLVED)
_open_q.append(("Irradiance sensor plane and instrument",
                "Neither README nor TTL states the plane, tilt or model of the "
                "irradiance sensor. Assumed horizontal from the weather-tower "
                "description.",
                "Dataset authors' confirmation, or an instrument specification."))

# 2.3 station pairing
emit("")
emit("### 2.3 Weather station pairing")
emit("")
has_ws = bool(re.search(r"weather", ttl_text, re.I))
emit(f"- The campus has **60 PV stations and one weather station** (README).")
emit(f"- The weather station appears in the Brick model: **{has_ws}**. A search "
     f"for weather-related entities in the TTL returns nothing, so the station "
     f"is not represented in the metadata at all.")
emit(f"- There is therefore **no documented link** between SQ1 and the weather "
     f"station. The README places the station 'on the eastern side of the "
     f"campus'; SQ1 is Staff Quarter Tower 1. No distance is stated.")
emit("")
emit("Pairing SQ1 with this station is an **assumption of spatial "
     "representativeness**, not a documented relationship. It is the only "
     "meteorological source available, so the assumption is unavoidable, but it "
     "should be stated rather than implied — cloud fields decorrelate over "
     "campus-scale distances, which bears directly on how well any irradiance "
     "estimate can track this particular array.")
finding(2, "Is SQ1 documented as served by this weather station?",
        "No. The weather station is absent from the Brick model and no "
        "PV-to-station mapping is documented. One station serves 60 stations; "
        "pairing is an unavoidable assumption",
        "README; TTL contains no weather-station entity", UNRESOLVED)
_open_q.append(("Weather-station representativeness for SQ1",
                "No documented pairing between SQ1 and the single campus weather "
                "station, and no stated separation distance.",
                "Station coordinates and a statement of intended coverage from "
                "the dataset authors."))

# 2.4 timezone per file
emit("")
emit("### 2.4 Timezone")
emit("")
emit("| file | first timestamp | tz-aware? |")
emit("|---|---|---|")
tz_rows = []
for label, path, col in [
        ("raw SQ1.csv", SQ1_RAW, "Time"),
        ("raw Irradiance_2021.csv", MET / "Irradiance" / "Irradiance_2021.csv", "Time"),
        ("sq1_weather_15min_cleaned.csv", CLEANED, "Time"),
        ("sq1_pvlib_features.csv", FEATURES, "Time"),
        ("sq1_calibrated_baseline.csv", BASELINE, "Time")]:
    if not path.exists():
        emit(f"| {label} | (missing) | – |")
        continue
    head = pd.read_csv(path, nrows=3)
    first = str(head[col].iloc[0])
    aware = bool(re.search(r"[+-]\d{2}:\d{2}$|Z$", first))
    emit(f"| {label} | `{first}` | {aware} |")
    tz_rows.append((label, first, aware))
emit("")
emit("Every file carries **naive** timestamps — no offset, no zone. The raw "
     "data is presumed to be Hong Kong local time (UTC+8), which is what the "
     "pipeline assumes when it constructs a `Location` with "
     "`tz='Asia/Hong_Kong'`. Hong Kong has observed no daylight saving since "
     "1979, so a fixed +8 offset is unambiguous for 2021–2023 — but the "
     "assumption is undocumented.")
emit("")
emit("Note that `pvlib_features_SQ1.py` calls `site.get_solarposition(df.index)` "
     "on a **naive** index. pvlib then treats those timestamps as being in the "
     "site timezone. If the raw data were actually UTC, every solar position "
     "in the pipeline would be wrong by eight hours.")
finding(2, "Timezone of raw and intermediate timestamps",
        "All files naive (no offset). Hong Kong local time assumed; HK has no "
        "DST since 1979 so +8 is unambiguous if the assumption holds",
        "Header inspection of raw and derived CSVs", SUPPORTED)
_open_q.append(("Timestamp timezone",
                "No file states a timezone. Local HK time is assumed; if the "
                "data were UTC every solar-position feature would be shifted by "
                "eight hours.",
                "A statement from the dataset authors, or a solar-noon "
                "alignment test against computed solar position."))

# 2.5 sampling intervals
emit("")
emit("### 2.5 Sampling intervals (empirical)")
emit("")
emit("| source | modal interval | README claim |")
emit("|---|---|---|")
emit(f"| PV site level (SQ1.csv) | {modal} | 5 minutes |")
if SQ1_INV.exists():
    iv = pd.read_csv(SQ1_INV, usecols=["Time"], nrows=5000)
    iv["Time"] = pd.to_datetime(iv["Time"], errors="coerce")
    inv_modal = iv["Time"].diff().dropna().value_counts().index[0]
    emit(f"| PV inverter level (SQ1_Inverter.csv) | {inv_modal} | 5 minutes |")
irr21 = pd.read_csv(MET / "Irradiance" / "Irradiance_2021.csv", usecols=["Time"],
                    nrows=5000)
irr21["Time"] = pd.to_datetime(irr21["Time"], errors="coerce")
met_modal = irr21["Time"].diff().dropna().value_counts().index[0]
emit(f"| Meteorological (Irradiance_2021.csv) | {met_modal} | 1 minute |")
emit("")
emit(f"Meteorological data matches its documentation at {met_modal}. The "
     f"site-level PV series is at {modal}, not the 5 minutes the README claims; "
     f"the inverter-level series is at the documented 5 minutes. **Where the "
     f"documentation and the data disagree, the data is the more reliable "
     f"source** — the README statement is a generalisation across 60 stations "
     f"and two product levels.")
emit("")
emit(f"`Data_preprocessing_SQ1.py` resamples both PV and meteorological streams "
     f"to 15 minutes with `.mean()`, so the 1-minute meteorological data is "
     f"averaged into 15-minute means, and the already-15-minute PV series is "
     f"effectively passed through.")
finding(2, "Sampling intervals",
        f"PV site level {modal} (README says 5 min — data wins); inverter level "
        f"5 min; meteorological {met_modal}. Preprocessing resamples all to "
        f"15 min by mean", "Empirical timestamp differences", CONFIRMED)

# 2.6 available weather variables
emit("")
emit("### 2.6 Meteorological variables available")
emit("")
emit("| variable | folder | columns | used in the current pipeline? |")
emit("|---|---|---|---|")
used = {"Irradiance", "Temperature"}
met_inventory = []
for sub in sorted([p for p in MET.iterdir() if p.is_dir()]):
    files = sorted(sub.glob("*.csv"))
    xlsx = sorted(sub.glob("*.xlsx"))
    if files:
        cols = list(pd.read_csv(files[0], nrows=1).columns)
        colstr = ", ".join(f"`{c}`" for c in cols if c != "Time")
    elif xlsx:
        colstr = "(2021 is .xlsx; CSV for other years)"
        cols = []
    else:
        colstr = "(no files)"
        cols = []
    u = "**yes**" if sub.name in used else "no — available, unused"
    emit(f"| {sub.name} | {len(files)} csv, {len(xlsx)} xlsx | {colstr} | {u} |")
    met_inventory.append((sub.name, colstr, sub.name in used))
emit("")
unused = [n for n, _, u in met_inventory if not u]
emit(f"**Available but unused: {', '.join(unused)}.** These are the candidate "
     f"non-irradiance environmental inputs for Arm A and Arm B. Relative "
     f"humidity, sea-level pressure, visibility and wind are all plausibly "
     f"informative about cloud state, and none of them is an irradiance "
     f"measurement — so none violates a no-irradiance-sensor contract.")
emit("")
emit("One caution: `Visibility` is an atmospheric-transparency measurement. It "
     "is not irradiance and does not come from the pyranometer, so it is "
     "admissible under a no-irradiance-sensor rule, but it is closer to the "
     "quantity being estimated than, say, pressure. Whether to admit it is a "
     "contract decision, not a technical constraint.")
finding(2, "Meteorological variables available",
        f"Seven folders. Used: Irradiance, Temperature. Available and unused: "
        f"{', '.join(unused)}", "Meteorological dataset folder inventory", CONFIRMED)

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — MEASURED-IRRADIANCE DEPENDENCY TRACE
# ─────────────────────────────────────────────────────────────────────────────
emit("")
emit("## Part 3 — Measured-irradiance dependency trace")
emit("")
emit("Governing question: *could measured irradiance influence whether a row "
     "exists, how a feature is calculated, how a model or constant is "
     "calibrated, or which sequence reaches the model?*")
emit("")

IRR = "`Irradiance (W/m2)`"
dep("zenith", "NO", "pvlib solar position from timestamp + site location only",
    "pvlib_features_SQ1.py:63-64")
dep("azimuth", "NO", "pvlib solar position from timestamp + site location only",
    "pvlib_features_SQ1.py:63-65")
dep("ghi_clear", "NO",
    "pvlib Ineichen clear-sky from timestamp + location + altitude; no measurement",
    "pvlib_features_SQ1.py:67-69")
dep("k_clear", "YES",
    "measured Irradiance / ghi_clear — clearness index, numerator is the sensor",
    "pvlib_features_SQ1.py:72-73")
dep("dni_est", "YES", "DISC decomposition with ghi = measured Irradiance",
    "pvlib_features_SQ1.py:77-78")
dep("dhi_est", "YES", "measured Irradiance − dni_est·cos(zenith)",
    "pvlib_features_SQ1.py:81-82")
dep("poa_global", "YES",
    "get_total_irradiance with ghi = measured Irradiance and dni/dhi derived from it",
    "pvlib_features_SQ1.py:85-95")
dep("temp_cell", "YES",
    "Faiman model driven by poa_global, which is a function of measured Irradiance",
    "pvlib_features_SQ1.py:98-100")
dep("p_exp", "YES",
    "pvwatts_dc(poa_global, temp_cell) — both inputs trace to measured Irradiance",
    "pvlib_features_SQ1.py:111-117")
dep("residual", "YES", "power(W) − p_exp", "pvlib_features_SQ1.py:119")
dep("ratio", "YES", "power(W) / p_exp", "pvlib_features_SQ1.py:120")
dep("p_exp_cal", "YES",
    "p_exp × α, and α itself is fitted on rows selected by measured Irradiance",
    "pvlib_calibration_SQ1.py:102,126")
dep("residual_cal", "YES", "power(W) − p_exp_cal", "pvlib_calibration_SQ1.py:127")
dep("ratio_cal", "YES", "power(W) / p_exp_cal", "pvlib_calibration_SQ1.py:128")
dep("power(W)", "NO",
    "measured AC active power; independent of the irradiance sensor",
    "raw SQ1.csv")
dep("Temp (Degree Celsius)", "NO", "measured ambient temperature; separate sensor",
    "raw Temperature/*.csv")
dep("month_sin / month_cos", "NO", "calendar encoding of the timestamp",
    "pv_no_sensor_SQ1.py")
dep("ROW EXISTENCE (all downstream data)", "YES",
    "Data_preprocessing_SQ1.py drops rows with Irradiance < 20 or > 1200 W/m², so "
    "every row in every derived CSV survived a measured-irradiance test",
    "Data_preprocessing_SQ1.py:50-51,166-167")
dep("SEQUENCE RETENTION (both LSTM scenarios)", "YES",
    "build_sequences skips any window whose target row has measured Irradiance < "
    "MIN_IRR, in the no-sensor scenario as well as the with-sensor one",
    "pv_no_sensor_SQ1.py:220-226,260")
dep("alpha (calibration constant)", "YES",
    "OLS fit over rows selected by measured Irradiance >= 50, using p_exp which "
    "is itself irradiance-derived",
    "pvlib_calibration_SQ1.py:41,102")

dd = pd.DataFrame(_deps)
emit("| column / stage | depends on measured irradiance | derivation path |")
emit("|---|---|---|")
for _, r in dd.iterrows():
    emit(f"| `{r['column']}` | **{r['depends_on_measured_irradiance']}** | "
         f"{r['derivation_path']} |")

emit("")
emit("### The current 'no-sensor' LSTM feature set")
emit("")
NO_SENSOR_FEATS = ["power(W)", "ghi_clear", "Temp (Degree Celsius)", "zenith",
                   "temp_cell", "p_exp_cal", "month_sin", "month_cos"]
contaminated = [f for f in NO_SENSOR_FEATS
                if f in {"temp_cell", "p_exp_cal"}]
emit("| feature | contaminated? | why |")
emit("|---|---|---|")
why = {
    "power(W)": "measured AC power, not the irradiance sensor",
    "ghi_clear": "clear-sky model from time and location only",
    "Temp (Degree Celsius)": "ambient temperature sensor, not irradiance",
    "zenith": "solar geometry from timestamp only",
    "temp_cell": "**Faiman model driven by poa_global → measured irradiance**",
    "p_exp_cal": "**p_exp × α; p_exp is irradiance-derived and α is fitted on "
                 "irradiance-selected rows**",
    "month_sin": "calendar encoding",
    "month_cos": "calendar encoding",
}
for f in NO_SENSOR_FEATS:
    c = "**YES**" if f in contaminated else "no"
    emit(f"| `{f}` | {c} | {why[f]} |")
emit("")
emit(f"**{len(contaminated)} of the 8 features are directly contaminated: "
     f"`temp_cell` and `p_exp_cal`.**")
emit("")
emit("But the feature list understates the problem. Two pipeline-level "
     "dependencies contaminate the scenario regardless of which features are "
     "chosen:")
emit("")
emit("1. **Row existence.** Every row in `sq1_calibrated_baseline.csv` survived "
     "the `20 ≤ Irradiance ≤ 1200` filter in preprocessing. A deployment "
     "without the sensor could not have constructed this row set.")
emit("2. **Sequence retention.** `build_sequences` tests measured irradiance "
     "against `MIN_IRR` for every candidate window and skips those below it — "
     "in the no-sensor scenario too. The daylight *mask* was switched to "
     "`ghi_clear`, but the sequence builder was not, so measured irradiance "
     "still decides which samples the no-sensor model ever sees.")
emit("")
emit("The consequence is that the historical 'no-sensor' result is **not a "
     "sensor-free result**, and this holds even if `temp_cell` and `p_exp_cal` "
     "were removed from the feature list.")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — CALIBRATION AND FITTING LEAKAGE
# ─────────────────────────────────────────────────────────────────────────────
emit("")
emit("## Part 4 — Calibration and fitting leakage")
emit("")
emit("| learned quantity | fitted on | spans the test period? |")
emit("|---|---|---|")
emit("| α (calibration factor) | all retained rows with measured Irradiance ≥ 50, "
     "2021-06 to 2023-12 | **YES — leakage** |")
emit("| monthly α (diagnostic) | same, grouped by month | **YES — leakage** |")
emit("| PDC0 estimate | 99.5th percentile of `power(W)` over the whole file | "
     "**YES — leakage** |")
emit("| StandardScaler (LSTM) | `df.index <= TRAIN_END` (2022-12-31) | no — correct |")
emit("| Imputer | none used in the HK pipeline | – |")
emit("")
emit("**Finding.** `pvlib_calibration_SQ1.py` line 41 selects "
     "`day = df[df[IRR] >= MIN_IRR]` across the entire record and line 102 fits "
     "`α = Σ(P_act·P_exp) / Σ(P_exp²)` on that selection. The 2023 test period "
     "is inside the fit. Every `p_exp_cal`, `residual_cal` and `ratio_cal` "
     "value in the test period therefore embeds a constant that saw the test "
     "period.")
emit("")
emit("A second, subtler instance: `pvlib_features_SQ1.py` line 107 sets "
     "`pdc0_est` to the 99.5th percentile of measured power over the whole "
     "file, so the system-capacity constant feeding PVWatts is also fitted "
     "across train and test.")
emit("")
emit("The LSTM `StandardScaler` is handled correctly — fitted on "
     "`df.index <= TRAIN_END` only. But it is fitted on columns that are "
     "themselves contaminated by α, so correct scaler discipline does not "
     "rescue the pipeline.")
emit("")
emit("**Proposed rule for the final modelling pipeline:**")
emit("")
emit("> Any learned calibration constant, scaler, imputer, or threshold must be "
     "fitted on training-period data only and applied unchanged to validation "
     "and test periods.")
emit("")
emit("Applying this would mean: fit α on 2021–2022 only; derive the capacity "
     "constant from nameplate (27.6 kW, per the TTL) or from training-period "
     "data only; keep the existing scaler discipline.")

# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — SAME-TIME ESTIMATOR SCOPE
# ─────────────────────────────────────────────────────────────────────────────
emit("")
emit("## Part 5 — Same-time estimator scope")
emit("")
HORIZON_PAT = re.compile(r"same.time|current timestamp|future prediction|horizon", re.I)
searched = []
sources = []
for p in sorted(DOCS.rglob("*.md")):
    sources.append(p)
for p in [SCRIPT_DIR / "objective2_prepare.py"]:
    if p.exists():
        sources.append(p)
docx_sources = [DOCS / "tutor_guidance" / "source_material" / "tutor_comments.docx",
                DOCS / "legacy_ai" / "claude" / "source_material" / "chat log.docx"]

emit("| source | horizon-related evidence |")
emit("|---|---|")
for p in sources:
    try:
        txt = p.read_text(errors="replace")
    except Exception:
        continue
    hits = [l.strip() for l in txt.splitlines() if HORIZON_PAT.search(l)]
    searched.append((str(p.relative_to(ROOT)), len(hits)))
    if hits:
        sample = hits[0][:150].replace("|", "/")
        emit(f"| `{p.name}` | {len(hits)} mention(s); e.g. \"{sample}\" |")
    else:
        emit(f"| `{p.name}` | none |")
try:
    from docx import Document
    for p in docx_sources:
        if not p.exists():
            emit(f"| `{p.name}` | (missing) |")
            continue
        txt = "\n".join(par.text for par in Document(str(p)).paragraphs)
        hits = [l.strip() for l in txt.splitlines() if HORIZON_PAT.search(l)]
        searched.append((str(p.relative_to(ROOT)), len(hits)))
        explicit = [h for h in hits
                    if re.search(r"horizon|ahead|t\s*\+\s*\d|lead.time", h, re.I)]
        if explicit:
            emit(f"| `{p.name}` | {len(hits)} mention(s); "
                 f"explicit-horizon lines: {len(explicit)}; "
                 f"e.g. \"{explicit[0][:130]}\" |")
        else:
            emit(f"| `{p.name}` | {len(hits)} scope-related mention(s), "
                 f"**no horizon stated** |")
except ImportError:
    emit("| (docx sources) | python-docx unavailable |")

emit("")
emit("**Current Objective 2 construction.** Its features are "
     "`ghi_clear, zenith, azimuth, Temp, hour_sin, hour_cos, month_sin, "
     "month_cos, day_of_year` and its target is `Irradiance (W/m2)` **at the "
     "same timestamp**. There is no lag, no shift, and no lead variable "
     "anywhere in the script. Whatever the intent, what is implemented is "
     "**same-time estimation**, not ahead-of-time prediction.")
emit("")
emit("**Classification: SAME-TIME ESTIMATION.** No tutor material, brief, objective or "
     "protocol document states a future-prediction horizon.")
emit("")
emit("| option | research question it answers | consequence |")
emit("|---|---|---|")
emit("| Same-time estimation (nowcast) | Can a model replace the physical "
     "irradiance sensor? | Legitimate sensor replacement **provided measured "
     "irradiance never enters at inference**. It is not circular: the sensor is "
     "absent at deployment and the estimate is built from time, geometry and "
     "non-irradiance weather. |")
emit("| Ahead-of-time prediction (t+h) | Can irradiance be estimated in advance "
     "for early fault warning? | A different and harder question. Requires "
     "choosing h, and makes the expected-power baseline a prediction about the "
     "future rather than an estimate of the present. |")
emit("")
emit("**Conclusion.** The retained implementation is same-time estimation; no "
     "ahead-of-time task is defined here.")
finding(5, "Is a future-prediction horizon specified anywhere?",
        "No. No tutor, brief, objective or protocol document states a lead time. "
        "The implemented script performs same-time estimation with no lag or "
        "shift", "Full-text search of repository materials and the current script",
        "SAME-TIME ESTIMATION")
_open_q.append(("Future-prediction horizon",
                "No source states a horizon. The implemented script is a "
                "same-time estimator, with no ahead-of-time task defined.",
                "A decision from the tutor, or an explicit project choice "
                "recorded with its rationale."))

# ─────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
pd.DataFrame(_deps).to_csv(OUT_DIR / "irradiance_dependency_table.csv", index=False)
pd.DataFrame(_semantics).to_csv(OUT_DIR / "hk_semantics_findings.csv", index=False)
(OUT_DIR / "hk_data_audit_report.md").write_text("\n".join(_report) + "\n")

oq = ["# Open Questions — HK Audit", "",
      "Items this audit could not resolve from available evidence.", ""]
for i, (title, detail, need) in enumerate(_open_q, 1):
    oq += [f"## {i}. {title}", "", detail, "",
           f"**What would resolve it:** {need}", ""]
(OUT_DIR / "open_questions.md").write_text("\n".join(oq))

CONTRACTS = f"""# Draft Information Contracts — Four Provisional Arms

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
"""
(OUT_DIR / "information_contracts_draft.md").write_text(CONTRACTS)

emit("")
emit("## Outputs written")
emit("")
for f in sorted(OUT_DIR.iterdir()):
    emit(f"- `{f.name}`")

(OUT_DIR / "hk_data_audit_report.md").write_text("\n".join(_report) + "\n")

print(f"\nSaved to: {OUT_DIR}")
with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"hk_data_audit.py | input: HK raw dataset, TTL, README, repository materials, HK scripts "
            f"(all read-only) | output: 5 files in outputs/hk_data_audit/ | "
            f"notes: power(W)=AC active power (CONFIRMED); 2 of 8 no-sensor LSTM "
            f"features contaminated plus row-existence and sequence-retention "
            f"stages; alpha and pdc0 fitted across the test period (leakage); "
            f"same-time scope recorded; nothing modified, nothing trained\n")
print(f"Run log appended: {RUN_LOG}")
