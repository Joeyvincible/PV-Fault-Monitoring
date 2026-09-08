"""
fault_preprocessing.py
=======================
Preprocessing for the 16-day Brazil PV fault dataset.

SOURCE: github.com/clayton-h-costa/pv_fault_dataset
PAPER:  Lazzaretti et al. (2020), Sensors MDPI, doi:10.3390/s20174688

WHAT CHANGED AND WHY
--------------------
An audit found that the previous revision fabricated a calendar from an
invented reference date and then derived solar-geometry features from it.
The MAT files carry no time variable, so every calendar-dependent feature
was unsupported. All of it has been removed. This script now produces only
features that follow from the measurements themselves.

Removed: fabricated 1 Hz calendar, site location object, solar position,
irradiance decomposition, plane-of-array retransposition, and the sun-angle
columns that depended on them.

MEASURED-POA PHYSICS BASELINE — APPROXIMATE, NOT GROUND TRUTH
-------------------------------------------------------------
The paper (Section 3.2) states the pyranometer was installed "following the
same inclination of the panel installation", so `irr` is already measured in
the plane of the array. It is therefore used DIRECTLY as PVWatts effective
irradiance. It is not decomposed and not transposed again.

The resulting expected-power baseline rests on three approximations, all of
which remain open:

  1. Measured tilted global irradiance is used as PVWatts effective
     irradiance. Optical, incidence-angle, and soiling losses may not be
     represented.
  2. Rear-module temperature (four PT100 sensors, arithmetic mean, paper
     Section 3.2) is used as a proxy for cell temperature. Cell temperature
     is typically higher than rear-surface temperature under load.
  3. System capacity is ambiguous. The paper says the system "may yield 5 kW
     peak installed capacity" while the module count implies 16 x 330 W =
     5,280 W nameplate. Both are computed; neither is asserted as correct.

This baseline is explicitly NOT validated physical ground truth, NOT
sensor-free (it consumes both the pyranometer and the module-temperature
sensors), and NOT a clear-sky baseline.

INPUT  : raw_data/converted_csv/dataset_amb.csv
         raw_data/converted_csv/dataset_elec.csv
OUTPUT : data/fault_dataset.csv
         outputs/source_event_register.csv

Historical files (raw_data/dataset_*.csv, data/fault_dataset_cleaned.csv)
are never read or written by this script.
"""

import datetime
import numpy as np
import pandas as pd
import pvlib
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_DIR / "raw_data"
CONVERTED_DIR = RAW_DATA_DIR / "converted_csv"

AMB_CSV  = CONVERTED_DIR / "dataset_amb.csv"
ELEC_CSV = CONVERTED_DIR / "dataset_elec.csv"

OUT_DIR      = PROJECT_DIR / "data"
OUTPUTS_DIR  = PROJECT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV        = OUT_DIR / "fault_dataset.csv"
EVENT_REGISTER = OUTPUTS_DIR / "source_event_register.csv"
RUN_LOG        = OUTPUTS_DIR / "run_log.txt"

# Capacity scenarios — the paper supports both readings, so both are kept.
PDC0_RATED     = 5000.0   # paper: "may yield 5 kW peak installed capacity"
PDC0_NAMEPLATE = 5280.0   # 16 modules x 330 W
GAMMA_PDC      = -0.0041  # /degC — CS6U-330P datasheet

# Cleaning thresholds
MIN_IRR = 20    # W/m^2 — daylight threshold
MAX_IRR = 1500  # W/m^2 — physical ceiling

LABEL_NAMES = {0: "Normal", 1: "Short-Circuit", 2: "Degradation",
               3: "Open-Circuit", 4: "Shadowing"}

# Reported sample counts, paper Section 4.1
PAPER_COUNTS = {0: 309253, 1: 5999, 2: 10371, 3: 6024, 4: 184311}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD CONVERTED CSVs
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 1: Load converted CSVs ===")
for p in (AMB_CSV, ELEC_CSV):
    if not p.exists():
        raise SystemExit(f"Missing converted input: {p}\n"
                         f"Run mat_to_csv.py first.")

amb  = pd.read_csv(AMB_CSV)
elec = pd.read_csv(ELEC_CSV)
print(f"  amb  shape: {amb.shape}  | columns: {list(amb.columns)}")
print(f"  elec shape: {elec.shape} | columns: {list(elec.columns)}")

if "sample_index" not in amb.columns or "sample_index" not in elec.columns:
    raise SystemExit("Both converted CSVs must carry sample_index.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MERGE ON sample_index
# ─────────────────────────────────────────────────────────────────────────────
# Row alignment rests on the acquisition architecture described in paper
# Section 3.4: a single cRIO logging both signal groups synchronously. It is
# well supported but not independently provable from the released files.
print("\n=== STEP 2: Merge on sample_index ===")
df = amb.merge(elec, on="sample_index", how="inner", validate="one_to_one")
df = df.sort_values("sample_index").reset_index(drop=True)
print(f"  Merged rows: {len(df):,}")
if len(df) != len(amb) or len(df) != len(elec):
    print(f"  WARNING: merge dropped rows — amb {len(amb):,}, "
          f"elec {len(elec):,}, merged {len(df):,}")

df = df.rename(columns={
    "irr": "irr_panel_wm2",      # measured panel-plane global irradiance
    "pvt": "t_module_rear_c",    # rear-module temperature (4x PT100 mean)
})
print(f"  Renamed: irr -> irr_panel_wm2, pvt -> t_module_rear_c")

for c in ["vdc1", "vdc2", "idc1", "idc2", "irr_panel_wm2",
          "t_module_rear_c", "f_nv"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SOURCE EVENT IDs ON THE FULL SEQUENCE, BEFORE ANY FILTERING
# ─────────────────────────────────────────────────────────────────────────────
# Event boundaries belong to the dataset authors' labels. A cleaning rule of
# ours must not be allowed to redefine where a labelled event starts or ends,
# so this runs on every row, before a single one is dropped.
print("\n=== STEP 3: Source event IDs (full sequence, pre-filter) ===")
df["source_event_id"] = (df["f_nv"] != df["f_nv"].shift()).cumsum()

register = (df.groupby("source_event_id")
              .agg(f_nv=("f_nv", "first"),
                   start_sample_index=("sample_index", "min"),
                   end_sample_index=("sample_index", "max"),
                   raw_sample_count=("sample_index", "size"))
              .reset_index())
register.to_csv(EVENT_REGISTER, index=False)

print(f"  Total source events: {len(register):,}")
print(f"  Events per class:")
for lbl in sorted(register["f_nv"].dropna().unique()):
    sub = register[register["f_nv"] == lbl]
    print(f"    {int(lbl)} ({LABEL_NAMES.get(int(lbl), '?'):15s}): "
          f"{len(sub):>6,} events | {sub['raw_sample_count'].sum():>9,} raw samples")
print(f"  Saved: {EVENT_REGISTER}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ELECTRICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 4: Electrical features ===")
# p_dc_computed is derived from the four string measurements (V x I per
# string). It is NOT evidence of an independent power meter — no such
# instrument exists in this dataset.
df["p1"]            = df["vdc1"] * df["idc1"]
df["p2"]            = df["vdc2"] * df["idc2"]
df["p_dc_computed"] = df["p1"] + df["p2"]

df["v_ratio"]     = df["vdc1"] / df["vdc2"].replace(0, np.nan)
df["i_ratio"]     = df["idc1"] / df["idc2"].replace(0, np.nan)
df["v_imbalance"] = (df["vdc1"] - df["vdc2"]).abs() / ((df["vdc1"] + df["vdc2"]) / 2).replace(0, np.nan)
df["i_imbalance"] = (df["idc1"] - df["idc2"]).abs() / ((df["idc1"] + df["idc2"]) / 2).replace(0, np.nan)
print("  Added: p1, p2, p_dc_computed, v_ratio, i_ratio, v_imbalance, i_imbalance")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — APPROXIMATE MEASURED-POA PHYSICS BASELINE
# ─────────────────────────────────────────────────────────────────────────────
# Measured panel-plane irradiance is used directly as effective irradiance.
# No decomposition, no retransposition, no solar geometry. See the module
# docstring for the three approximations this rests on.
print("\n=== STEP 5: Expected power (both capacity scenarios) ===")
for label, pdc0 in [("5k0", PDC0_RATED), ("5k28", PDC0_NAMEPLATE)]:
    p_exp = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=df["irr_panel_wm2"].clip(lower=0),
        temp_cell=df["t_module_rear_c"],
        pdc0=pdc0,
        gamma_pdc=GAMMA_PDC,
    ).clip(lower=0)
    df[f"p_exp_{label}"]    = p_exp
    df[f"residual_{label}"] = df["p_dc_computed"] - p_exp
    df[f"ratio_{label}"]    = df["p_dc_computed"] / p_exp.replace(0, np.nan)
    print(f"  pdc0={pdc0:>7.0f} W -> p_exp_{label}, residual_{label}, ratio_{label}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 6: Cleaning ===")
before = len(df)

essential = ["f_nv", "irr_panel_wm2", "t_module_rear_c", "p_dc_computed",
             "vdc1", "vdc2", "idc1", "idc2"]
df = df.dropna(subset=essential)
after_nan = len(df)

df = df[df["irr_panel_wm2"] >= MIN_IRR]
df = df[df["irr_panel_wm2"] <= MAX_IRR]
df = df[df["p_dc_computed"] >= 0]
df["f_nv"] = df["f_nv"].astype(int)

after = len(df)
print(f"  Rows before cleaning : {before:,}")
print(f"  After NaN drop       : {after_nan:,}  ({before - after_nan:,} removed)")
print(f"  After threshold drops: {after:,}  ({after_nan - after:,} removed)")
print(f"  Total removed        : {before - after:,}")

print("\n  Label distribution vs paper (Section 4.1):")
print(f"    {'Label':<20} {'ours':>10} {'paper':>10} {'diff':>10}  match")
counts = df["f_nv"].value_counts().sort_index()
for lbl in sorted(PAPER_COUNTS):
    ours = int(counts.get(lbl, 0))
    paper = PAPER_COUNTS[lbl]
    diff = ours - paper
    match = "EXACT" if diff == 0 else "MISMATCH"
    print(f"    {lbl} {LABEL_NAMES[lbl]:<17} {ours:>10,} {paper:>10,} "
          f"{diff:>+10,}  {match}")

controlled_exact = all(int(counts.get(l, 0)) == PAPER_COUNTS[l] for l in (1, 2, 3))
normal_diff = int(counts.get(0, 0)) - PAPER_COUNTS[0]
shadow_diff = int(counts.get(4, 0)) - PAPER_COUNTS[4]
print(f"\n  Controlled faults (1,2,3) all exact: {controlled_exact}")
print(f"  Normal difference   : {normal_diff:+,}")
print(f"  Shadowing difference: {shadow_diff:+,}")
print(f"  Combined difference : {normal_diff + shadow_diff:+,}")
print("  LIKELY explanation (NOT verified): the paper labelled a fixed")
print("  07:30-17:00 window while this pipeline filters on irradiance")
print("  (>= 20 W/m^2), so the two rules admit different edge-of-day rows.")
print("  Without authentic timestamps this cannot be confirmed, and the")
print("  difference above remains unexplained evidence.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SAVE
# ─────────────────────────────────────────────────────────────────────────────
FINAL_COLUMNS = [
    "sample_index", "source_event_id", "irr_panel_wm2", "t_module_rear_c", "f_nv",
    "vdc1", "vdc2", "idc1", "idc2", "p1", "p2", "p_dc_computed",
    "v_ratio", "i_ratio", "v_imbalance", "i_imbalance",
    "p_exp_5k0", "residual_5k0", "ratio_5k0",
    "p_exp_5k28", "residual_5k28", "ratio_5k28",
]
missing = [c for c in FINAL_COLUMNS if c not in df.columns]
if missing:
    raise SystemExit(f"Expected columns absent: {missing}")

df = df[FINAL_COLUMNS]
df.to_csv(OUT_CSV, index=False)

print(f"\n=== Saved: {OUT_CSV} ===")
print(f"  Shape  : {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Retained source events: {df['source_event_id'].nunique():,} "
      f"of {len(register):,}")

with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"fault_preprocessing.py | "
            f"input: {AMB_CSV.name} {len(amb):,} rows + {ELEC_CSV.name} {len(elec):,} rows | "
            f"output: {OUT_CSV.name} {len(df):,} rows, "
            f"{EVENT_REGISTER.name} {len(register):,} rows | "
            f"notes: controlled_faults_exact={controlled_exact}, "
            f"normal_diff={normal_diff:+}, shadow_diff={shadow_diff:+}, "
            f"no calendar or solar-geometry features\n")
print(f"  Run log appended: {RUN_LOG}")
print("\nPreprocessing complete. Next: event_support_analysis.py (read-only).")
