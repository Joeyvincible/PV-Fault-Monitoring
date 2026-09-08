"""
objective3_feature_audit.py
=====================
Objective 3 — design audit: cross-domain feature contract.

READ-ONLY. Writes no transfer code and TRAINS NOTHING, including the
Brazil-side source-performance gate, which is marked PENDING IMPLEMENTATION
rather than estimated. Objectives 1 and 3 are frozen and untouched.

PERMITTED INPUTS
  02_Fault_Classification/data/fault_dataset.csv
  01_HK_Detection/outputs/objective2_modeling/objective2_grid.csv
  02_Fault_Classification/outputs/extended_ablation/
      extended_ablation_summary.csv
FORBIDDEN
  fault_dataset_cleaned.csv, any historical HK output, and pv_cross_dataset.py
  as a source of design (its feature list references zenith, azimuth, dni_est
  and a GHI-derived poa_global, all removed by the Brazil preprocessing pipeline).

WRITES ONLY into this directory (Objective3_Outputs/objective3_design/).

BRAZIL TIMESTAMP LIMITATION
  The processed Brazil release has no valid timestamps. No datetime is
  fabricated anywhere here, and sample_index is treated as a sample counter,
  never as clock time. Recording groups are reconstructed with exactly the
  frozen Objective 1 rule: a gap in sample_index > 3600 starts a new group,
  yielding 16 groups. source_event_id is NOT used for grouping — it marks
  label transitions, not recording boundaries.

PART 2a AGGREGATION IS DIAGNOSTIC ONLY
  Non-overlapping consecutive 900-sample blocks within a recording group,
  never crossing a group boundary, incomplete trailing blocks dropped.
  Described throughout as a 900-second / nominal 15-minute SUPPORT
  AGGREGATION, never a clock-aligned 15-minute resample. Blocks spanning more
  than one f_nv value are left UNLABELLED — no class is invented. These rows
  train nothing.
"""

import datetime
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

OUT_DIR     = Path(__file__).resolve().parent          # Objective3_Outputs/objective3_design
NAVE        = OUT_DIR.parent.parent                    # repository root
BRAZIL_CSV  = NAVE / "02_Fault_Classification" / "data" / "fault_dataset.csv"
HK_CSV      = NAVE / "01_HK_Detection" / "outputs" / "objective2_modeling" / "objective2_grid.csv"
ABLATION    = (NAVE / "02_Fault_Classification" / "outputs"
               / "extended_ablation" / "extended_ablation_summary.csv")
RUN_LOG     = OUT_DIR / "run_log.txt"   # Keep audit logs within the Objective 3 output tree.

TZ = "Asia/Hong_Kong"
GAP_THRESHOLD = 3600      # Recording-group reconstruction threshold.
BLOCK = 900               # 900-second support aggregation (diagnostic only)
PDC0_BR_RATED, PDC0_BR_NAMEPLATE, PDC0_HK = 5000.0, 5280.0, 27606.0
CLEAN_MONTHS = [1, 2, 11, 12]
SENSOR_MAX = 1200.0

# ── Terminology guard (Part 4) ───────────────────────────────────────────────
BANNED = ["hk accuracy", "hk f1", "hk recall", "hk precision", "validated on hk",
          "hk classification accuracy", "confirmed fault",
          "hk fault detection accuracy"]
_checked = []


def check(text, where):
    low = text.lower()
    hits = [b for b in BANNED if b in low]
    assert not hits, f"BANNED PHRASE in {where}: {hits} -> {text[:120]!r}"
    _checked.append((where, text))
    return text


# ── Input guard ──────────────────────────────────────────────────────────────
FORBIDDEN_INPUTS = ["fault_dataset_cleaned.csv", "outputs/no_sensor",
                    "outputs/lstm_with_sensor", "pv_cross_dataset.py",
                    "ablation_summary_v2.csv", "outputs/calibration"]
_read = []


def rd(path, **kw):
    p = Path(path)
    s = str(p).replace("\\", "/")
    bad = [f for f in FORBIDDEN_INPUTS if f in s]
    assert not bad, f"FORBIDDEN INPUT: {bad} -> {p}"
    _read.append(str(p.relative_to(NAVE)))
    return pd.read_csv(p, **kw)


# ── Self-check: no model fitting anywhere in this script ─────────────────────
def assert_no_model_fitting():
    src = Path(__file__).read_text().splitlines()
    body = [l for l in src if "NOFIT-EXEMPT" not in l]
    pat = re.compile(
        r"\.fit\(|\.fit_transform\(|\.partial_fit\(|sk" + "learn|"
        r"to" + "rch|Classifier\(|Regressor\(|train_test_split", re.I)  # NOFIT-EXEMPT
    hits = [(i + 1, l.strip()[:90]) for i, l in enumerate(body) if pat.search(l)]
    return hits


_rep = []


def emit(s=""):
    check(s, "report line")
    print(s)
    _rep.append(s)


print("=" * 78)
print("  OBJECTIVE 3 — DESIGN AUDIT (read-only, nothing trained)")
print("=" * 78)
fit_hits = assert_no_model_fitting()
print(f"\nNo-model-fitting self-check: "
      f"{'PASS — no fitting call found' if not fit_hits else 'FAIL'}")
for ln, t in fit_hits:
    print(f"    line {ln}: {t}")
assert not fit_hits, "Model-fitting call detected"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
br = rd(BRAZIL_CSV)
hk = rd(HK_CSV, index_col=0)
hk.index = pd.to_datetime(hk.index, utc=True).tz_convert(TZ)
abl = rd(ABLATION)

# Joint two-feature support (Part 2b) is computed by objective3_gate_check.py, which
# must be run first. It is read here rather than recomputed so the spec and the
# CSV can never disagree.
JOINT_CSV = OUT_DIR / "objective3_joint_support.csv"
assert JOINT_CSV.exists(), ("objective3_joint_support.csv missing — run "
                            "objective3_gate_check.py before regenerating the spec")
jt = rd(JOINT_CSV)
J = {(r.capacity, r.brazil_representation):
     (float(r.hull_containment_pct), float(r.bin_containment_pct))
     for r in jt.itertuples()}
HK_JOINT_N = int(jt["hk_n"].iloc[0])

# Macro F1 for the reduced aggregate representations, read directly from
# extended_ablation_summary.csv rather than quoted from memory. These are
# CONTEXT for the size of the string-signal loss, NOT the source-performance
# gate - none of them is the shared contract (Part 5c).
def best_f1(name):
    r = abl[abl["feature_set"] == name].sort_values("macro_f1_oof", ascending=False)
    assert len(r), f"feature set absent from ablation summary: {name}"
    return float(r.iloc[0]["macro_f1_oof"]), str(r.iloc[0]["model"])

f1_pwr_irr, m_pwr_irr = best_f1("Computed power + measured irradiance")
f1_pwr,     m_pwr     = best_f1("Computed total power only")
f1_full,    m_full    = best_f1("Electrical (full)")
print(f"\nAblation figures verified from source (best model per set):")
print(f"  Electrical (full)                    {f1_full:.4f}  ({m_full})")
print(f"  Computed power + measured irradiance {f1_pwr_irr:.4f}  ({m_pwr_irr})")
print(f"  Computed total power only            {f1_pwr:.4f}  ({m_pwr})")
print("\nINPUT FILES READ:")
for f in _read:
    print(f"  {f}")

# Reconstruct Brazil recording groups from sample-index gaps.
br = br.sort_values("sample_index").reset_index(drop=True)
br["recording_group"] = (br["sample_index"].diff() > GAP_THRESHOLD
                         ).fillna(False).cumsum().astype(int)
n_groups = br["recording_group"].nunique()
assert n_groups == 16, f"Expected 16 recording groups, got {n_groups}"
print(f"\nBrazil: {len(br):,} rows, {n_groups} recording groups "
      f"(gap > {GAP_THRESHOLD} rule, reconstructed — not stored in the release)")

# ── Part 2a: 900-sample support aggregation, DIAGNOSTIC ONLY ─────────────────
# SPLIT FIRST, THEN AGGREGATE. fault_dataset.csv is filtered, so
# sample_index contains gaps and 900 consecutive dataframe ROWS may span far
# more than 900 seconds. Chunking rows 1-900, 901-1800 and rejecting chunks
# that contain a gap would also discard contiguous stretches either side of
# that gap unnecessarily. Instead each recording_group is split into maximal
# contiguous sample_index runs, and blocks are formed inside each run.
blocks = []
runs_total = 0
naive_blocks = 0          # what naive row-based chunking would have produced
for g, sub in br.groupby("recording_group"):
    sub = sub.sort_values("sample_index").reset_index(drop=True)
    naive_blocks += len(sub) // BLOCK
    run_id = (sub["sample_index"].diff() != 1).cumsum()
    for _, run in sub.groupby(run_id):
        runs_total += 1
        run = run.reset_index(drop=True)
        for b in range(len(run) // BLOCK):          # trailing remainder dropped
            seg = run.iloc[b * BLOCK:(b + 1) * BLOCK]
            si = seg["sample_index"].to_numpy()
            # four integrity assertions required of every retained block
            assert len(seg) == BLOCK, "block is not exactly 900 rows"
            assert seg["recording_group"].nunique() == 1, "block spans groups"
            assert bool(np.all(np.diff(si) == 1)), "block is not contiguous"
            assert int(si[-1] - si[0]) == BLOCK - 1, "block span != 899 samples"
            labs = seg["f_nv"].unique()
            blocks.append({
                "recording_group": int(g), "block": b,
                "start_sample_index": int(si[0]), "end_sample_index": int(si[-1]),
                "p_dc_computed": float(seg["p_dc_computed"].mean()),
                "irr_panel_wm2": float(seg["irr_panel_wm2"].mean()),
                # mixed blocks are left UNLABELLED - no class is invented
                "f_nv": int(labs[0]) if len(labs) == 1 else np.nan,
                "mixed": len(labs) > 1})
brb = pd.DataFrame(blocks)
N_RUNS = runs_total
N_DISCONT = runs_total - n_groups          # internal gaps inside groups
N_NAIVE = naive_blocks
N_VALID = len(brb)
N_MIXED = int(brb["mixed"].sum())
N_SINGLE = N_VALID - N_MIXED
print(f"\nPart 2a — 900-second support aggregation (split-first, diagnostic only):")
print(f"  contiguous sample_index runs found : {N_RUNS:,} "
      f"({N_DISCONT:,} discontinuities inside {n_groups} recording groups)")
print(f"  naive row-based chunking would give: {N_NAIVE:,} blocks")
print(f"  VALID continuous 900-second blocks : {N_VALID:,} "
      f"({N_NAIVE - N_VALID:,} rejected as not temporally contiguous)")
print(f"  mixed-label among valid            : {N_MIXED:,} (left unlabelled)")
print(f"  single-label, potentially usable   : {N_SINGLE:,}")
print(f"  integrity assertions (900 rows / one group / one run / diff==1): PASS")

# ── HK populations ──────────────────────────────────────────────────────────
elig = hk["eligible"].astype(bool)
hk_e = hk[elig]
POP = {
    "HK eligible, full 2023":
        hk_e[hk_e.index.year == 2023],
    "HK eligible, clean-sensor months 2023 (Jan/Feb/Nov/Dec)":
        hk_e[(hk_e.index.year == 2023) & hk_e.index.month.isin(CLEAN_MONTHS)],
    "HK eligible, all years 2021-2023 (supplementary)":
        hk_e,
}
print("\nHK comparison populations (Objective 2 eligibility: "
      "ghi_clear >= 50 and zenith < 85):")
for k, v in POP.items():
    print(f"  {len(v):>7,}  {k}")


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def overlap_coef(a, b, bins=120):
    lo = float(min(np.nanmin(a), np.nanmin(b)))
    hi = float(max(np.nanmax(a), np.nanmax(b)))
    if hi <= lo:
        return np.nan
    e = np.linspace(lo, hi, bins + 1)
    da, _ = np.histogram(a, bins=e, density=True)
    db, _ = np.histogram(b, bins=e, density=True)
    return float(np.sum(np.minimum(da, db)) * (e[1] - e[0]))


def describe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(n=int(x.size), mean=float(x.mean()), median=float(q[2]),
                sd=float(x.std(ddof=1)), p5=float(q[0]), p25=float(q[1]),
                p75=float(q[3]), p95=float(q[4]),
                min=float(x.min()), max=float(x.max()))


def compare(qty, br_rep, br_vals, hk_pop, hk_vals, gate):
    a = np.asarray(br_vals, float); a = a[np.isfinite(a)]
    b = np.asarray(hk_vals, float); b = b[np.isfinite(b)]
    ks = ks_2samp(a, b)
    da, db = describe(a), describe(b)
    inside = float(np.mean((b >= a.min()) & (b <= a.max())))
    inside_core = float(np.mean((b >= np.percentile(a, 0.5))
                                & (b <= np.percentile(a, 99.5))))
    row = {"quantity": qty, "brazil_representation": br_rep,
           "hk_population": hk_pop, "used_for_gate": gate,
           "ks_statistic": round(float(ks.statistic), 4),
           "overlap_coefficient": round(overlap_coef(a, b), 4),
           "hk_inside_brazil_minmax_pct": round(100 * inside, 2),
           "hk_inside_brazil_p0.5_p99.5_pct": round(100 * inside_core, 2),
           "ranges_overlap": bool(a.min() <= b.max() and b.min() <= a.max())}
    row.update({f"br_{k}": round(v, 4) for k, v in da.items()})
    row.update({f"hk_{k}": round(v, 4) for k, v in db.items()})
    return row


rows = []
# --- Quantity 1: capacity-normalised power ---------------------------------
for cap_lab, cap in [("rated 5.00 kW", PDC0_BR_RATED),
                     ("nameplate 5.28 kW", PDC0_BR_NAMEPLATE)]:
    for pop_lab in ["HK eligible, full 2023",
                    "HK eligible, all years 2021-2023 (supplementary)"]:
        hv = POP[pop_lab]["power_W"] / PDC0_HK
        gate = pop_lab.startswith("HK eligible, full 2023")
        rows.append(compare(f"normalised power (Brazil / {cap_lab})",
                            "native 1 Hz", br["p_dc_computed"] / cap,
                            pop_lab, hv, gate))
        rows.append(compare(f"normalised power (Brazil / {cap_lab})",
                            "900-s support aggregation (diagnostic)",
                            brb["p_dc_computed"] / cap, pop_lab, hv, False))
        # Normal-only Brazil, as a domain (non-fault) comparison
        rows.append(compare(f"normalised power (Brazil / {cap_lab}, Normal only)",
                            "native 1 Hz",
                            br.loc[br["f_nv"] == 0, "p_dc_computed"] / cap,
                            pop_lab, hv, False))

# --- Quantity 2: irradiance -------------------------------------------------
gate_pop = "HK eligible, clean-sensor months 2023 (Jan/Feb/Nov/Dec)"
for pop_lab, gate, note in [
        (gate_pop, True, "GATE population"),
        ("HK eligible, full 2023", False,
         "known sensor-corruption diagnostic — EXCLUDED from the gate")]:
    hv = POP[pop_lab]["irr_meas"]
    rows.append({**compare("irradiance (W/m2)", "native 1 Hz",
                           br["irr_panel_wm2"], pop_lab, hv, gate),
                 "note": note})
    rows.append({**compare("irradiance (W/m2)",
                           "900-s support aggregation (diagnostic)",
                           brb["irr_panel_wm2"], pop_lab, hv, False),
                 "note": note})

dist = pd.DataFrame(rows)   # written in the output block below


def ks_of(qty_sub, rep, pop):
    m = dist[(dist.quantity == qty_sub) & (dist.brazil_representation == rep)
             & (dist.hk_population == pop)]
    return float(m.iloc[0].ks_statistic) if len(m) else np.nan


PWR = "normalised power (Brazil / rated 5.00 kW)"
KS_PWR_NAT = ks_of(PWR, "native 1 Hz", "HK eligible, full 2023")
KS_PWR_AGG = ks_of(PWR, "900-s support aggregation (diagnostic)",
                   "HK eligible, full 2023")
KS_IRR_NAT = ks_of("irradiance (W/m2)", "native 1 Hz", gate_pop)
KS_IRR_AGG = ks_of("irradiance (W/m2)", "900-s support aggregation (diagnostic)",
                   gate_pop)
KS_IRR_FULL = ks_of("irradiance (W/m2)", "native 1 Hz", "HK eligible, full 2023")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — SEMANTIC COMPARABILITY
# ─────────────────────────────────────────────────────────────────────────────
# Two distinct senses of "comparable" are kept apart throughout. PROXY means
# the quantities are alignable well enough for exploratory binary screening,
# NOT that they are the same physical quantity.
C_OK, C_NORM, C_NO, C_UNRES = ("PHYSICALLY EQUIVALENT",
                               "PROXY-COMPARABLE ONLY (not physically equivalent)",
                               "NOT COMPARABLE", "UNRESOLVED")
SEM = [
 dict(quantity="Power",
      brazil="p_dc_computed = vdc1*idc1 + vdc2*idc2; DC array power; 1 Hz",
      hong_kong="power(W); AC active power (CONFIRMED from inverter schema); "
                "15-min interval mean labelled at interval start",
      classification=C_NORM, evidence="CONFIRMED",
      reasoning="Capacity normalisation puts both on a 0-1 scale, but does NOT "
                "make them the same physical quantity: Brazil is pre-inverter DC, "
                "HK is post-inverter AC. Objective 2 rejected introducing an "
                "assumed inverter efficiency, and that reasoning applies "
                "unchanged here. Carried into Part 2 as the only viable shared "
                "electrical signal, with the DC/AC residual left explicitly "
                "unresolved. Closing it would require either DC-side measurement "
                "at SQ1 (not recorded anywhere in the HK release) or a "
                "load-dependent inverter efficiency curve for the SolarEdge unit "
                "(not published in the metadata)."),
 dict(quantity="Irradiance",
      brazil="irr_panel_wm2; panel-plane global irradiance, pyranometer at "
             "module inclination (paper Section 3.2); tilt 30 deg",
      hong_kong="Irradiance (W/m2); plane UNRESOLVED, but SQ1 is tilt 0 deg so "
                "GHI and plane-of-array coincide; station pairing to SQ1 "
                "undocumented",
      classification=C_NORM, evidence="SUPPORTED ASSUMPTION",
      reasoning="Same unit and, for each array, the same physical quantity "
                "(irradiance in the plane of that array). Two caveats that do "
                "not block comparison but must travel with it: the HK sensor "
                "plane is undocumented (inert at tilt 0), and one campus weather "
                "station serves 60 PV stations with no documented pairing to "
                "SQ1. The arrays differ in tilt (30 deg vs 0 deg), so identical "
                "sky conditions produce different plane irradiance."),
 dict(quantity="Module / ambient temperature",
      brazil="t_module_rear_c; rear-module temperature, mean of 4 PT100 sensors",
      hong_kong="Temp (Degree Celsius); ambient air temperature at the weather "
                "station",
      classification=C_NO, evidence="CONFIRMED",
      reasoning="Different physical quantities. Rear-module temperature runs "
                "roughly 15-25 C above ambient under load, and the offset is "
                "itself a function of irradiance and wind, so it is not a fixed "
                "bias that could be subtracted. Mapping one to the other would "
                "require a thermal model with wind speed and mounting "
                "parameters, which is exactly the kind of unjustified extra "
                "parameter this project has repeatedly declined. EXCLUDED from "
                "the shared contract."),
 dict(quantity="Expected power",
      brazil="p_exp_5k0 / p_exp_5k28; PVWatts DC from measured panel-plane "
             "irradiance",
      hong_kong="none - Objective 2 deliberately dropped DC expected power "
                "because PVWatts models DC while power(W) is AC",
      classification=C_NO, evidence="CONFIRMED",
      reasoning="Absent in the target domain by design, not by oversight. "
                "Reconstructing it for HK would reintroduce the DC/AC mismatch "
                "Objective 2 removed."),
 dict(quantity="Residual / ratio",
      brazil="residual_5k0, ratio_5k0 (and 5k28 variants)",
      hong_kong="none in the final pipeline",
      classification=C_NO, evidence="CONFIRMED",
      reasoning="Derived from expected power, so unavailable for the same "
                "reason. Note these were among the most informative Brazil "
                "features."),
 dict(quantity="String voltage / current and derived imbalance",
      brazil="vdc1, vdc2, idc1, idc2, v_ratio, i_ratio, v_imbalance, i_imbalance",
      hong_kong="ABSENT - HK records no string-level measurement at any station",
      classification=C_NO, evidence="CONFIRMED",
      reasoning="This is the decisive finding of the audit. Objective 1 "
                "established that string voltage removal collapsed short-circuit "
                "F1 to 0.002 and that voltage removal cost roughly twice what "
                "current removal cost. The features carrying Brazil's fault "
                "discrimination are precisely the ones the target domain does "
                "not have."),
 dict(quantity="Sampling support",
      brazil="1 Hz native",
      hong_kong="15-minute interval means",
      classification=C_NO, evidence="CONFIRMED",
      reasoning="Not a quantity to be mapped but a property that distorts every "
                "distribution comparison: averaging ~900 one-second samples "
                "reduces variance substantially. Handled by the diagnostic "
                "900-second support aggregation in Part 2a, which is never used "
                "for training."),
 dict(quantity="Capacity",
      brazil="5.0 kW rated (paper) / 5.28 kW nameplate (16 x 330 W)",
      hong_kong="27.6 kW (SQ1 TTL ext:ratedPowerOutput)",
      classification=C_NORM, evidence="CONFIRMED",
      reasoning="Provides the normalisation basis for power. The Brazil figure "
                "is itself ambiguous, so both variants are carried; Objective 1 "
                "found results insensitive to the choice (Macro F1 difference "
                "0.0069)."),
 dict(quantity="Derived clearness-like ratio (power_norm / irradiance)",
      brazil="computable from the two quantities above",
      hong_kong="computable from the two quantities above",
      classification=C_UNRES, evidence="SUPPORTED ASSUMPTION",
      reasoning="Constructible in both domains and would carry more information "
                "than either parent alone, but inherits every caveat of both - "
                "including the unresolved DC/AC gap in the numerator. Offered as "
                "a candidate for the shared contract, not adopted here."),
]
sem = pd.DataFrame(SEM)
SHARED = [d["quantity"] for d in SEM
          if d["classification"] in (C_OK, C_NORM) and d["quantity"] != "Capacity"]

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
NAT = "Brazil native 1 Hz"
AGG = "Brazil 900-s support aggregation (diagnostic)"
fig, ax = plt.subplots(2, 2, figsize=(15, 11))


def dens(a, axx, label, color, ls="-"):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    lo, hi = np.percentile(a, [0.2, 99.8])
    e = np.linspace(lo, hi, 90)
    d, _ = np.histogram(a, bins=e, density=True)
    axx.plot((e[:-1] + e[1:]) / 2, d, ls, color=color, lw=1.8, label=label)


pw_hk = POP["HK eligible, full 2023"]["power_W"] / PDC0_HK
dens(br["p_dc_computed"] / PDC0_BR_RATED, ax[0, 0], NAT, "tab:orange")
dens(brb["p_dc_computed"] / PDC0_BR_RATED, ax[0, 0], AGG, "tab:red", "--")
dens(pw_hk, ax[0, 0], "HK eligible, full 2023 (AC / 27.6 kW)", "tab:blue")
ax[0, 0].set_title("Capacity-normalised power\n(Brazil DC / 5.00 kW vs HK AC / 27.6 kW)",
                   fontsize=11)
ax[0, 0].set_xlabel("normalised power (-)"); ax[0, 0].set_ylabel("density")

irr_gate = POP[gate_pop]["irr_meas"]
dens(br["irr_panel_wm2"], ax[0, 1], NAT, "tab:orange")
dens(brb["irr_panel_wm2"], ax[0, 1], AGG, "tab:red", "--")
dens(irr_gate, ax[0, 1], "HK clean-sensor months 2023 (GATE)", "tab:blue")
ax[0, 1].set_title("Irradiance — GATE population\n(HK clean-sensor months only)",
                   fontsize=11)
ax[0, 1].set_xlabel("irradiance (W/m²)"); ax[0, 1].set_ylabel("density")

dens(br["irr_panel_wm2"], ax[1, 0], NAT, "tab:orange")
dens(POP["HK eligible, full 2023"]["irr_meas"], ax[1, 0],
     "HK full 2023 (sensor-corruption diagnostic)", "grey")
ax[1, 0].axvline(SENSOR_MAX, color="red", ls=":", lw=1.5,
                 label=f"{SENSOR_MAX:.0f} W/m² reference line")
ax[1, 0].set_title("Irradiance — full 2023, KNOWN SENSOR CORRUPTION\n"
                   "EXCLUDED from the transfer-feasibility gate", fontsize=11,
                   color="darkred")
ax[1, 0].set_xlabel("irradiance (W/m²)"); ax[1, 0].set_ylabel("density")

for a_, lab, c in [(br["p_dc_computed"] / PDC0_BR_RATED, NAT, "tab:orange"),
                   (brb["p_dc_computed"] / PDC0_BR_RATED, AGG, "tab:red"),
                   (pw_hk, "HK eligible, full 2023", "tab:blue")]:
    v = np.sort(np.asarray(a_, float)); v = v[np.isfinite(v)]
    ax[1, 1].plot(v, np.linspace(0, 1, v.size), color=c, lw=1.8, label=lab)
ax[1, 1].set_title("Normalised power — empirical CDF\n(support overlap)", fontsize=11)
ax[1, 1].set_xlabel("normalised power (-)"); ax[1, 1].set_ylabel("cumulative")

for a_ in ax.ravel():
    a_.legend(fontsize=8); a_.grid(alpha=0.3)
fig.suptitle("Objective 3 — Brazil / Hong Kong distribution overlap for the "
             "shared quantities", fontsize=13)
CAP = check(
    f"Brazil is shown in two representations: native 1 Hz and a 900-second "
    f"support aggregation (diagnostic only, never used for training, mixed-label "
    f"blocks unlabelled). Matching temporal support changes the KS statistic for "
    f"normalised power from {KS_PWR_NAT:.3f} to {KS_PWR_AGG:.3f} (HK eligible "
    f"full 2023) and for irradiance from {KS_IRR_NAT:.3f} to {KS_IRR_AGG:.3f} "
    f"(HK clean-sensor months); this measures how "
    f"sensitive the discrepancy is to temporal support, NOT a causal "
    f"apportionment of domain mismatch. The full-2023 HK irradiance panel is a "
    f"known sensor-corruption diagnostic and is excluded from the gate.",
    "figure caption")
import textwrap
fig.tight_layout(rect=[0.01, 0.10, 0.99, 0.95])
fig.text(0.5, 0.045, textwrap.fill(CAP, width=132), ha="center", va="center",
         fontsize=9, bbox=dict(fc="#f7f7f7", ec="grey", boxstyle="round,pad=0.6"))

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SPECIFICATION (Parts 1, 3, 4, 5)
# ─────────────────────────────────────────────────────────────────────────────
n_shared = len(SHARED)
SPEC = check(f"""# Objective 3 — Design Specification: Cross-Domain Feature Contract

**Design audit only. No transfer code written and no model trained.** The
Brazil-side source-performance gate is **answered from frozen Objective 1
outputs without retraining**, via the information-equivalence argument in Part
5c — established from the saved pipeline, not from feature names. Objectives 1
and 3 are frozen and untouched.

Generated: {datetime.datetime.now().isoformat(timespec='seconds')}

## Framing

Objective 3 is **cross-domain compatibility and potential-fault screening**:
whether a classifier learned from the labelled Brazil dataset produces
meaningful screening behaviour when applied to Hong Kong, while explicitly
recognising that HK predictions cannot be checked against truth. HK has no
per-timestep fault labels, so no statement in this document asserts that any HK
prediction is correct.

---

## Part 1 — Semantic comparability

Full table with reasoning in `objective3_semantic_comparability.csv`. Summary:

| Quantity | Classification |
|---|---|
| Power (capacity-normalised) | PROXY-COMPARABLE ONLY — DC vs AC, not physically equivalent |
| Irradiance | PROXY-COMPARABLE ONLY — plane and station pairing unresolved |
| Module / ambient temperature | NOT COMPARABLE |
| Expected power | NOT COMPARABLE (absent in HK by design) |
| Residual / ratio | NOT COMPARABLE (absent in HK) |
| String voltage / current + imbalance | NOT COMPARABLE (absent in HK) |
| Sampling support | NOT COMPARABLE (handled diagnostically in Part 2a) |
| Capacity | PROXY-COMPARABLE ONLY (normalisation basis) |
| Derived clearness-like ratio | UNRESOLVED (candidate, not adopted) |

### Physical equivalence versus proxy comparability

**Neither of the two shared quantities is physically equivalent across the two
domains, and this spec does not claim otherwise.** Brazil power is pre-inverter
DC and HK power is post-inverter AC — different physical quantities measured at
different points in the energy chain, and capacity normalisation rescales
magnitude without reconciling the conversion. Brazil irradiance is panel-plane
at 30 deg tilt from a co-located pyranometer; HK irradiance is of unresolved
plane from a weather station whose pairing to SQ1 is undocumented.

The correct classification for both is therefore **proxy-comparable for
exploratory binary screening, not physically equivalent**. A proxy alignment can
support a Normal-versus-anomalous screening signal, where what matters is
whether output is depressed relative to the same site's own conditioned
expectation. It cannot support quantitative cross-domain physical claims, and it
cannot support fault-type assignment.

### The four problems, addressed

**1. AC/DC.** Brazil power is pre-inverter DC; HK is post-inverter AC.
Capacity normalisation puts both on a 0-1 scale but does not reconcile the
physics: it rescales magnitude, not the conversion. Closing the gap would need
either DC-side measurement at SQ1 — recorded nowhere in the HK release, which
carries only `dcVoltage(V)` and no DC current — or a load-dependent inverter
efficiency curve for the SolarEdge unit, which the metadata does not publish.
An assumed fixed efficiency is rejected here for the same reason Objective 2
rejected it. **The residual is left explicitly unresolved and travels with every
result.**

**2. Temperature.** Rear-module temperature runs roughly 15-25 C above ambient
under load, and the offset varies with irradiance and wind rather than being a
subtractable constant. No defensible mapping exists without a thermal model
carrying new unjustified parameters. **Temperature is excluded from the shared
contract.**

**3. String signals — the decisive finding.** Objective 1 established that
removing string voltage collapsed short-circuit F1 to **0.002**, and that
voltage removal cost roughly twice what current removal cost (0.310-0.377
against 0.124-0.181 across four model families). Those findings rest entirely on
`vdc1`, `vdc2`, `idc1`, `idc2` and their derived imbalance features. **Hong Kong
records none of them at any station.**

Stated precisely: the features carrying Brazil's **strongest** fault
discrimination are unavailable in the target domain. This is not the claim that
all Brazil discrimination came from string-level signals — it did not. The
reduced aggregate representations retained non-trivial classification ability:
`Computed power + measured irradiance` reached Macro F1 **{f1_pwr_irr:.4f}**
(MLP) and `Computed total power only` **{f1_pwr:.4f}** (HistGBM), against
**{f1_full:.4f}** for the full electrical set — all read directly from
`extended_ablation_summary.csv`. A transferred classifier restricted to the
shared contract therefore does not start from zero, but it does lose the signal
that lifted the source task from moderate to near-ceiling separability, and it
loses it disproportionately on the classes named in Part 3.

Those figures are context for the size of the loss. `Computed power + measured
irradiance` is additionally shown in Part 5c to be **information-equivalent to
the shared contract for the actual saved MLP pipeline**, which is what allows it
to serve as the source-performance gate without retraining. That equivalence is
established from the pipeline, not from the feature names, and it does **not**
extend to `POWER_PHYSICS` or any other nearby set.

**4. Sampling rate.** Brazil is 1 Hz; HK is 15-minute means. A 10-minute
induction is ~600 Brazil samples but sits **inside a single HK interval**, where
it would appear only as a partial depression of one averaged value. A
transferred classifier could therefore not resolve a Brazil-style short
induction in HK at all. Only conditions persisting across multiple 15-minute
intervals — degradation, extended shading, sustained damage — are resolvable.

---

### 2b. Joint two-feature support

Marginal statistics are not sufficient, because the classifier consumes the two
features **jointly**. The method below was fixed before any result was seen: two
descriptive containment measures, no learned domain classifier, no KDE
classifier, no fitted predictive model of any kind. HK is the clean-sensor /
no-known-fault 2023 population (n = {HK_JOINT_N:,} rows with both quantities
present).

| Brazil representation | Capacity | HK inside Brazil convex hull | HK inside Brazil-occupied cells (20x20) |
|---|---|---|---|
| All classes, native 1 Hz | 5.00 kW | {J[("5.00 kW","Brazil all classes, native 1 Hz")][0]:.2f}% | {J[("5.00 kW","Brazil all classes, native 1 Hz")][1]:.2f}% |
| All classes, native 1 Hz | 5.28 kW | {J[("5.28 kW","Brazil all classes, native 1 Hz")][0]:.2f}% | {J[("5.28 kW","Brazil all classes, native 1 Hz")][1]:.2f}% |
| Normal only, native 1 Hz | 5.00 kW | {J[("5.00 kW","Brazil Normal only, native 1 Hz")][0]:.2f}% | {J[("5.00 kW","Brazil Normal only, native 1 Hz")][1]:.2f}% |
| Normal only, native 1 Hz | 5.28 kW | {J[("5.28 kW","Brazil Normal only, native 1 Hz")][0]:.2f}% | {J[("5.28 kW","Brazil Normal only, native 1 Hz")][1]:.2f}% |
| All classes, 900-s diagnostic | 5.00 kW | {J[("5.00 kW","Brazil all classes, 900-s diagnostic")][0]:.2f}% | {J[("5.00 kW","Brazil all classes, 900-s diagnostic")][1]:.2f}% |
| All classes, 900-s diagnostic | 5.28 kW | {J[("5.28 kW","Brazil all classes, 900-s diagnostic")][0]:.2f}% | {J[("5.28 kW","Brazil all classes, 900-s diagnostic")][1]:.2f}% |

Full table in `objective3_joint_support.csv`; figure in `objective3_joint_support.png`.

**Which HK population each statistic uses — the two power KS values are not in
conflict.** They are computed on different populations, both legitimately:

| Statistic | HK population | n | Value |
|---|---|---|---|
| Marginal KS, normalised power | HK eligible, **full 2023** | 15,870 | **0.320** |
| Marginal KS, irradiance | HK **clean-sensor months** (gate) | 4,749 | **0.340** |
| Joint-table KS, normalised power | HK **clean-sensor months** (gate) | 4,745 | **0.383** |
| Joint-table KS, irradiance | HK **clean-sensor months** (gate) | 4,745 | **0.340** |

Power does not depend on the corrupted HK irradiance sensor, so its marginal
statistic may legitimately use the larger full-2023 population; irradiance may
not. The joint diagnostic requires **both** quantities present, so it is
confined to the clean-sensor population throughout, and the small 4,749 -> 4,745
difference is the four clean-sensor rows lacking `power_W`. **Restricted to the
same clean-sensor population, the power KS is 0.383, not 0.320.** Every table in
this document names its population; no statistic is quoted without one.

**Both are reported deliberately, and neither alone defines a pass or fail.**
Convex-hull containment is permissive — a hull bridges empty interior regions,
so a point can lie "inside" a region the source never occupied. Occupied-bin
containment is more sensitive to actual joint source occupancy. Gate 2 is not
reduced to either statistic.

**What the two measures show together.** For Brazil all-classes native, the two
agree closely ({J[("5.00 kW","Brazil all classes, native 1 Hz")][0]:.1f}% hull
against {J[("5.00 kW","Brazil all classes, native 1 Hz")][1]:.1f}% bins), so HK
sits inside genuinely occupied source regions rather than inside hull-bridged
voids. The Normal-only comparison is where they separate sharply
({J[("5.00 kW","Brazil Normal only, native 1 Hz")][0]:.1f}% hull against
{J[("5.00 kW","Brazil Normal only, native 1 Hz")][1]:.1f}% bins): the hull looks
almost as permissive as before, but roughly a third of HK points fall in cells
Brazil's Normal class never occupied. This is exactly the gap the hull measure
conceals, and it is the more relevant number for a screening model whose
reference is Brazil Normal.

**The cross-domain normalised-power-per-irradiance offset.** In the joint plane,
Brazil's Normal class occupies a tight upper ridge while HK sits systematically
below it — less capacity-normalised power at the same irradiance. Restricting to
irradiance above 200 W/m², the median of capacity-normalised power per kW/m² is
**0.967 for Brazil Normal (IQR 0.925-0.990) against 0.799 for HK clean-sensor
(IQR 0.668-0.947)**, a ratio of **0.83** — HK approximately **17% lower** in
this comparison. The convex hull spans the empty region between the two ridges,
which is why hull containment stays high while occupied-bin containment falls.

**The cause of this offset cannot be isolated from the available evidence.** It
may reflect any combination of the AC-versus-DC measurement point, array
mounting and the 30 deg versus 0 deg tilt difference, the unresolved HK
irradiance plane and the undocumented weather-station-to-SQ1 pairing, module and
ambient temperature regimes, soiling, system losses, and other domain
differences not enumerated here. **No inverter conversion efficiency is assumed,
quantified, or attributed** — Objective 2 deliberately declined to introduce one,
and nothing in this audit establishes a value. The offset is **not evidence of a
fault at SQ1.**

**This is the most consequential Gate 2 finding for a screening transfer, and it
is a result to be observed rather than adjusted.** A classifier whose Normal
reference is Brazil's ridge will score a substantial share of ordinary HK
operation as unlike Brazil Normal. That behaviour **is part of the
transfer-compatibility finding**: it is what direct transfer across these two
domains actually does, and suppressing it before measuring it would destroy the
result. See Part 5d for the binding no-target-adaptation rule.

**HK Normal-status caveat.** The clean-sensor HK subset contains no known fault
event, but it carries **no verified per-timestamp Normal labels**. It is
described throughout as "clean-sensor / no-known-fault". All findings here are
stated **relative to Brazil Normal**, and none is a validated comparison between
two confirmed-normal populations.

The 900-s diagnostic rows are reported for completeness. They show
**substantially lower containment**, but that representation simultaneously
changes the temporal support **and** reduces the source sample to {len(brb):,}
blocks. Those two effects cannot be separated from this diagnostic alone, and no
further experiment is run here to separate them. The 900-s rows are therefore
**not used to determine Gate 2**, and no causal reading of the lower containment
is offered in either direction.

---

## Part 3 — What can a Brazil classifier's output mean on HK?

| Class | Physically applicable to SQ1? | Can a prediction be interpreted? |
|---|---|---|
| **Short-Circuit** | The condition can occur | **No.** Assignment rests on string-level voltage behaviour HK does not measure. Any short-circuit prediction would be an extrapolation from aggregate power alone, and Objective 1 showed this class is exactly the one that collapses without voltage. |
| **Open-Circuit** | The condition can occur | **No**, for the same reason. |
| **Degradation** | Yes in principle | **Weakly.** Brazil's was induced by a resistive load bank over ~10 minutes; HK degradation would develop over months. Same label, different physical process and timescale. |
| **Shadowing** | Yes — urban campus with surrounding buildings | **Most defensibly of the four.** Reduced output under partial obstruction is visible in aggregate power, and does not require string measurement. |
| **Normal** | Yes | **Yes.** |

### Recommendation

**Use the output as a binary Normal-versus-anomalous screening signal, and
report the predicted class distribution separately as a diagnostic of transfer
behaviour rather than as a claim about HK conditions.**

Reasoning: two of the five classes cannot be assigned on the available sensing,
and a third maps onto a different physical process. Retaining the fault-type
distinction would imply a discrimination the input features cannot support.
Collapsing to binary keeps exactly the claim the data can carry — that a
timestamp looks unlike Brazil's Normal class — while the class distribution is
still worth reporting because a collapse toward one class is itself the
diagnostic that distinguishes a domain mismatch from a working transfer.

Temporal clustering of flags is retained as a secondary check, since a screening
signal that fires in coherent runs aligned with known HK context is more
credible than one firing at dispersed isolated timestamps.

---

## Part 4 — Reportable metrics without labels

**Computable and reportable:**

1. Predicted class distribution on HK, compared against Brazil's class priors.
2. Temporal clustering of predicted non-Normal flags — dispersed or
   concentrated, and the run-length distribution.
3. Alignment of flags with known HK context: the Typhoon Saola proxy event
   window (1-15 Sept 2023), the March-October 2023 sensor-fault window, and the
   2023 residual shift identified in Objective 2.

   **September lies inside the March-October measured-irradiance corruption
   period.** Because the shared contract includes measured HK irradiance, any
   Objective 3 result over the Saola window is computed on a knowingly corrupted
   input. It can therefore be reported **only as a corrupted-input stress test,
   never as evidence of PV-event detection.** The same caution applies to any
   comparison against Objective 2 screening signals over that window: both
   signals share the same corrupted input, so agreement between them is not
   independent corroboration.
4. Agreement between the transferred classifier's flags and Objective 2's
   anomaly-screening flags — one screening signal against another, **neither
   validated against truth**.

**Not computable, and must not be reported:** accuracy, precision, recall, F1,
or confusion matrices on HK. HK has no per-timestep fault labels, so every one
of these would presuppose a ground truth that does not exist.

A banned-phrase guard is specified for the implementation stage, covering the
{len(BANNED)} label-presuming phrases enumerated in the `BANNED` constant in
`objective3_feature_audit.py`. They are deliberately not reproduced here: this
document is itself passed through the guard, and listing them would trip it on
its own prose. The guard is already active in this audit script.

---

## Part 5 — Proposed minimal experiment

### 5a. Shared feature set

> ### ⚠ Headline finding — the shared contract is {n_shared} features
>
> **Capacity-normalised power, and irradiance. Nothing else survives the
> intersection.** This is below the prompt's three-feature threshold, and
> neither of the two survivors is clean:
>
> 1. **Power carries an unresolved DC-versus-AC mismatch.** Brazil's
>    `p_dc_computed` is DC string power; HK's `power_W` is AC active power at
>    the inverter output. No inverter efficiency was assumed to bridge them, so
>    the gap is carried, not closed.
> 2. **HK's irradiance plane is unresolved** — tilted-plane versus horizontal is
>    undocumented, as is the weather-station-to-SQ1 pairing.
> 3. **The discrimination that mattered most is absent.** Objective 1 showed the
>    strongest fault discrimination — Short-Circuit above all — depended heavily
>    on string-level voltage and current that HK does not record at any station.
>
> Taken together, **this makes the semantic gate marginal before distribution
> overlap is even considered.** Two features, one semantically mismatched, one
> geometrically undocumented, and the classes of interest stripped of the
> measurements that separated them.

Whether the experiment remains worthwhile on those terms is addressed in the
gate summary below and in `objective3_open_decisions.md`; it is not a decision this
audit takes alone.

The derived clearness-like ratio (normalised power / irradiance) is available as
a third candidate but is not adopted here, since it inherits both parents'
caveats and adds no independent information.

### 5b. Normalisation

Power divided by rated capacity — Brazil by 5.00 kW with 5.28 kW carried as a
sensitivity, HK by 27.6 kW from the SQ1 TTL entry. Irradiance is left in native
W/m², since both are already the same unit. Any further scaling (standardisation)
is fitted on **Brazil training data only** and applied unchanged to HK.

#### Target-side handling of HK power — required, and easy to get wrong

The frozen `POWER_PLUS_IRR` configuration fits its `SimpleImputer` and
`StandardScaler` on **raw Brazil watts** (`p_dc_computed`), not on a normalised
quantity. **Feeding HK power divided by 27.6 kW into that parameterisation would
be wrong** — the scaler expects values on a Brazil-watt scale and would receive
values near 0-1, standardising them to a large constant negative offset and
destroying the transfer.

The correct target-side step expresses HK power in **Brazil-capacity-equivalent
watts** before the Brazil-fitted preprocessing:

```
p_hk_brazil_equivalent_W = power_W * (CAPACITY_BR / CAPACITY_HK)
                         = power_W * (5000.0 / 27606.0)     # = 0.18112005
                         = power_W * (5280.0 / 27606.0)     # = 0.19126277  (5.28 kW variant)
```

Irradiance (`irr_meas`, W/m²) is passed through **unchanged**, since Brazil's
`irr_panel_wm2` is already in native W/m².

This is exactly equivalent to the capacity-normalised formulation, because the
Brazil-fitted scaler is affine. Writing `mu`, `sd` for the Brazil training-fold
mean and standard deviation of raw watts:

```
(p_hk/C_HK - mu/C_BR) / (sd/C_BR)  ==  (p_hk*(C_BR/C_HK) - mu) / sd
```

The left side is the shared-contract view (both domains capacity-normalised, the
scaler refitted on Brazil normalised power); the right side is what the frozen
raw-watt parameterisation computes when fed Brazil-capacity-equivalent watts.
They are algebraically identical, and were verified numerically to agree to
within 8.9e-16. **Use the right-hand form**, because it preserves the frozen
parameterisation exactly and requires nothing to be refitted.

Note the direction of the conversion: HK power is scaled **into** Brazil's
capacity frame, not the reverse. This carries no assumption about conversion
efficiency or system losses — it is a unit change on the capacity axis only.

### 5c. Model and the source-performance gate

**Source-performance gate: EVIDENCED from frozen outputs, without retraining.**

The earlier draft of this spec held the gate PENDING on the ground that no
nearby configuration should be substituted for the untested shared contract.
That reasoning was right in general but wrong in this specific case, and the
difference was established by reading the pipeline rather than the feature name.

Objective 1's `Computed power + measured irradiance` is defined in
`extended_ablation.py` as `POWER_PLUS_IRR = ["p_dc_computed", "irr_panel_wm2"]`.
The proposed shared representation is `p_dc_computed / capacity` and
`irr_panel_wm2`. The two differ only by a **constant positive divisor on one
feature**, and the saved winning model is MLP, which is in `SCALE_SENSITIVE`,
so a fold-fitted `StandardScaler` sits between the raw feature and the model:

| Pipeline stage | Effect of dividing one feature by a constant C > 0 |
|---|---|
| `inf -> NaN` | unchanged — `x/C` is finite exactly when `x` is |
| `SimpleImputer(median)`, training fold only | `median(x/C) = median(x)/C` — equivariant, same point in scaled space |
| `StandardScaler`, training fold only | `(x/C - mean(x)/C) / (sd(x)/C) = (x - mean(x)) / sd(x)` — **C cancels exactly** |
| Fold allocation, seed, hyperparameters, MLP downsampling indices | identical; none depends on feature scale |

The matrix reaching the MLP is therefore **mathematically identical**, so the
two representations are information-equivalent for the actual saved pipeline.
The equivalence is a property of this pipeline, not of the feature list: for a
model **not** in `SCALE_SENSITIVE` the argument would rest instead on tree
invariance to monotone transforms, and for a pipeline without a scaler it would
not hold at all. The one honest caveat is floating point — `x/C` then
standardise is exact in real arithmetic but not guaranteed bit-identical in
IEEE-754, with relative differences at machine epsilon.

Accordingly the **five-class** component of the gate is answered by the frozen
result **Macro F1 0.6840 (MLP)**, obtained under the identical group-disjoint
5-fold protocol and fold allocation. The exclusion of `POWER_PHYSICS` and other
nearby sets still stands — those genuinely add physics-derived or environmental
inputs the contract excludes.

#### No fitted model artefact survives — what implementation must therefore do

A read-only search of the project for serialised model artefacts (`.pkl`,
`.joblib`, `.pt`, `.h5`, `.keras`, `.sav`, `.onnx`) returns **three files, all
irrelevant**: `lstm_model.pt`, `lstm_no_sensor.pt` and `lstm_with_sensor.pt`,
dated 2026-07-03, all under `outputs/lstm_with_sensor/` and `outputs/no_sensor/`
— Hong Kong LSTMs from the historical pipeline that Objective 2 established was
contaminated, and which are on this audit's forbidden-input list. They are not
Brazil models, not `POWER_PLUS_IRR`, and not usable.

`extended_ablation.py` and `pv_fault_classifier.py` contain **no model
persistence at all** — no serialisation call of any kind: no `joblib`, no
`pickle`, and no deep-learning checkpoint write. What
survives from Objective 1 is out-of-fold predictions, per-fold and per-class
metrics, and confusion-matrix counts.

Therefore **no fitted, transferable `POWER_PLUS_IRR` MLP exists and none can be
recovered.** Objective 3 implementation must fit the already-fixed Brazil
`POWER_PLUS_IRR` configuration — same feature list, same fold allocation, same
seed, same hyperparameters, same imputer and scaler discipline. This is
**implementation of a frozen design, not new model selection.** No architecture,
feature set, hyperparameter or protocol choice is reopened, and the fitted result
must reproduce the frozen five-class Macro F1 of 0.6840 as a correctness check
before any HK data is touched.

#### The primary HK endpoint is locked

The transferred model is the **same five-class source model**, with its
predictions **collapsed to Normal versus non-Normal for interpretation**. The
collapse happens at the prediction stage, downstream of the model.

**A newly designed binary classifier must not be substituted.** The Gate 3
evidence rests on an information-equivalence argument about one specific frozen
five-class configuration and its saved out-of-fold predictions. Training a
binary model instead would be a different configuration whose performance is not
evidenced by anything frozen, and it would break the equivalence that
established Gate 3 in the first place. The collapsed binary figures in this
document describe the existing five-class model interpreted as Normal versus
non-Normal — they are not, and must not be reported as, results from a
separately trained binary classifier.

### 5d. HK application

Objective 2's sensor-free eligibility mask (`ghi_clear >= 50 W/m²` and
`solar_zenith < 85°`).

#### Binding rule: the primary experiment is TRUE DIRECT TRANSFER

**No target-domain adaptation of any kind.** The primary transferred classifier
uses **Brazil-fitted preprocessing unchanged on HK**: the Brazil training-fold
median imputer, the Brazil training-fold `StandardScaler`, and the Brazil-fitted
model, with HK power converted to Brazil-capacity-equivalent watts by the fixed
constant in Part 5b and irradiance passed through in native W/m². Nothing is
refitted, recentred, rescaled or recalibrated on HK.

This is binding and rules out, for the primary result: recentring the
normalisation on HK's own operating distribution; standardising HK against HK
statistics; threshold calibration against an HK baseline; and any quantile,
mean or variance matching between the domains. All of these are target-domain
adaptation and are **out of scope for Objective 3.**

The cross-domain normalised-power-per-irradiance offset documented in Part 2b is
**part of the transfer-compatibility result, not a prerequisite to resolve before
observing transfer behaviour.** Correcting for it in advance would mean reporting
the behaviour of an adapted model while describing it as direct transfer.

An HK-relative descriptive comparator — for example flag rates referenced to
HK's own operating distribution — **may be reported separately** and is useful
context. It must not alter the features or preprocessing supplied to the
transferred classifier, and it must never be presented as direct transfer.

**Because the shared contract includes measured HK irradiance, the primary
transfer application is restricted to the clean-sensor periods — January,
February, November and December 2023.** Full-2023 application may be reported
only as a separately labelled corrupted-input stress test, never as the primary
result. Feeding a knowingly corrupted sensor signal into the primary experiment
would confound target-sensor failure with transfer failure.

### 5e. Sampling reconciliation

Three options:

| Option | Effect |
|---|---|
| Train on native 1 Hz Brazil, apply to HK 15-min | Preserves all Brazil fault events and the full label set, but source and target see different temporal support; the model meets narrower-variance inputs at application time |
| Aggregate Brazil to 900-sample blocks and train on those | Matches support, but reduces Brazil from {len(br):,} rows to **{N_SINGLE:,} usable labelled blocks** (see the continuity accounting below), and a ~600-second induction can sit entirely inside one 900-second block |
| Train on native, apply to HK, and report the Part 2a diagnostic alongside | Keeps the source task intact while quantifying how much of the observed distribution discrepancy is support-driven |

#### Continuity accounting for the 900-second blocks

`fault_dataset.csv` is a **filtered** dataset, so `sample_index`
contains gaps and 900 consecutive dataframe rows do not necessarily span 900
seconds. Blocks are therefore built by splitting first and aggregating second:
each recording group is divided into maximal contiguous `sample_index` runs
(cut wherever `diff() != 1`), and non-overlapping 900-sample blocks are formed
**inside** each run, discarding only the trailing remainder of each run. Every
retained block is asserted to belong to one recording group, lie within one
contiguous run, contain exactly 900 rows, and satisfy `np.diff(sample_index)==1`
throughout. Blocks lost to filtered-data gaps are **not** counted as valid
temporal-support matches.

| Quantity | Count |
|---|---|
| Recording groups | {n_groups} |
| Contiguous `sample_index` runs | {N_RUNS:,} |
| Discontinuities inside groups | {N_DISCONT:,} |
| 900-row blocks under naive row-based chunking | {N_NAIVE:,} |
| **Valid continuous 900-second blocks retained** | **{N_VALID:,}** |
| — of which mixed-label (left unlabelled) | {N_MIXED:,} |
| — of which single-label, usable for supervised aggregation | **{N_SINGLE:,}** |

**Implication for temporal aggregation as a transfer-training strategy.**
Matching HK's 900-second support costs the source task {N_NAIVE - N_VALID:,}
blocks to continuity alone and a further {N_MIXED:,} to label mixing, leaving
{N_SINGLE:,} labelled training rows drawn from {n_groups} recording groups —
too few to support the group-disjoint 5-fold protocol at anything like the
frozen Objective 1 statistical resolution, and far too few for five classes.

The class composition of the survivors is decisive, and it was verified
directly from the post-continuity blocks rather than inferred:

| Class | Pure 900-s blocks | Longest event in register |
|---|---|---|
| Normal | 323 | 56,719 samples |
| Shadowing | 130 | 6,555 samples |
| Degradation | **3** | 2,695 samples |
| Short-Circuit | **0** | **619 samples** |
| Open-Circuit | **0** | **606 samples** |

The mechanism is simply that **standard controlled inductions are ~600 seconds,
which is shorter than a 900-second support block.** A short event cannot
generate a pure block at all: any block overlapping it necessarily also contains
Normal seconds, so it carries two distinct `f_nv` values, is mixed-label, and is
excluded by the existing rule. There is no separate dilution effect to invoke —
a 600-second fault sitting inside a 900-second block *is* a mixed-label block by
construction. Temporal aggregation therefore removes short controlled events
from the usable supervised set entirely.

The only controlled-class survivors are the **3 Degradation blocks**, and they
trace to the two anomalous long-duration degradation events in the register
(event 1272, 1,779 samples; event 1430, 2,695 samples) — the only controlled
events anywhere in the dataset exceeding 900 samples. They fall in just 2 of the
16 recording groups.

**Grouped five-class training on 900-second blocks is therefore not merely
under-powered — it is impossible.** Two of the five classes have no training
examples at any capacity, and a third has three examples spanning two groups,
which cannot survive a group-disjoint 5-fold split.

**Recommendation: the third option.** Train on native 1 Hz Brazil, apply to HK,
and carry the Part 2a block statistics as a reported diagnostic of how much of
the observed distribution discrepancy is support-driven rather than domain-
driven. Note also that Brazil has no timestamps, so any aggregation is by sample
position within a recording group, never by clock.

### 5f. The three gates

| Gate | Source | Status |
|---|---|---|
| **Semantic** | Part 1 | **FAILS for fault-type transfer, PARTIAL for binary screening.** Neither shared quantity is physically equivalent — both are **proxy-comparable only** (DC vs AC; unresolved irradiance plane and station pairing). Two of five classes cannot be assigned on the available sensing, and the string features carrying the strongest discrimination are absent. |
| **Domain overlap** | Part 2 | **MARGINAL — a qualitative design judgement, not a predefined statistical threshold.** No pass/fail cut-off is attached to any KS or overlap value, and Gate 2 is not reduced to a single statistic. Populations differ by quantity and are named individually in Part 2 and 2b: power on HK eligible full 2023, irradiance on HK clean-sensor months. Full-2023 measured irradiance is excluded as a sensor-corruption diagnostic. See `objective3_distribution_comparison.csv`, `objective3_joint_support.csv` and the figures. |
| **Source performance** | Part 5c | **EVIDENCED, no retraining required.** Five-class Macro F1 **0.6840**; collapsed binary anomalous F1 **0.8268**, balanced accuracy **0.8749**. Both read from frozen Objective 1 outputs under the identical grouped protocol, via a verified information-equivalence argument — not inferred from feature names. |

### 5g. Distinguishing a negative result from a broken mapping

Specified **before anything is run**, because it is the crux of whether a
negative result is reportable:

1. **Source-performance floor.** If the two-feature Brazil model cannot itself
   discriminate on Brazil under the frozen protocol, a failed HK transfer says
   nothing about the domains — the mapping never carried discrimination to
   begin with. This is why gate 3 must be settled first and must not be
   estimated.
2. **Sanity transfer within Brazil.** Apply the restricted model across
   held-out Brazil recording groups. Success there with failure on HK isolates
   the domain change; failure in both indicates the feature set, not the domain.
3. **Input-range audit at application time.** Record the fraction of HK rows
   falling outside Brazil's observed support per feature. Widespread
   extrapolation means the classifier is being asked about inputs it never saw,
   which is a mapping problem rather than evidence about HK.
4. **Degenerate-output check.** If the predicted distribution collapses to a
   single class, distinguish collapse-to-Normal (consistent with HK genuinely
   operating normally) from collapse-to-a-fault-class (a strong indicator of
   distribution shift rather than a finding about HK).
5. **Corrupted-input contrast.** Compare the clean-sensor primary result with
   the full-2023 stress test. A large divergence attributable to the known
   March-October corruption is a target-sensor artefact, not a transfer result.

**Reading the full-2023 irradiance KS correctly.** The full-2023 comparison
gives a *lower* KS (0.276) than the clean-sensor gate comparison (0.340). This
must not be read as better physical domain agreement, and equally must not be
asserted to be definitely caused by corruption. The defensible statement is
narrower: the full-2023 population **includes the known March-October
corrupted-sensor period, so its KS cannot be interpreted as evidence of better
physical agreement between the domains.** The comparison is not clean enough to
support an interpretation in either direction, which is precisely why it is
excluded from the gate.

Only if the source-performance floor is cleared, the within-Brazil sanity
transfer succeeds, and extrapolation is limited can a negative HK outcome be
attributed to genuine domain difference.

---

## Classification of findings

CONFIRMED: the AC/DC asymmetry, the absence of HK string measurement, the
sampling-support mismatch, the capacity figures, the March-October sensor
corruption, and Objective 1's string-signal results.
SUPPORTED ASSUMPTION: irradiance comparability (HK plane inert at tilt 0), the
weather-station pairing to SQ1.
UNRESOLVED: the DC/AC residual magnitude, the derived clearness-like ratio's
usefulness, and the source-performance gate.
""", "design spec")

OPEN = check(f"""# Objective 3 — Open Decisions

Requiring approval before any implementation. This audit trained nothing and
takes none of these decisions unilaterally.

## 1. Does a two-feature contract justify proceeding at all?

The shared contract is **capacity-normalised power and irradiance**. Everything
carrying Brazil's fault discrimination — string voltage, string current, their
imbalance ratios, expected power, residual and ratio — is absent in Hong Kong.

Three courses:

| Option | What it yields |
|---|---|
| Do not proceed | Objective 3 is reported as a design-stage negative result: the cross-domain feature contract is too thin to support fault-type transfer, evidenced rather than asserted |
| Proceed as a documented negative result | Run the minimal experiment expecting failure, with Part 5g in place so the failure is attributable |
| Proceed as binary screening only | Discard fault-type prediction, transfer a Normal-versus-anomalous signal, report agreement with Objective 2 screening |

**Recommendation: the third — binary screening — and it is now actionable.**
The source-performance gate is settled from frozen evidence (decision 2), and it
lands more favourably for binary screening than the five-class figure suggested:
collapsing the same model to Normal-versus-anomalous gives anomalous F1 0.8268
and balanced accuracy 0.8749, against a five-class Macro F1 of 0.6840. The
five-class weakness is concentrated in *distinguishing fault types* — precisely
the capability Part 3 already rules out on HK — rather than in separating normal
from anomalous, which is the capability the transfer endpoint actually needs.

Option one (do not proceed) is no longer the strongest reading, because the
mapping demonstrably carries screening discrimination on the source domain. A
failed HK transfer would therefore be attributable rather than uninterpretable,
which is what makes a negative result publishable.

## 2. Source-performance gate — RESOLVED, no longer an open decision

This no longer requires training anything. `Computed power + measured
irradiance` is information-equivalent to the shared contract for the saved MLP
pipeline (a constant positive divisor that the fold-fitted StandardScaler
cancels exactly), so the frozen Objective 1 outputs already answer it:

- **five-class** Macro F1 **0.6840**
- **collapsed binary** (Normal vs anomalous) anomalous F1 **0.8268**, balanced
  accuracy **0.8749**, Normal recall **0.8703**, anomalous recall **0.8794**,
  anomalous precision **0.7801**

Both come from the pooled out-of-fold predictions under the identical
group-disjoint protocol. The binary figures describe the **existing five-class
model interpreted as Normal vs non-Normal** — not a separately trained binary
classifier. No numerical pass mark is invented here: the evidence is reported
and final acceptance remains an implementation decision.

## 2b. Which Brazil capacity is used is now the live normalisation question

See decision 4 — it moves from a formality to a real choice only for the joint
support diagnostic, since the source-performance gate is invariant to it.

## 3. Is the derived clearness-like ratio admitted?

Adding `normalised power / irradiance` would make the contract three features.
It adds no independent information and inherits the DC/AC residual, but may help
a low-capacity model. Not adopted here.

## 4. Which Brazil capacity is authoritative?

5.00 kW (paper) or 5.28 kW (16 x 330 W nameplate). Objective 1 found results
insensitive (Macro F1 difference 0.0069), and the source-performance gate is
mathematically invariant to the choice, since the divisor cancels in the
scaler. The choice does move the joint-support diagnostic, but only slightly —
hull containment 95.68% vs 95.49% and occupied-bin containment 89.97% vs 88.24%
for Brazil all-classes. **It does not change Gate 2.** Both are carried in the
comparison files; one must still be fixed for implementation.

## 5. Does binary screening still need the fault-type head?

If the recommendation in Part 3 is accepted, the Brazil model could be trained
directly as a binary Normal-versus-anomalous classifier rather than five-class
and collapsed afterwards. Training binary directly is cleaner; collapsing
afterwards preserves the class distribution as a transfer diagnostic. This audit
assumes five-class-then-collapse, to keep the diagnostic.
""", "open decisions")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
if "--verify-only" in sys.argv:
    print(f"\nBanned-phrase guard: {len(_checked)} strings checked, all PASS")
    print("\nVERIFY-ONLY MODE — no files written. Would write:")
    for f in ["objective3_semantic_comparability.csv", "objective3_distribution_comparison.csv",
              "objective3_distribution_overlap.png", "objective3_distribution_overlap.pdf",
              "objective3_design_spec.md", "objective3_open_decisions.md"]:
        print(f"    {OUT_DIR.name}/{f}")
    plt.close(fig)
    sys.exit(0)

sem.to_csv(OUT_DIR / "objective3_semantic_comparability.csv", index=False)
dist.to_csv(OUT_DIR / "objective3_distribution_comparison.csv", index=False)
fig.savefig(OUT_DIR / "objective3_distribution_overlap.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "objective3_distribution_overlap.pdf", bbox_inches="tight")
plt.close(fig)
(OUT_DIR / "objective3_design_spec.md").write_text(SPEC)
(OUT_DIR / "objective3_open_decisions.md").write_text(OPEN)
print(f"\nWrote 6 files to {OUT_DIR}")
with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"objective3_feature_audit.py | input: fault_dataset.csv, "
            f"objective2_grid.csv, extended_ablation_summary.csv | "
            f"output: 6 files in Objective3_Outputs/objective3_design/ | "
            f"notes: design audit only, nothing trained; direct transfer, "
            f"no target adaptation; shared contract = "
            f"{n_shared} features; source-performance gate EVIDENCED "
            f"from frozen outputs, no fitted artefact survives\n")
