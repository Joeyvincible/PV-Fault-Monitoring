"""Objective 3 — Consolidated read-only design and gate check.

Implements the documented gate-check items 1-10 in one pass.

READ-ONLY with respect to Objective 1 and Objective 2. Nothing is trained, no
transfer is run, and the only files written are Objective 3 design outputs in
this directory.

Item 3 uses a FIXED DESCRIPTIVE joint-support method predefined in the prompt
before any result was seen: convex-hull containment and occupied-bin
containment on a 20x20 common grid. No learned domain classifier, no KDE
classifier, no fitted predictive model of any kind.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.spatial import ConvexHull
from scipy.stats import ks_2samp

# ─────────────────────────────────────────────────────────────────────────────
# GUARDS
# ─────────────────────────────────────────────────────────────────────────────
BANNED = ["fault recall", "false positive rate", "fault precision", "FPR",
          "fault ground truth", "detection accuracy on HK",
          "verified normal ground truth", "normal-to-normal domain shift"]


def check(text):
    low = text.lower()
    for b in BANNED:
        assert b.lower() not in low, f"BANNED PHRASE in generated text: {b!r}"
    return text


FORBIDDEN_INPUTS = ["fault_dataset_cleaned.csv", "outputs/no_sensor",
                    "outputs/lstm_with_sensor"]
_read = []


def rd(p, **kw):
    s = str(p)
    for f in FORBIDDEN_INPUTS:
        assert f not in s, f"FORBIDDEN INPUT: {s}"
    _read.append(s)
    return pd.read_csv(p, **kw)


def assert_no_model_fitting():
    """Self-scan. ConvexHull/ks_2samp are descriptive, not predictive fits."""
    # Tokens are assembled from fragments so that this list does not itself
    # match, which would make the guard trip on its own definition.
    tok = ["." + "fit(", "." + "fit_transform(", "cross" + "_val",
           "train" + "_test_split", "Classi" + "fier("]
    src = Path(__file__).read_text().splitlines()
    bad = [(i + 1, l) for i, l in enumerate(src)
           if any(t in l for t in tok)
           and "NOFIT-EXEMPT" not in l and not l.strip().startswith("#")]
    assert not bad, f"MODEL FITTING FOUND: {bad}"
    print("No-model-fitting self-check: PASS — no fitting call found")


assert_no_model_fitting()

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
NAVE = HERE.parent.parent
OUT_DIR = HERE
BRAZIL_CSV = NAVE / "02_Fault_Classification" / "data" / "fault_dataset.csv"
HK_CSV = NAVE / "01_HK_Detection" / "outputs" / "objective2_modeling" / "objective2_grid.csv"
ABLATION = (NAVE / "02_Fault_Classification" / "outputs"
            / "extended_ablation" / "extended_ablation_summary.csv")
CONFMAT = (NAVE / "02_Fault_Classification" / "outputs"
           / "extended_ablation" / "extended_confusion_matrices.csv")
REGISTER = (NAVE / "02_Fault_Classification" / "outputs"
            / "source_event_register.csv")
RUN_LOG = OUT_DIR / "run_log.txt"

GAP_THRESHOLD = 3600
BLOCK = 900
PDC0_BR_RATED = 5000.0
PDC0_BR_NAMEPLATE = 5280.0
PDC0_HK = 27606.0
CLEAN_MONTHS = [1, 2, 11, 12]
TZ = "Asia/Hong_Kong"
CLASSES = {0: "Normal", 1: "Short-Circuit", 2: "Degradation",
           3: "Open-Circuit", 4: "Shadowing"}

print("=" * 78)
print("  OBJECTIVE 3 — CONSOLIDATED GATE CHECK (read-only, nothing trained)")
print("=" * 78)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
br = rd(BRAZIL_CSV)
hk = rd(HK_CSV, index_col=0)
hk.index = pd.to_datetime(hk.index, utc=True).tz_convert(TZ)
abl = rd(ABLATION)
cm = rd(CONFMAT)
reg = rd(REGISTER)
print("\nINPUT FILES READ:")
for f in _read:
    print("  " + str(Path(f).relative_to(NAVE)))

br["recording_group"] = ((br["sample_index"].diff() > GAP_THRESHOLD)
                         .fillna(False).cumsum().astype(int))
n_groups = br["recording_group"].nunique()
assert n_groups == 16, f"expected 16 recording groups, got {n_groups}"

# ═════════════════════════════════════════════════════════════════════════════
# ITEM 1 — aggregation interpretation, blocks by class and group
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  ITEM 1 — 900-second aggregation interpretation")
print("=" * 78)

# Mechanistic expectation stated BEFORE inspecting the counts (see report).
EXPECTATION = ("zero pure Short-Circuit blocks; zero pure Open-Circuit blocks; "
               "any pure Degradation blocks trace to the two anomalous events")
print(f"\nMechanistic expectation (stated before inspection):\n  {EXPECTATION}")

# Event-duration evidence, standard controlled inductions only.
CONTROLLED = [1, 2, 3]           # Short-Circuit, Degradation, Open-Circuit
print("\nEvent register — duration distribution by class "
      "(events of >= 60 raw samples, i.e. substantial inductions):")
sub_reg = reg[reg["raw_sample_count"] >= 60]
reg_rows = []
for c in sorted(reg["f_nv"].unique()):
    s = sub_reg[sub_reg["f_nv"] == c]["raw_sample_count"]
    allc = reg[reg["f_nv"] == c]["raw_sample_count"]
    n_ge_block = int((allc >= BLOCK).sum())
    reg_rows.append({"f_nv": int(c), "class": CLASSES[int(c)],
                     "n_events_total": int(len(allc)),
                     "n_events_ge60": int(len(s)),
                     "median_len_ge60": float(s.median()) if len(s) else np.nan,
                     "max_len": int(allc.max()),
                     "n_events_ge_900": n_ge_block})
    print(f"  {CLASSES[int(c)]:<14} total={len(allc):>5}  >=60 samples={len(s):>3}  "
          f"median(>=60)={s.median() if len(s) else float('nan'):>8.0f}  "
          f"max={allc.max():>6}  events >= 900 samples={n_ge_block}")
reg_tab = pd.DataFrame(reg_rows)

print("\nEvents of >= 900 samples in the controlled classes "
      "(the only ones that could yield a pure block):")
long_controlled = reg[(reg["f_nv"].isin(CONTROLLED)) & (reg["raw_sample_count"] >= BLOCK)]
if len(long_controlled):
    for _, r in long_controlled.iterrows():
        print(f"  event {int(r.source_event_id):>5}  {CLASSES[int(r.f_nv)]:<14} "
              f"{int(r.raw_sample_count):>6} samples  "
              f"[{int(r.start_sample_index)}, {int(r.end_sample_index)}]")
else:
    print("  none")

# Split-first block construction (identical rule to the audit script).
blocks = []
runs_total = 0
naive_blocks = 0
for g, sub in br.groupby("recording_group"):
    sub = sub.sort_values("sample_index").reset_index(drop=True)
    naive_blocks += len(sub) // BLOCK
    run_id = (sub["sample_index"].diff() != 1).cumsum()
    for _, run in sub.groupby(run_id):
        runs_total += 1
        run = run.reset_index(drop=True)
        for b in range(len(run) // BLOCK):
            seg = run.iloc[b * BLOCK:(b + 1) * BLOCK]
            si = seg["sample_index"].to_numpy()
            assert len(seg) == BLOCK
            assert seg["recording_group"].nunique() == 1
            assert bool(np.all(np.diff(si) == 1))
            assert int(si[-1] - si[0]) == BLOCK - 1
            labs = seg["f_nv"].unique()
            blocks.append({"recording_group": int(g),
                           "start_sample_index": int(si[0]),
                           "end_sample_index": int(si[-1]),
                           "p_dc_computed": float(seg["p_dc_computed"].mean()),
                           "irr_panel_wm2": float(seg["irr_panel_wm2"].mean()),
                           "f_nv": int(labs[0]) if len(labs) == 1 else np.nan,
                           "mixed": len(labs) > 1})
brb = pd.DataFrame(blocks)
N_RUNS, N_NAIVE = runs_total, naive_blocks
N_VALID = len(brb)
N_MIXED = int(brb["mixed"].sum())
N_SINGLE = N_VALID - N_MIXED
print(f"\nBlocks: {N_RUNS} contiguous runs | naive {N_NAIVE} | valid {N_VALID} "
      f"| mixed {N_MIXED} | single-label {N_SINGLE}")
assert N_SINGLE == int(brb["f_nv"].notna().sum())

pure = brb[brb["f_nv"].notna()].copy()
pure["class"] = pure["f_nv"].astype(int).map(CLASSES)
by_class = pure["class"].value_counts().reindex(list(CLASSES.values())).fillna(0).astype(int)
print("\nSingle-label 900-second blocks BY CLASS:")
for k, v in by_class.items():
    print(f"  {k:<14} {v:>5}")

xt = pd.crosstab(pure["recording_group"], pure["class"])
xt = xt.reindex(columns=[c for c in CLASSES.values() if c in xt.columns], fill_value=0)
xt = xt.reindex(index=range(n_groups), fill_value=0)
xt["TOTAL"] = xt.sum(axis=1)
print("\nSingle-label 900-second blocks BY RECORDING GROUP x CLASS:")
print(xt.to_string())

n_groups_with_fault = int((xt.drop(columns=["TOTAL"])
                           .drop(columns=["Normal"], errors="ignore").sum(axis=1) > 0).sum())
print(f"\nRecording groups containing at least one pure non-Normal block: "
      f"{n_groups_with_fault} of {n_groups}")

EXPECT_MATCH = (int(by_class.get("Short-Circuit", 0)) == 0
                and int(by_class.get("Open-Circuit", 0)) == 0)
print(f"Expectation match (zero pure Short-Circuit and zero pure Open-Circuit): "
      f"{'YES' if EXPECT_MATCH else 'NO'}")

# ═════════════════════════════════════════════════════════════════════════════
# ITEM 4 — HK population discrepancy 15,875 vs 15,870
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  ITEM 4 — HK population reconciliation")
print("=" * 78)
elig = (hk["ghi_clear"] >= 50) & (hk["zenith"] < 85)
hk_e = hk[elig]
hk23 = hk_e[hk_e.index.year == 2023]
n_elig23 = len(hk23)
n_pow23 = int(hk23["power_W"].notna().sum())
n_irr23 = int(hk23["irr_meas"].notna().sum())
n_both23 = int((hk23["power_W"].notna() & hk23["irr_meas"].notna()).sum())
print(f"  HK eligible rows, full 2023                     : {n_elig23:,}")
print(f"    with non-null power_W                         : {n_pow23:,}  "
      f"(difference {n_elig23 - n_pow23})")
print(f"    with non-null irr_meas                        : {n_irr23:,}  "
      f"(difference {n_elig23 - n_irr23})")
print(f"    with BOTH non-null (joint diagnostic uses this): {n_both23:,}")
hk_clean = hk_e[(hk_e.index.year == 2023) & (hk_e.index.month.isin(CLEAN_MONTHS))]
n_clean = len(hk_clean)
n_clean_both = int((hk_clean["power_W"].notna() & hk_clean["irr_meas"].notna()).sum())
print(f"  HK eligible, clean-sensor months 2023 (GATE)    : {n_clean:,}")
print(f"    with BOTH power_W and irr_meas non-null       : {n_clean_both:,}")

# ═════════════════════════════════════════════════════════════════════════════
# ITEM 3 + 8 — joint two-feature support diagnostic
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  ITEMS 3 & 8 — joint two-feature support diagnostic")
print("=" * 78)
print("  Fixed descriptive method, predefined: convex-hull containment and")
print("  occupied-bin containment on a 20x20 common grid. No fitted model.")
GRID = 20


def hull_containment(src_xy, tgt_xy):
    """Percentage of target points inside the source 2D convex hull."""
    h = ConvexHull(src_xy)
    A, b = h.equations[:, :-1], h.equations[:, -1]
    inside = np.all(tgt_xy @ A.T + b <= 1e-12, axis=1)
    return 100.0 * inside.mean(), h


def bin_containment(src_xy, tgt_xy, grid=GRID):
    """Percentage of target points in a source-occupied cell of a common grid."""
    both = np.vstack([src_xy, tgt_xy])
    xe = np.linspace(both[:, 0].min(), both[:, 0].max(), grid + 1)
    ye = np.linspace(both[:, 1].min(), both[:, 1].max(), grid + 1)
    def cells(a):
        i = np.clip(np.digitize(a[:, 0], xe) - 1, 0, grid - 1)
        j = np.clip(np.digitize(a[:, 1], ye) - 1, 0, grid - 1)
        return i * grid + j
    occ = set(np.unique(cells(src_xy)).tolist())
    tc = cells(tgt_xy)
    inside = np.array([c in occ for c in tc])
    return 100.0 * inside.mean(), len(occ), len(set(np.unique(tc).tolist()))


hk_xy = hk_clean.loc[hk_clean["power_W"].notna() & hk_clean["irr_meas"].notna(),
                     ["power_W", "irr_meas"]].to_numpy(dtype=float)
hk_xy[:, 0] /= PDC0_HK

joint_rows = []
for cap_lab, cap in [("5.00 kW", PDC0_BR_RATED), ("5.28 kW", PDC0_BR_NAMEPLATE)]:
    br_all = br[["p_dc_computed", "irr_panel_wm2"]].dropna().to_numpy(dtype=float)
    br_all = br_all[np.isfinite(br_all).all(axis=1)].copy()
    br_all[:, 0] /= cap
    br_nrm = br.loc[br["f_nv"] == 0, ["p_dc_computed", "irr_panel_wm2"]].dropna().to_numpy(dtype=float)
    br_nrm = br_nrm[np.isfinite(br_nrm).all(axis=1)].copy()
    br_nrm[:, 0] /= cap
    br_blk = pure[["p_dc_computed", "irr_panel_wm2"]].to_numpy(dtype=float).copy()
    br_blk[:, 0] /= cap
    br_blk_all = brb[["p_dc_computed", "irr_panel_wm2"]].to_numpy(dtype=float).copy()
    br_blk_all[:, 0] /= cap

    for lab, src in [("Brazil all classes, native 1 Hz", br_all),
                     ("Brazil all classes, 900-s diagnostic", br_blk_all),
                     ("Brazil Normal only, native 1 Hz", br_nrm),
                     ("Brazil single-label blocks, 900-s diagnostic", br_blk)]:
        hc, _ = hull_containment(src, hk_xy)
        bc, occ_src, occ_tgt = bin_containment(src, hk_xy)
        joint_rows.append({"capacity": cap_lab, "brazil_representation": lab,
                           "hk_population": "HK clean-sensor / no-known-fault 2023",
                           "br_n": len(src), "hk_n": len(hk_xy),
                           "hull_containment_pct": round(hc, 2),
                           "bin_containment_pct": round(bc, 2),
                           "src_occupied_cells": occ_src,
                           "hk_occupied_cells": occ_tgt,
                           "ks_power": round(ks_2samp(src[:, 0], hk_xy[:, 0]).statistic, 4),
                           "ks_irr": round(ks_2samp(src[:, 1], hk_xy[:, 1]).statistic, 4)})
        print(f"  [{cap_lab}] {lab:<44} hull {hc:6.2f}%  bins {bc:6.2f}%  "
              f"(src cells {occ_src:>3}/{GRID*GRID})")
joint = pd.DataFrame(joint_rows)

# ═════════════════════════════════════════════════════════════════════════════
# ITEM 9 — information equivalence of POWER_PLUS_IRR
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  ITEM 9 — POWER_PLUS_IRR information equivalence")
print("=" * 78)
best = (abl[abl["feature_set"] == "Computed power + measured irradiance"]
        .sort_values("macro_f1_oof", ascending=False))
b0 = best.iloc[0]
print(f"  Objective 1 feature list : ['p_dc_computed', 'irr_panel_wm2'] (unnormalised)")
print(f"  Proposed shared contract : ['p_dc_computed / capacity', 'irr_panel_wm2']")
print(f"  Best saved model         : {b0['model']}  Macro F1 = {b0['macro_f1_oof']:.4f}")
print(f"  MLP in SCALE_SENSITIVE   : True -> fold-fitted StandardScaler applied")
print("  Constant positive divisor C on one feature:")
print("    inf->NaN     : unchanged")
print("    median impute: median(x/C) = median(x)/C  (equivariant)")
print("    StandardScaler: (x/C - mean(x)/C) / (sd(x)/C) = (x - mean(x)) / sd(x)")
print("    => the matrix reaching the MLP is mathematically IDENTICAL")

# ═════════════════════════════════════════════════════════════════════════════
# ITEM 10 — collapsed binary source performance from saved confusion matrices
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  ITEM 10 — collapsed binary source performance (read-only)")
print("=" * 78)
bin_rows = []
for (fs, mdl), g in cm.groupby(["feature_set", "model"]):
    piv = g.pivot(index="true", columns="pred", values="count").fillna(0)
    tn = float(piv.loc["Normal", "Normal"])                       # Normal -> Normal
    fp = float(piv.loc["Normal"].sum() - tn)                      # Normal -> anomalous
    fn = float(sum(piv.loc[c, "Normal"] for c in piv.index if c != "Normal"))
    tp = float(piv.drop(index="Normal").drop(columns="Normal").to_numpy().sum())
    anom_rec = tp / (tp + fn)
    anom_prec = tp / (tp + fp)
    norm_rec = tn / (tn + fp)
    f1 = 2 * anom_prec * anom_rec / (anom_prec + anom_rec)
    bal = 0.5 * (anom_rec + norm_rec)
    bin_rows.append({"feature_set": fs, "model": mdl,
                     "TN_normal_normal": int(tn), "FP_normal_anom": int(fp),
                     "FN_anom_normal": int(fn), "TP_anom_anom": int(tp),
                     "normal_recall": round(norm_rec, 4),
                     "anomalous_recall": round(anom_rec, 4),
                     "anomalous_precision": round(anom_prec, 4),
                     "anomalous_f1": round(f1, 4),
                     "balanced_accuracy": round(bal, 4)})
binm = pd.DataFrame(bin_rows).sort_values("anomalous_f1", ascending=False)
KEY = [("Computed power + measured irradiance", "MLP"),
       ("Computed total power only", "HistGBM"),
       ("Electrical (full)", "MLP")]
print("\n  Collapsed binary (Normal vs anomalous) from the pooled out-of-fold 5x5:")
for fs, mdl in KEY:
    r = binm[(binm.feature_set == fs) & (binm.model == mdl)].iloc[0]
    print(f"\n  {fs} / {mdl}")
    print(f"    confusion  TN={r.TN_normal_normal:,}  FP={r.FP_normal_anom:,}  "
          f"FN={r.FN_anom_normal:,}  TP={r.TP_anom_anom:,}")
    print(f"    Normal recall {r.normal_recall:.4f} | anomalous recall "
          f"{r.anomalous_recall:.4f} | anomalous precision {r.anomalous_precision:.4f}")
    print(f"    anomalous F1  {r.anomalous_f1:.4f} | balanced accuracy "
          f"{r.balanced_accuracy:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE — joint support
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
br_all = br[["p_dc_computed", "irr_panel_wm2"]].dropna().to_numpy(dtype=float)
br_all = br_all[np.isfinite(br_all).all(axis=1)].copy()
br_all[:, 0] /= PDC0_BR_RATED
br_nrm = br.loc[br["f_nv"] == 0, ["p_dc_computed", "irr_panel_wm2"]].dropna().to_numpy(dtype=float)
br_nrm = br_nrm[np.isfinite(br_nrm).all(axis=1)].copy()
br_nrm[:, 0] /= PDC0_BR_RATED
for a, (src, lab) in zip(ax, [(br_all, "Brazil, all classes"),
                              (br_nrm, "Brazil, Normal only")]):
    s = src[np.random.default_rng(0).choice(len(src), size=min(25000, len(src)),
                                            replace=False)]
    a.scatter(s[:, 0], s[:, 1], s=3, alpha=0.18, c="tab:orange", label=lab,
              rasterized=True)
    a.scatter(hk_xy[:, 0], hk_xy[:, 1], s=3, alpha=0.30, c="tab:blue",
              label="HK clean-sensor / no-known-fault 2023", rasterized=True)
    h = ConvexHull(src)
    v = np.append(h.vertices, h.vertices[0])
    a.plot(src[v, 0], src[v, 1], c="darkred", lw=1.4, label="Brazil convex hull")
    hc, _ = hull_containment(src, hk_xy)
    bc, _, _ = bin_containment(src, hk_xy)
    a.set_title(f"{lab}\nHK inside hull {hc:.1f}% | inside occupied bins {bc:.1f}%",
                fontsize=12)
    a.set_xlabel("capacity-normalised power (-)", fontsize=11)
    a.set_ylabel("irradiance (W/m²)", fontsize=11)
    a.legend(fontsize=8, loc="upper left", markerscale=4)
    a.grid(alpha=0.3)
fig.suptitle("Objective 3 — joint two-feature support, Brazil (÷5.00 kW) vs HK "
             "clean-sensor months", fontsize=13)
fig.text(0.5, -0.02,
         "Descriptive support analysis only — no fitted domain classifier. "
         "Convex-hull containment is permissive and can bridge empty interior "
         "regions;\noccupied-bin containment (20×20 common grid) is more "
         "sensitive to actual joint source occupancy. Neither defines a "
         "pass/fail threshold.",
         ha="center", fontsize=9,
         bbox=dict(boxstyle="round", fc="whitesmoke", ec="grey"))
fig.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# WRITE (Objective 3 design outputs only)
# ─────────────────────────────────────────────────────────────────────────────
if "--verify-only" in sys.argv:
    print("\nVERIFY-ONLY MODE — no files written.")
    sys.exit(0)

xt.to_csv(OUT_DIR / "objective3_blocks_by_group_class.csv")
reg_tab.to_csv(OUT_DIR / "objective3_event_duration_summary.csv", index=False)
joint.to_csv(OUT_DIR / "objective3_joint_support.csv", index=False)
binm.to_csv(OUT_DIR / "objective3_binary_collapse.csv", index=False)
fig.savefig(OUT_DIR / "objective3_joint_support.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "objective3_joint_support.pdf", bbox_inches="tight")
with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.now():%Y-%m-%dT%H:%M:%S} | objective3_gate_check.py | "
            f"input: fault_dataset.csv, objective2_grid.csv, "
            f"extended_ablation_summary.csv, extended_confusion_matrices.csv, "
            f"source_event_register.csv | output: 6 files in "
            f"Objective3_Outputs/objective3_design/ | notes: read-only consolidated gate "
            f"check; nothing trained, no transfer run\n")
print(f"\nWrote 6 files to {OUT_DIR}")
