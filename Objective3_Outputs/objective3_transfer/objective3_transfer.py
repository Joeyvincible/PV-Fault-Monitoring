"""Objective 3 — Minimal direct-transfer implementation.

Implementation of a FROZEN design. No design question is reopened, no
hyperparameter is chosen, no threshold is tuned, and no target-domain
adaptation of any kind is performed.

Direct transfer only: Brazil-fitted preprocessing is applied unchanged to HK,
with HK power converted into Brazil-capacity-equivalent watts. Every fitting
call in this file is marked `# BRAZIL-FIT` and a self-scan asserts that no
unmarked fitting call exists anywhere, so no HK-fitted transformation can be
introduced silently.
"""
import sys
import re
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.spatial import ConvexHull

from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import f1_score, confusion_matrix
import joblib

# ═════════════════════════════════════════════════════════════════════════════
# GUARDS
# ═════════════════════════════════════════════════════════════════════════════
BANNED = ["hk accuracy", "hk f1", "hk recall", "hk precision",
          "validated on hk", "hk classification accuracy", "confirmed fault",
          "p(anomalous)"]


def check(text, where="text"):
    low = text.lower()
    for b in BANNED:
        assert b not in low, f"BANNED PHRASE {b!r} in {where}"
    return text


FORBIDDEN_INPUTS = ["fault_dataset_cleaned.csv", "outputs/no_sensor",
                    "outputs/lstm_with_sensor", "pv_cross_dataset.py", ".pt"]
_read = []


def rd(p, **kw):
    s = str(p)
    for f in FORBIDDEN_INPUTS:
        assert f not in s, f"FORBIDDEN INPUT: {s}"
    _read.append(s)
    return pd.read_csv(p, **kw)


def assert_fits_are_brazil_only():
    """Every fitting call must be marked # BRAZIL-FIT and occur in Step 1."""
    src = Path(__file__).read_text().splitlines()
    pat = re.compile(r"\." + r"fit\(|\." + r"fit_transform\(")   # SCAN-EXEMPT
    hits = [(i + 1, l.strip()) for i, l in enumerate(src)
            if pat.search(l) and "SCAN-EXEMPT" not in l]
    unmarked = [(i, l) for i, l in hits if "BRAZIL-FIT" not in l]
    assert not unmarked, f"UNMARKED FITTING CALL (possible HK fit): {unmarked}"
    print(f"Fit-call self-scan: PASS — {len(hits)} fitting call(s), "
          f"all marked BRAZIL-FIT")
    return hits


FIT_HITS = assert_fits_are_brazil_only()

# HK data may not be loaded until the Step 2 checkpoint has passed.
CHECKPOINT_PASSED = False

# ═════════════════════════════════════════════════════════════════════════════
# PATHS AND CONFIGURATION CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
HERE = Path(__file__).resolve().parent
NAVE = HERE.parent.parent
OUT = HERE
BRAZIL_CSV = NAVE / "02_Fault_Classification" / "data" / "fault_dataset.csv"
HK_CSV = NAVE / "01_HK_Detection" / "outputs" / "objective2_modeling" / "objective2_grid.csv"
ABL_DIR = NAVE / "02_Fault_Classification" / "outputs" / "extended_ablation"
CONFMAT = ABL_DIR / "extended_confusion_matrices.csv"
ABLATION = ABL_DIR / "extended_ablation_summary.csv"
BINCOLLAPSE = NAVE / "Objective3_Outputs" / "objective3_design" / "objective3_binary_collapse.csv"
ARM_B = NAVE / "01_HK_Detection" / "outputs" / "objective2_modeling" / "residuals_full_arm_b.csv"
ARM_C = NAVE / "01_HK_Detection" / "outputs" / "objective2_modeling" / "residuals_full_arm_c.csv"
RUN_LOG = OUT / "run_log.txt"

GAP_THRESHOLD = 3600
RANDOM_SEED = 42
LABELS = [0, 1, 2, 3, 4]                       # canonical class order
LABEL_NAMES = {0: "Normal", 1: "Short-Circuit", 2: "Degradation",
               3: "Open-Circuit", 4: "Shadowing"}
CANON = [LABEL_NAMES[l] for l in LABELS]
APPROVED_FOLDS = {1: [2, 10, 11], 2: [1, 3, 7, 12], 3: [0, 9, 13],
                  4: [5, 6, 14], 5: [4, 8, 15]}
FEATS = ["p_dc_computed", "irr_panel_wm2"]     # POWER_PLUS_IRR feature set
CAPACITY_BR = 5000.0
CAPACITY_BR_SENS = 5280.0
CAPACITY_HK = 27606.0
CLEAN_MONTHS = [1, 2, 11, 12]
TZ = "Asia/Hong_Kong"
FROZEN_FS = "Computed power + measured irradiance"
FROZEN_MODEL = "MLP"
GRID = 20

print("=" * 78)
print("  OBJECTIVE 3 — MINIMAL DIRECT-TRANSFER IMPLEMENTATION")
print("=" * 78)

VERIFY_ONLY = "--verify-only" in sys.argv

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — fit the five group-disjoint fold pipelines
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  STEP 1 — recreate the five frozen fold pipelines")
print("=" * 78)

df = rd(BRAZIL_CSV).sort_values("sample_index").reset_index(drop=True)
y = df["f_nv"].to_numpy()
n = len(df)
df["recording_group"] = ((df["sample_index"].diff() > GAP_THRESHOLD)
                         .fillna(False).cumsum().astype(int))
assert sorted(df["recording_group"].unique()) == list(range(16)), \
    "recording-group reconstruction did not yield 16 groups"
print(f"  Brazil rows {n:,} | recording groups 16: PASS")

# Reconstruct and verify the approved fold allocation.
sizes = df.groupby("recording_group").size()
anchors = [11, 12, 13, 14, 15]
folds = {i + 1: [a] for i, a in enumerate(anchors)}
totals = {i + 1: int(sizes[a]) for i, a in enumerate(anchors)}
for g in sorted([x for x in range(16) if x not in anchors],
                key=lambda x: (-int(sizes[x]), x)):
    for f in sorted(folds, key=lambda f: (totals[f], f)):
        if (g == 8 and 9 in folds[f]) or (g == 9 and 8 in folds[f]):
            continue
        folds[f].append(g)
        totals[f] += int(sizes[g])
        break
folds = {f: sorted(v) for f, v in folds.items()}
assert folds == APPROVED_FOLDS, f"Fold allocation mismatch: {folds}"
df["fold"] = 0
for f, gs in folds.items():
    df.loc[df["recording_group"].isin(gs), "fold"] = f
assert (df["fold"] > 0).all()
print(f"  Fold allocation reproduces APPROVED_FOLDS: PASS")


def downsample_to_second_largest(idx_tr, y_tr):
    """Frozen rule: classes larger than the SECOND-largest reduced to it."""
    counts = {int(c): int((y_tr == c).sum()) for c in np.unique(y_tr)}
    target = sorted(counts.values(), reverse=True)[1]
    keep = []
    for c in sorted(counts):
        ci = idx_tr[y_tr == c]
        keep.append(ci if len(ci) <= target
                    else resample(ci, n_samples=target, replace=False,
                                  random_state=RANDOM_SEED))
    return np.sort(np.concatenate(keep)), target


PIPELINES = {}
oof = np.full(n, -1, dtype=np.int8)
fold_rows = []

if VERIFY_ONLY:
    print("\n  VERIFY-ONLY: fold pipelines not fitted, HK not touched.")
else:
    for f in sorted(folds):
        te = (df["fold"] == f).to_numpy()
        idx_tr = np.where(~te)[0]
        idx_fit, target = downsample_to_second_largest(idx_tr, y[idx_tr])
        fp = hashlib.sha1(idx_fit.tobytes()).hexdigest()

        X_fit = df.loc[idx_fit, FEATS].to_numpy(dtype=float)
        X_te = df.loc[te, FEATS].to_numpy(dtype=float)
        X_fit[~np.isfinite(X_fit)] = np.nan
        X_te[~np.isfinite(X_te)] = np.nan

        imp = SimpleImputer(strategy="median").fit(X_fit)          # BRAZIL-FIT
        X_fit, X_te = imp.transform(X_fit), imp.transform(X_te)
        sc = StandardScaler().fit(X_fit)                           # BRAZIL-FIT
        X_fit, X_te = sc.transform(X_fit), sc.transform(X_te)

        m = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                          early_stopping=True, n_iter_no_change=10,
                          random_state=RANDOM_SEED).fit(X_fit, y[idx_fit])  # BRAZIL-FIT
        assert list(m.classes_) == LABELS, \
            f"fold {f} classes_ {list(m.classes_)} != canonical {LABELS}"

        pred = m.predict(X_te)
        oof[te] = pred
        ff = f1_score(y[te], pred, average="macro", labels=LABELS,
                      zero_division=0)
        PIPELINES[f] = {"imputer": imp, "scaler": sc, "model": m,
                        "train_groups": folds[f], "fit_rows": int(len(idx_fit)),
                        "downsample_target": int(target),
                        "downsample_sha1": fp,
                        "train_raw_watt_mean": float(sc.mean_[0]),
                        "train_raw_watt_scale": float(sc.scale_[0]),
                        "idx_fit": idx_fit}
        fold_rows.append({"fold": f, "train_groups": str(folds[f]),
                          "test_rows": int(te.sum()), "fit_rows": len(idx_fit),
                          "downsample_target": target,
                          "held_out_macro_f1": round(float(ff), 4),
                          "downsample_sha1": fp})
        print(f"  fold {f}: train groups {folds[f]} | fit rows {len(idx_fit):,} "
              f"| held-out Macro F1 {ff:.4f}")
    assert (oof >= 0).all(), "row without an out-of-fold prediction"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — BLOCKING correctness checkpoint (count level)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  STEP 2 — correctness checkpoint (BLOCKING, count level)")
print("=" * 78)

cm_frozen_long = rd(CONFMAT)
cf = cm_frozen_long[(cm_frozen_long.feature_set == FROZEN_FS)
                    & (cm_frozen_long.model == FROZEN_MODEL)]
assert len(cf) == 25, f"frozen 5x5 not found (got {len(cf)} cells)"
FROZEN_CM = (cf.pivot(index="true", columns="pred", values="count")
             .reindex(index=CANON, columns=CANON).to_numpy().astype(np.int64))
binc = rd(BINCOLLAPSE)
bz = binc[(binc.feature_set == FROZEN_FS) & (binc.model == FROZEN_MODEL)].iloc[0]
FROZEN_2X2 = np.array([[int(bz.TN_normal_normal), int(bz.FP_normal_anom)],
                       [int(bz.FN_anom_normal), int(bz.TP_anom_anom)]],
                      dtype=np.int64)
abl = rd(ABLATION)
FROZEN_MACRO = float(abl[(abl.feature_set == FROZEN_FS)
                         & (abl.model == FROZEN_MODEL)].iloc[0].macro_f1_oof)


def metrics_from_counts(cm5):
    """Macro F1 from the 5x5; anomalous F1 and balanced accuracy from its
    Normal / non-Normal collapse. All derived from counts only."""
    f1s = []
    for i in range(5):
        tp = cm5[i, i]
        fp = cm5[:, i].sum() - tp
        fn = cm5[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro = float(np.mean(f1s))
    tn = int(cm5[0, 0])
    fp = int(cm5[0, 1:].sum())
    fn = int(cm5[1:, 0].sum())
    tp = int(cm5[1:, 1:].sum())
    a_rec = tp / (tp + fn)
    a_prec = tp / (tp + fp)
    n_rec = tn / (tn + fp)
    a_f1 = 2 * a_prec * a_rec / (a_prec + a_rec)
    bal = 0.5 * (a_rec + n_rec)
    return macro, a_f1, bal, np.array([[tn, fp], [fn, tp]], dtype=np.int64)


chk_rows = []
if not VERIFY_ONLY:
    REPRO_CM = confusion_matrix(y, oof, labels=LABELS).astype(np.int64)
    r_macro, r_af1, r_bal, REPRO_2X2 = metrics_from_counts(REPRO_CM)
    f_macro, f_af1, f_bal, F2 = metrics_from_counts(FROZEN_CM)

    cm_equal = bool(np.array_equal(REPRO_CM, FROZEN_CM))
    b2_equal = bool(np.array_equal(REPRO_2X2, FROZEN_2X2))
    cm_absdiff = int(np.abs(REPRO_CM - FROZEN_CM).sum())

    print(f"\n  Pooled 5x5 OOF counts identical to frozen : "
          f"{'PASS' if cm_equal else 'FAIL'}  (total abs cell diff {cm_absdiff:,})")
    print(f"  Collapsed 2x2 counts identical to frozen  : "
          f"{'PASS' if b2_equal else 'FAIL'}")
    print(f"  Macro F1 recomputed from counts           : {r_macro:.4f} "
          f"(frozen {FROZEN_MACRO:.4f})")
    print(f"  Anomalous F1 recomputed from counts       : {r_af1:.4f} "
          f"(frozen {float(bz.anomalous_f1):.4f})")
    print(f"  Balanced accuracy recomputed from counts  : {r_bal:.4f} "
          f"(frozen {float(bz.balanced_accuracy):.4f})")

    for nm, rep, fro in [("macro_f1", r_macro, FROZEN_MACRO),
                         ("anomalous_f1", r_af1, float(bz.anomalous_f1)),
                         ("balanced_accuracy", r_bal, float(bz.balanced_accuracy))]:
        chk_rows.append({"quantity": nm, "reproduced": round(rep, 6),
                         "frozen": fro, "abs_diff": round(abs(rep - fro), 6),
                         "match_4dp": bool(round(rep, 4) == round(fro, 4))})
    for i, tl in enumerate(CANON):
        for j, pl in enumerate(CANON):
            chk_rows.append({"quantity": f"cm[{tl}->{pl}]",
                             "reproduced": int(REPRO_CM[i, j]),
                             "frozen": int(FROZEN_CM[i, j]),
                             "abs_diff": int(abs(REPRO_CM[i, j] - FROZEN_CM[i, j])),
                             "match_4dp": bool(REPRO_CM[i, j] == FROZEN_CM[i, j])})
    for a, lab in [((0, 0), "TN"), ((0, 1), "FP"), ((1, 0), "FN"), ((1, 1), "TP")]:
        chk_rows.append({"quantity": f"binary_{lab}",
                         "reproduced": int(REPRO_2X2[a]),
                         "frozen": int(FROZEN_2X2[a]),
                         "abs_diff": int(abs(REPRO_2X2[a] - FROZEN_2X2[a])),
                         "match_4dp": bool(REPRO_2X2[a] == FROZEN_2X2[a])})

    metrics_ok = (round(r_macro, 4) == round(FROZEN_MACRO, 4)
                  and round(r_af1, 4) == round(float(bz.anomalous_f1), 4)
                  and round(r_bal, 4) == round(float(bz.balanced_accuracy), 4))
    CHECKPOINT_PASSED = bool(cm_equal and b2_equal and metrics_ok)
    print(f"\n  STEP 2 CHECKPOINT: {'PASS' if CHECKPOINT_PASSED else 'FAIL'}")
    pd.DataFrame(chk_rows).to_csv(OUT / "objective3_source_checkpoint.csv", index=False)

    if not CHECKPOINT_PASSED:
        print("\n  BLOCKING FAILURE — the refit does not reproduce the frozen")
        print("  configuration at count level. HK data has NOT been accessed.")
        print("  Reporting and stopping, as required.")
        with open(RUN_LOG, "a") as fh:
            fh.write(f"{datetime.now():%Y-%m-%dT%H:%M:%S} | objective3_transfer.py | "
                     f"STEP 2 CHECKPOINT FAILED — HK not accessed; "
                     f"5x5 identical={cm_equal}, 2x2 identical={b2_equal}, "
                     f"abs cell diff={cm_absdiff}\n")
        sys.exit(1)

    # Persist the five pipelines ONLY after the blocking checkpoint passed.
    for f, P in PIPELINES.items():
        joblib.dump({"imputer": P["imputer"], "scaler": P["scaler"],
                     "model": P["model"],
                     "metadata": {"fold": f, "train_groups": P["train_groups"],
                                  "canonical_class_order": LABELS,
                                  "canonical_class_names": CANON,
                                  "features": FEATS,
                                  "configuration": f"{FROZEN_FS} / {FROZEN_MODEL}",
                                  "random_seed": RANDOM_SEED,
                                  "downsample_target": P["downsample_target"],
                                  "downsample_sha1": P["downsample_sha1"],
                                  "capacity_br_W": CAPACITY_BR,
                                  "capacity_hk_W": CAPACITY_HK,
                                  "checkpoint": "step2 count-level PASS"}},
                    OUT / f"fold_{f}_pipeline.joblib")
    print(f"  Persisted 5 fold pipelines (post-checkpoint).")

if VERIFY_ONLY:
    print("\n" + "=" * 78)
    print("  VERIFY-ONLY MODE — no fitting, no HK access, no files written.")
    print("=" * 78)
    print("  Would write, all inside Objective3_Outputs/objective3_transfer/:")
    for fn in ["objective3_source_checkpoint.csv", "fold_1..5_pipeline.joblib",
               "objective3_hk_screening.csv", "objective3_per_fold_screening.csv",
               "objective3_screening_distribution.csv", "objective3_temporal_clustering.csv",
               "objective3_objective2_agreement.csv", "objective3_screening_checks.csv",
               "objective3_stress_tests.csv", "objective3_capacity_sensitivity.csv",
               "objective3_transfer_report.md", "objective3_transfer_figures.png/.pdf",
               "run_log.txt"]:
        print(f"    {fn}")
    sys.exit(0)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — HK preparation (only reachable after the checkpoint passed)
# ═════════════════════════════════════════════════════════════════════════════
assert CHECKPOINT_PASSED, "HK access attempted before the Step 2 checkpoint"
print("\n" + "=" * 78)
print("  STEP 3 — HK preparation")
print("=" * 78)

hk = rd(HK_CSV, index_col=0)
hk.index = pd.to_datetime(hk.index, utc=True).tz_convert(TZ)

ELIG = (hk["ghi_clear"] >= 50) & (hk["zenith"] < 85)   # Objective 2 geometry rule


def build_population(name, month_mask, label):
    """Common rule: geometry eligibility, then both features present."""
    sel = ELIG & month_mask
    raw = hk[sel]
    pm = raw["power_W"].isna()
    im = raw["irr_meas"].isna()
    complete = raw[~pm & ~im]
    return {"population": name, "label": label,
            "raw_eligible": int(len(raw)),
            "feature_complete": int(len(complete)),
            "excluded_missing_inputs": int(len(raw) - len(complete)),
            # verified separately per feature - never inferred from the total
            "missing_power_W": int(pm.sum()),
            "missing_irr_meas": int(im.sum()),
            "missing_both": int((pm & im).sum()),
            "data": complete}


yr = hk.index.year
mo = hk.index.month
POPS = [
    build_population("clean-sensor 2023 (Jan/Feb/Nov/Dec)",
                     (yr == 2023) & mo.isin(CLEAN_MONTHS), "PRIMARY"),
    build_population("March-October 2023",
                     (yr == 2023) & mo.isin(range(3, 11)),
                     "corrupted-input stress test"),
    build_population("full 2023", (yr == 2023),
                     "mixed-period stress summary (NOT a corruption contrast)"),
    build_population("Saola proxy window (1-15 Sept 2023)",
                     (yr == 2023) & (mo == 9) & (hk.index.day <= 15),
                     "corrupted-input stress test"),
]
print(f"  {'population':<40}{'raw elig':>10}{'complete':>10}{'excluded':>10}")
for p in POPS:
    print(f"  {p['population']:<40}{p['raw_eligible']:>10,}"
          f"{p['feature_complete']:>10,}{p['excluded_missing_inputs']:>10,}")

# ── target-side transformation identity check, per fold ─────────────────────
print("\n  Target-side transformation identity check (per fold):")
prim = POPS[0]["data"]
p_hk_raw = prim["power_W"].to_numpy(dtype=float)
id_rows = []
for f, P in PIPELINES.items():
    mu, sd = P["train_raw_watt_mean"], P["train_raw_watt_scale"]
    lhs = (p_hk_raw / CAPACITY_HK - mu / CAPACITY_BR) / (sd / CAPACITY_BR)
    rhs = (p_hk_raw * (CAPACITY_BR / CAPACITY_HK) - mu) / sd
    d = float(np.abs(lhs - rhs).max())
    id_rows.append({"fold": f, "max_abs_diff": d})
    print(f"    fold {f}: max |lhs - rhs| = {d:.3e}")
MAX_ID_DIFF = max(r["max_abs_diff"] for r in id_rows)
assert MAX_ID_DIFF < 1e-6, f"identity violated: {MAX_ID_DIFF}"


def to_brazil_equiv_watts(power_W, cap_br=CAPACITY_BR):
    """HK watts -> Brazil-capacity-equivalent watts. No HK-fitted statistic."""
    return power_W * (cap_br / CAPACITY_HK)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — apply the five-fold ensemble
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  STEP 4 — five-fold ensemble application (pre-specified deployment")
print("           summary; NOT an out-of-fold validated object)")
print("=" * 78)


def apply_ensemble(data, cap_br=CAPACITY_BR):
    """Each fold's own imputer -> its own scaler -> its own model.
    Probabilities reindexed into canonical order before averaging."""
    X0 = np.column_stack([to_brazil_equiv_watts(
        data["power_W"].to_numpy(dtype=float), cap_br),
        data["irr_meas"].to_numpy(dtype=float)])
    X0[~np.isfinite(X0)] = np.nan
    per_fold_proba, per_fold_pred = {}, {}
    for f, P in PIPELINES.items():
        Xi = P["scaler"].transform(P["imputer"].transform(X0))
        pr = P["model"].predict_proba(Xi)
        cls = list(P["model"].classes_)
        assert cls == LABELS, f"fold {f} classes_ drifted: {cls}"
        order = [cls.index(l) for l in LABELS]
        assert order == list(range(5)), "class reindex order unexpected"
        pr = pr[:, order]                      # explicit canonical reindex
        assert pr.shape[1] == 5
        per_fold_proba[f] = pr
        per_fold_pred[f] = np.array(LABELS)[pr.argmax(axis=1)]
    stack = np.stack([per_fold_proba[f] for f in sorted(PIPELINES)], axis=0)
    mean_p = stack.mean(axis=0)
    argmax = np.array(LABELS)[mean_p.argmax(axis=1)]
    binary = (argmax != 0).astype(int)
    score = 1.0 - mean_p[:, 0]                 # UNCALIBRATED ranking score
    return {"mean_proba": mean_p, "pred5": argmax, "binary": binary,
            "score": score, "per_fold_pred": per_fold_pred,
            "per_fold_proba": per_fold_proba}


RES = {}
for p in POPS:
    RES[p["population"]] = apply_ensemble(p["data"])
prim_res = RES[POPS[0]["population"]]
print(f"  Primary population scored: {len(prim_res['binary']):,} rows")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — reporting on the primary clean-sensor population
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  STEP 5 — reporting (clean-sensor primary)")
print("=" * 78)

# Two Brazil reference distributions, neither an expected HK prior.
br_empirical = pd.Series(y).value_counts().reindex(LABELS).fillna(0).astype(int)
bal_counts = np.zeros(5, dtype=int)
for f, P in PIPELINES.items():
    yy = y[P["idx_fit"]]
    for l in LABELS:
        bal_counts[l] += int((yy == l).sum())

dist_rows = []
for p in POPS:
    r = RES[p["population"]]
    vc = pd.Series(r["pred5"]).value_counts().reindex(LABELS).fillna(0).astype(int)
    for l in LABELS:
        dist_rows.append({
            "population": p["population"], "label_role": p["label"],
            "denominator_feature_complete_rows": p["feature_complete"],
            "class": LABEL_NAMES[l], "predicted_count": int(vc[l]),
            "predicted_proportion": round(float(vc[l]) / max(len(r["pred5"]), 1), 6),
            "brazil_empirical_label_proportion": round(float(br_empirical[l]) / n, 6),
            "brazil_balanced_training_proportion": round(
                float(bal_counts[l]) / bal_counts.sum(), 6)})
dist = pd.DataFrame(dist_rows)
print("\n  Five-class ensemble prediction distribution — clean-sensor primary:")
for l in LABELS:
    row = dist[(dist.population == POPS[0]["population"])
               & (dist["class"] == LABEL_NAMES[l])].iloc[0]
    print(f"    {LABEL_NAMES[l]:<14} {row.predicted_count:>7,}  "
          f"{row.predicted_proportion*100:>6.2f}%   "
          f"(Brazil empirical {row.brazil_empirical_label_proportion*100:>5.2f}%, "
          f"balanced train {row.brazil_balanced_training_proportion*100:>5.2f}%)")

SCREEN_RATE = float(prim_res["binary"].mean())
print(f"\n  Binary screening rate (classifier-screened non-Normal): "
      f"{SCREEN_RATE*100:.2f}% of {POPS[0]['feature_complete']:,} "
      f"feature-complete rows")

q = np.percentile(prim_res["score"], [0, 5, 25, 50, 75, 95, 100])
print(f"  ensemble_anomalous_score (UNCALIBRATED, ranking only) quantiles:")
print(f"    min {q[0]:.4f} | p5 {q[1]:.4f} | p25 {q[2]:.4f} | median {q[3]:.4f}"
      f" | p75 {q[4]:.4f} | p95 {q[5]:.4f} | max {q[6]:.4f}")

# ── per-fold HK behaviour and fold-to-fold agreement ────────────────────────
pf_rows = []
for f in sorted(PIPELINES):
    pr = prim_res["per_fold_pred"][f]
    vc = pd.Series(pr).value_counts().reindex(LABELS).fillna(0).astype(int)
    pf_rows.append({"fold": f, "screening_rate": round(float((pr != 0).mean()), 6),
                    "denominator_rows": len(pr),
                    **{f"pred_{LABEL_NAMES[l]}": int(vc[l]) for l in LABELS}})
pf = pd.DataFrame(pf_rows)
print("\n  Per-fold HK screening rate (denominator "
      f"{POPS[0]['feature_complete']:,} rows):")
for r in pf.itertuples():
    print(f"    fold {r.fold}: {r.screening_rate*100:6.2f}%")

stack_pred = np.stack([prim_res["per_fold_pred"][f] for f in sorted(PIPELINES)])
unanimous5 = float((stack_pred == stack_pred[0]).all(axis=0).mean())
bin_stack = (stack_pred != 0).astype(int)
unanimous_bin = float(((bin_stack.sum(axis=0) == 0)
                       | (bin_stack.sum(axis=0) == 5)).mean())
pair_rows = []
fl = sorted(PIPELINES)
for i in range(len(fl)):
    for j in range(i + 1, len(fl)):
        pair_rows.append({"fold_a": fl[i], "fold_b": fl[j],
                          "five_class_agreement": round(float(
                              (prim_res["per_fold_pred"][fl[i]]
                               == prim_res["per_fold_pred"][fl[j]]).mean()), 6),
                          "binary_agreement": round(float(
                              ((prim_res["per_fold_pred"][fl[i]] != 0)
                               == (prim_res["per_fold_pred"][fl[j]] != 0)).mean()), 6)})
pairs = pd.DataFrame(pair_rows)
print(f"  Unanimous five-class across all 5 folds: {unanimous5*100:.2f}%")
print(f"  Unanimous binary across all 5 folds    : {unanimous_bin*100:.2f}%")
print(f"  Pairwise binary agreement range        : "
      f"{pairs.binary_agreement.min()*100:.2f}% - {pairs.binary_agreement.max()*100:.2f}%")

# ── temporal clustering by TIMESTAMP continuity ─────────────────────────────
ts = prim["power_W"].index
ordr = np.argsort(ts.values)
ts_sorted = ts[ordr]
flag_sorted = prim_res["binary"][ordr]
STEP = pd.Timedelta(minutes=15)
runs = []
cur = 0
for i in range(len(ts_sorted)):
    if flag_sorted[i] == 1:
        contiguous = (i > 0 and flag_sorted[i - 1] == 1
                      and (ts_sorted[i] - ts_sorted[i - 1]) == STEP)
        cur = cur + 1 if contiguous else 1
    else:
        if cur:
            runs.append(cur)
        cur = 0
if cur:
    runs.append(cur)
runs = np.array(runs) if runs else np.array([], dtype=int)
tc_rows = [{"statistic": "n_anomalous_runs", "value": int(len(runs))},
           {"statistic": "n_flagged_rows", "value": int(flag_sorted.sum())},
           {"statistic": "denominator_feature_complete_rows",
            "value": int(len(flag_sorted))},
           {"statistic": "mean_run_length_intervals",
            "value": round(float(runs.mean()), 3) if len(runs) else 0.0},
           {"statistic": "median_run_length_intervals",
            "value": float(np.median(runs)) if len(runs) else 0.0},
           {"statistic": "max_run_length_intervals",
            "value": int(runs.max()) if len(runs) else 0},
           {"statistic": "runs_of_length_1", "value": int((runs == 1).sum())},
           {"statistic": "runs_ge_4_intervals_ge_1h",
            "value": int((runs >= 4).sum())}]
tc = pd.DataFrame(tc_rows)
print(f"\n  Temporal clustering (runs defined by exact 15-min timestamp "
      f"continuity):")
print(f"    {len(runs):,} runs | mean {runs.mean() if len(runs) else 0:.2f} "
      f"intervals | max {runs.max() if len(runs) else 0} | "
      f"singletons {(runs == 1).sum():,}")

# ── agreement with Objective 2 Arm B / Arm C flags ──────────────────────────
print("\n  Agreement with Objective 2 sensor-free Arm B / Arm C (frozen flags):")
ag_rows = []
try:
    ab = rd(ARM_B, index_col=0)
    ac = rd(ARM_C, index_col=0)
    ab.index = pd.to_datetime(ab.index)
    ac.index = pd.to_datetime(ac.index)
    o3 = pd.DataFrame({"o3_flag": prim_res["binary"]}, index=prim.index)
    for arm_name, arm in [("Arm B", ab), ("Arm C", ac)]:
        j = o3.join(arm[["alert"]], how="inner")
        if len(j) == 0:
            ag_rows.append({"arm": arm_name, "status": "no overlapping timestamps"})
            continue
        both = int(((j.o3_flag == 1) & (j.alert == 1)).sum())
        o3_only = int(((j.o3_flag == 1) & (j.alert == 0)).sum())
        o2_only = int(((j.o3_flag == 0) & (j.alert == 1)).sum())
        neither = int(((j.o3_flag == 0) & (j.alert == 0)).sum())
        agree = (both + neither) / len(j)
        ag_rows.append({"arm": arm_name, "status": "available",
                        "overlapping_rows_denominator": int(len(j)),
                        "both_flagged": both, "objective3_only": o3_only,
                        "objective2_only": o2_only, "neither": neither,
                        "raw_agreement": round(float(agree), 6),
                        "objective3_flag_rate": round(float(j.o3_flag.mean()), 6),
                        "objective2_flag_rate": round(float(j.alert.mean()), 6)})
        print(f"    {arm_name}: overlap {len(j):,} rows | both {both:,} | "
              f"O3-only {o3_only:,} | O2-only {o2_only:,} | "
              f"raw agreement {agree*100:.2f}%")
except FileNotFoundError:
    ag_rows.append({"arm": "Arm B / Arm C", "status": "unavailable"})
    print("    UNAVAILABLE — frozen flags not recoverable")
ag = pd.DataFrame(ag_rows)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Part 5g checks
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  STEP 6 — Part 5g checks")
print("=" * 78)


def hull_containment(src_xy, tgt_xy):
    h = ConvexHull(src_xy)
    A, b = h.equations[:, :-1], h.equations[:, -1]
    return 100.0 * np.all(tgt_xy @ A.T + b <= 1e-12, axis=1).mean()


def bin_containment(src_xy, tgt_xy, grid=GRID):
    both = np.vstack([src_xy, tgt_xy])
    xe = np.linspace(both[:, 0].min(), both[:, 0].max(), grid + 1)
    ye = np.linspace(both[:, 1].min(), both[:, 1].max(), grid + 1)

    def cells(a):
        i = np.clip(np.digitize(a[:, 0], xe) - 1, 0, grid - 1)
        j = np.clip(np.digitize(a[:, 1], ye) - 1, 0, grid - 1)
        return i * grid + j
    occ = set(np.unique(cells(src_xy)).tolist())
    return 100.0 * np.array([c in occ for c in cells(tgt_xy)]).mean()


# HK in capacity-normalised representation, same as the transfer run
hk_xy = np.column_stack([prim["power_W"].to_numpy(dtype=float) / CAPACITY_HK,
                         prim["irr_meas"].to_numpy(dtype=float)])
p5g_rows = []

# Check 1 — source-performance floor
p5g_rows.append({"check": "1. Source-performance floor",
                 "scope": "Brazil, frozen grouped protocol",
                 "result": f"Step 2 count-level reproduction PASS; "
                           f"Macro F1 {r_macro:.4f}, anomalous F1 {r_af1:.4f}, "
                           f"balanced accuracy {r_bal:.4f}",
                 "verdict": "SATISFIED"})

# Check 2 — within-Brazil sanity transfer
held = [fr["held_out_macro_f1"] for fr in fold_rows]
p5g_rows.append({"check": "2. Within-Brazil sanity transfer",
                 "scope": "each fold model on its held-out Brazil groups",
                 "result": "per-fold held-out Macro F1 " +
                           ", ".join(f"{h:.4f}" for h in held) +
                           f" (mean {np.mean(held):.4f}, min {min(held):.4f})",
                 "verdict": "SATISFIED" if min(held) > 0.4 else "WEAK"})

# Check 3 — per-fold input-range extrapolation audit, Gate 2 methods
print("\n  Check 3 — per-fold extrapolation audit (fold's own training data):")
ex_rows = []
for f, P in PIPELINES.items():
    tr = df.loc[P["idx_fit"], FEATS].to_numpy(dtype=float)
    tr = tr[np.isfinite(tr).all(axis=1)]
    tr_xy = np.column_stack([tr[:, 0] / CAPACITY_BR, tr[:, 1]])
    lo_p = float((hk_xy[:, 0] < tr_xy[:, 0].min()).mean() * 100)
    hi_p = float((hk_xy[:, 0] > tr_xy[:, 0].max()).mean() * 100)
    lo_i = float((hk_xy[:, 1] < tr_xy[:, 1].min()).mean() * 100)
    hi_i = float((hk_xy[:, 1] > tr_xy[:, 1].max()).mean() * 100)
    hc = hull_containment(tr_xy, hk_xy)
    bc = bin_containment(tr_xy, hk_xy)
    ex_rows.append({"fold": f, "hk_below_power_range_pct": round(lo_p, 3),
                    "hk_above_power_range_pct": round(hi_p, 3),
                    "hk_below_irr_range_pct": round(lo_i, 3),
                    "hk_above_irr_range_pct": round(hi_i, 3),
                    "hull_containment_pct": round(hc, 2),
                    "bin_containment_pct": round(bc, 2)})
    print(f"    fold {f}: power below {lo_p:5.2f}% above {hi_p:5.2f}% | "
          f"irr below {lo_i:5.2f}% above {hi_i:5.2f}% | "
          f"hull {hc:6.2f}% | bins {bc:6.2f}%")
ex = pd.DataFrame(ex_rows)
p5g_rows.append({"check": "3. Input-range extrapolation (per fold, Gate 2 methods)",
                 "scope": "each fold's own Brazil training data vs HK primary",
                 "result": f"hull containment {ex.hull_containment_pct.min():.2f}-"
                           f"{ex.hull_containment_pct.max():.2f}% "
                           f"(mean {ex.hull_containment_pct.mean():.2f}%); "
                           f"occupied-bin {ex.bin_containment_pct.min():.2f}-"
                           f"{ex.bin_containment_pct.max():.2f}% "
                           f"(mean {ex.bin_containment_pct.mean():.2f}%); "
                           f"max out-of-range on any axis "
                           f"{max(ex[['hk_below_power_range_pct','hk_above_power_range_pct','hk_below_irr_range_pct','hk_above_irr_range_pct']].to_numpy().max(), 0):.2f}%. "
                           f"Most HK rows lie within broad source support, so "
                           f"gross out-of-range extrapolation alone is unlikely "
                           f"to explain the very high screening rate. Local and "
                           f"conditional distribution shift remain possible and "
                           f"are not excluded; approximately "
                           f"{100-ex.bin_containment_pct.max():.0f}-"
                           f"{100-ex.bin_containment_pct.min():.0f}% of HK rows "
                           f"fall outside occupied training cells.",
                 "verdict": "MOSTLY WITHIN BROAD SOURCE SUPPORT "
                            "(local shift not excluded)"})

# Check 4 — degenerate-output check
vc_prim = pd.Series(prim_res["pred5"]).value_counts().reindex(LABELS).fillna(0).astype(int)
dom_l = int(vc_prim.idxmax())
dom_share = float(vc_prim.max()) / len(prim_res["pred5"])
# Reported DESCRIPTIVELY. No collapse threshold is applied: none was
# preregistered, and inventing a cut-off after seeing the values would be a
# post-hoc criterion.
share_txt = ", ".join(f"{LABEL_NAMES[l]} {vc_prim[l]/len(prim_res['pred5'])*100:.2f}%"
                      for l in sorted(LABELS, key=lambda l: -vc_prim[l]))
p5g_rows.append({"check": "4. Degenerate-output check",
                 "scope": "HK clean-sensor primary, ensemble five-class argmax",
                 "result": f"Output highly concentrated: {share_txt} of "
                           f"{len(prim_res['pred5']):,} rows. Per-fold screening "
                           f"behaviour similar throughout "
                           f"({pf.screening_rate.min()*100:.2f}-"
                           f"{pf.screening_rate.max()*100:.2f}%), unanimous binary "
                           f"agreement {unanimous_bin*100:.2f}%, pairwise "
                           f"{pairs.binary_agreement.min()*100:.2f}-"
                           f"{pairs.binary_agreement.max()*100:.2f}%; the "
                           f"concentration is not attributable to a single "
                           f"unstable fold. No collapse threshold applied - none "
                           f"was preregistered.",
                 "verdict": f"DESCRIPTIVE: highly concentrated on "
                            f"{LABEL_NAMES[dom_l]}; not fold-driven"})
print(f"\n  Check 4 — output concentration: {share_txt}")

# Check 5 — corrupted-input contrast (clean-sensor vs Mar-Oct)
mo_pop = POPS[1]
mo_res = RES[mo_pop["population"]]
mo_rate = float(mo_res["binary"].mean())
p5g_rows.append({"check": "5. Corrupted-input contrast",
                 "scope": "clean-sensor primary vs March-October 2023",
                 "result": f"Aggregate screening prevalence nearly unchanged: "
                           f"clean-sensor {SCREEN_RATE*100:.2f}% of "
                           f"{POPS[0]['feature_complete']:,} rows vs March-October "
                           f"{mo_rate*100:.2f}% of {mo_pop['feature_complete']:,} "
                           f"rows. NOT a matched counterfactual: the periods differ "
                           f"in time and operating conditions, and identical "
                           f"aggregate rates may conceal different row-level "
                           f"predictions or class distributions. No conclusion "
                           f"drawn about the effect of irradiance corruption.",
                 "verdict": "COMPUTED (not a matched counterfactual)"})
p5g = pd.DataFrame(p5g_rows)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — stress tests and capacity sensitivity
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("  STEP 7 — stress tests and capacity sensitivity")
print("=" * 78)
st_rows = []
for p in POPS:
    r = RES[p["population"]]
    vc = pd.Series(r["pred5"]).value_counts().reindex(LABELS).fillna(0).astype(int)
    st_rows.append({"population": p["population"], "label": p["label"],
                    "raw_eligible": p["raw_eligible"],
                    "feature_complete_denominator": p["feature_complete"],
                    "excluded_missing_inputs": p["excluded_missing_inputs"],
                    "missing_power_W": p["missing_power_W"],
                    "missing_irr_meas": p["missing_irr_meas"],
                    "missing_both": p["missing_both"],
                    "screening_rate": round(float(r["binary"].mean()), 6),
                    "score_median": round(float(np.median(r["score"])), 6),
                    **{f"pred_{LABEL_NAMES[l]}": int(vc[l]) for l in LABELS}})
st = pd.DataFrame(st_rows)
print(f"  {'population':<40}{'denom':>9}{'screen rate':>13}")
for r in st.itertuples():
    print(f"  {r.population:<40}{r.feature_complete_denominator:>9,}"
          f"{r.screening_rate*100:>12.2f}%")

cap_rows = []
for cap_lab, cap in [("5.00 kW (primary)", CAPACITY_BR),
                     ("5.28 kW (sensitivity)", CAPACITY_BR_SENS)]:
    r = apply_ensemble(prim, cap)
    vc = pd.Series(r["pred5"]).value_counts().reindex(LABELS).fillna(0).astype(int)
    cap_rows.append({"capacity": cap_lab, "scale_factor": round(cap / CAPACITY_HK, 8),
                     "denominator_rows": len(r["binary"]),
                     "screening_rate": round(float(r["binary"].mean()), 6),
                     "score_median": round(float(np.median(r["score"])), 6),
                     **{f"pred_{LABEL_NAMES[l]}": int(vc[l]) for l in LABELS}})
cap = pd.DataFrame(cap_rows)
print(f"\n  Capacity sensitivity (clean-sensor primary):")
for r in cap.itertuples():
    print(f"    {r.capacity:<24} screening rate {r.screening_rate*100:6.2f}%")
CAP_DELTA = abs(cap.screening_rate.iloc[0] - cap.screening_rate.iloc[1])

# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
a = ax[0, 0]
a.hist(prim_res["score"], bins=60, color="tab:blue", alpha=0.85)
a.set_title("Uncalibrated ensemble_anomalous_score — HK clean-sensor\n"
            "(ranking and shape only; no threshold applied)", fontsize=11)
a.set_xlabel("ensemble_anomalous_score (uncalibrated)", fontsize=10)
a.set_ylabel("rows", fontsize=10)
a.grid(alpha=0.3)

a = ax[0, 1]
xs = np.arange(5)
a.bar(xs - 0.26, [dist[(dist.population == POPS[0]["population"])
                       & (dist["class"] == c)].predicted_proportion.iloc[0]
                  for c in CANON], width=0.26, label="HK predicted", color="tab:blue")
a.bar(xs, [float(br_empirical[l]) / n for l in LABELS], width=0.26,
      label="Brazil empirical labels", color="tab:orange")
a.bar(xs + 0.26, [float(bal_counts[l]) / bal_counts.sum() for l in LABELS],
      width=0.26, label="Brazil balanced training", color="tab:green")
a.set_xticks(xs); a.set_xticklabels(CANON, rotation=20, ha="right", fontsize=9)
a.set_title("Five-class distribution — neither Brazil distribution is an\n"
            "expected HK prior", fontsize=11)
a.set_ylabel("proportion", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 0]
a.bar([f"fold {f}" for f in pf.fold], pf.screening_rate * 100, color="tab:purple")
a.axhline(SCREEN_RATE * 100, ls="--", c="k",
          label=f"ensemble {SCREEN_RATE*100:.1f}%")
a.set_title("Per-fold classifier-screened non-Normal rate\n"
            f"(denominator {POPS[0]['feature_complete']:,} rows)", fontsize=11)
a.set_ylabel("% of rows", fontsize=10); a.legend(fontsize=9); a.grid(alpha=0.3)

a = ax[1, 1]
a.bar(range(len(st)), st.screening_rate * 100,
      color=["tab:blue", "tab:red", "tab:grey", "tab:red"])
a.set_xticks(range(len(st)))
a.set_xticklabels(["clean-sensor\n(PRIMARY)", "Mar-Oct\n(corrupted)",
                   "full 2023\n(mixed)", "Saola\n(corrupted)"], fontsize=8)
a.set_title("Screening rate by population\n"
            "red = corrupted-input stress test", fontsize=11)
a.set_ylabel("% of feature-complete rows", fontsize=10); a.grid(alpha=0.3)

fig.suptitle("Objective 3 — direct transfer, Brazil-fitted pipeline applied "
             "unchanged to Hong Kong", fontsize=13)
fig.text(0.5, -0.015,
         "On HK, \"anomalous\" means classifier-screened non-Normal only — not a "
         "verified anomaly. HK has no per-timestamp labels, so the screening "
         "rate is a transfer-behaviour\nstatistic and an elevated rate may "
         "primarily reflect cross-domain mismatch rather than HK condition. "
         "No HK-fitted preprocessing, calibration or threshold was used.",
         ha="center", fontsize=9,
         bbox=dict(boxstyle="round", fc="whitesmoke", ec="grey"))
fig.tight_layout()

# ═════════════════════════════════════════════════════════════════════════════
# WRITE
# ═════════════════════════════════════════════════════════════════════════════
pd.DataFrame(fold_rows).to_csv(OUT / "objective3_fold_summary.csv", index=False)
hkout = pd.DataFrame(prim_res["mean_proba"],
                     columns=[f"mean_P_{c}" for c in CANON], index=prim.index)
hkout["pred_class"] = [LABEL_NAMES[l] for l in prim_res["pred5"]]
hkout["binary_screened_non_normal"] = prim_res["binary"]
hkout["ensemble_anomalous_score_uncalibrated"] = prim_res["score"]
for f in sorted(PIPELINES):
    hkout[f"fold{f}_pred"] = [LABEL_NAMES[l] for l in prim_res["per_fold_pred"][f]]
hkout.to_csv(OUT / "objective3_hk_screening.csv")
pf.to_csv(OUT / "objective3_per_fold_screening.csv", index=False)
pairs.to_csv(OUT / "objective3_fold_agreement.csv", index=False)
dist.to_csv(OUT / "objective3_screening_distribution.csv", index=False)
tc.to_csv(OUT / "objective3_temporal_clustering.csv", index=False)
ag.to_csv(OUT / "objective3_objective2_agreement.csv", index=False)
pd.concat([p5g, ex.assign(check="3. per-fold detail")],
          ignore_index=True).to_csv(OUT / "objective3_screening_checks.csv", index=False)
st.to_csv(OUT / "objective3_stress_tests.csv", index=False)
cap.to_csv(OUT / "objective3_capacity_sensitivity.csv", index=False)
pd.DataFrame(id_rows).to_csv(OUT / "objective3_data_identity_check.csv", index=False)
fig.savefig(OUT / "objective3_transfer_figures.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "objective3_transfer_figures.pdf", bbox_inches="tight")

REPORT = check(f"""# Objective 3 — Direct-Transfer Results

Implementation of a frozen design. Nothing was tuned, no target-domain
adaptation was applied, and no threshold was fitted on HK.

Generated: {datetime.now():%Y-%m-%dT%H:%M:%S}

## Step 2 — source correctness checkpoint (blocking)

Pooled 5x5 out-of-fold counts identical to frozen: **{cm_equal}**
Collapsed 2x2 counts identical to frozen: **{b2_equal}**

Recomputed from those counts: Macro F1 **{r_macro:.4f}** (frozen
{FROZEN_MACRO:.4f}), anomalous F1 **{r_af1:.4f}** (frozen
{float(bz.anomalous_f1):.4f}), balanced accuracy **{r_bal:.4f}** (frozen
{float(bz.balanced_accuracy):.4f}).

The five fold pipelines were persisted only after this checkpoint passed.

## Step 3 — populations and the target-side transformation

| Population | Role | Raw eligible | Feature-complete | Excluded |
|---|---|---|---|---|
""" + "\n".join(
    f"| {p['population']} | {p['label']} | {p['raw_eligible']:,} | "
    f"{p['feature_complete']:,} | {p['excluded_missing_inputs']:,} |"
    for p in POPS) + f"""

Every rate below states its denominator. HK power was converted into
Brazil-capacity-equivalent watts (x {CAPACITY_BR/CAPACITY_HK:.8f}); irradiance
passed through unchanged. The identity

`(p_hk/C_HK - mu/C_BR)/(sd/C_BR) == (p_hk*(C_BR/C_HK) - mu)/sd`

holds to **{MAX_ID_DIFF:.3e}** across all five folds.

## Steps 4-5 — transfer behaviour on the clean-sensor primary population

Classifier-screened non-Normal rate: **{SCREEN_RATE*100:.2f}%** of
{POPS[0]['feature_complete']:,} feature-complete rows.

Dominant predicted class: **{LABEL_NAMES[dom_l]}** at {dom_share*100:.2f}%.

Per-fold screening rates: """ + ", ".join(
    f"fold {r.fold} {r.screening_rate*100:.2f}%" for r in pf.itertuples()) + f"""
Unanimous binary agreement across all five folds: {unanimous_bin*100:.2f}%.

`ensemble_anomalous_score` is **uncalibrated** — the source models were trained
on balanced downsampled folds and no probability calibration was fitted. It is
used for ranking and shape only and is not an estimated probability that the HK
system is faulty. Median {np.median(prim_res['score']):.4f}, p5-p95
{q[1]:.4f}-{q[5]:.4f}. No threshold is applied to it anywhere; the binary flag
follows solely from the five-class ensemble argmax.

Temporal clustering, with runs defined by exact 15-minute timestamp continuity
(never dataframe adjacency): {len(runs):,} runs, mean
{runs.mean() if len(runs) else 0:.2f} intervals, max
{runs.max() if len(runs) else 0}, {int((runs == 1).sum()):,} singletons.

## Framing

On HK, *anomalous* means **classifier-screened non-Normal only**. It is not a
verified anomaly or fault. The clean-sensor subset has no known fault event and
equally no per-timestamp Normal ground truth, so the screening rate is a
**transfer-behaviour statistic**; an elevated rate may primarily reflect
cross-domain mismatch rather than HK condition.

Agreement with Objective 2 sensor-free Arm B / Arm C screening is descriptive
only and is not independent fault validation. Arms B/C do not consume the
corrupted measured-irradiance signal used by Objective 3, but both analyses
concern the same HK site and period and neither has verified HK fault labels;
therefore agreement cannot be presented as corroboration of a true PV event.

The five-fold arithmetic-mean ensemble is a **pre-specified deployment summary**
on top of the frozen configuration. Gate 3 validates the underlying
POWER_PLUS_IRR configuration and representation; the ensemble aggregation rule
itself was never evaluated and is not out-of-fold validated.

## Interpretation

If direct transfer performs poorly on HK, the result cannot be dismissed as a
source representation with no discriminatory value: the same two-feature
representation demonstrated substantial grouped source-domain screening
performance. Poor HK behaviour would therefore provide evidence of limited
cross-domain transfer under the documented semantic, operating-distribution and
sensing differences.

The observed behaviour is reported as consistent with limited cross-domain
transfer and plausibly influenced by the documented operating-distribution
offset, together with the semantic, sensing and extrapolation evidence. This
experiment cannot establish that the offset uniquely caused the predictions, and
no attempt was made to tune the result away.

## Result-report interpretation

### 1. Corrupted-period comparison

Aggregate screening prevalence was nearly unchanged across the two populations:
**{SCREEN_RATE*100:.2f}%** of {POPS[0]['feature_complete']:,} clean-sensor
feature-complete rows against **{mo_rate*100:.2f}%** of
{mo_pop['feature_complete']:,} March-October feature-complete rows.

**This is not a matched counterfactual comparison.** The two populations differ
in time of year and in operating conditions, not only in irradiance-sensor
integrity. Identical aggregate rates may also conceal different row-level
predictions and different class distributions. No conclusion is drawn about
whether irradiance corruption affects the predictions, in either direction.

### 2. Output concentration (descriptive)

Output on the clean-sensor primary population is **highly concentrated**:
{share_txt}. Per-fold screening behaviour is similar throughout
({pf.screening_rate.min()*100:.2f}-{pf.screening_rate.max()*100:.2f}%), with
{unanimous_bin*100:.2f}% unanimous binary agreement across all five fold models
and pairwise binary agreement of {pairs.binary_agreement.min()*100:.2f}-
{pairs.binary_agreement.max()*100:.2f}%. **The concentration is not attributable
to a single unstable fold.** No collapse threshold is applied; none was
preregistered.

### 3. Saola proxy window

The {float(st[st.population.str.startswith('Saola')].screening_rate.iloc[0])*100:.2f}%
screening rate applies to {POPS[3]['feature_complete']:,} feature-complete rows
out of {POPS[3]['raw_eligible']:,} eligible, with
{POPS[3]['excluded_missing_inputs']:,} excluded for missing transfer inputs. The
retained subset may therefore be selective. It is recorded as a
**corrupted-input stress-test observation only**, and is evidence **neither for
nor against** successful PV-event detection.

### 4. Extrapolation

Most HK rows lie within broad source support: per-fold convex-hull containment
{ex.hull_containment_pct.min():.2f}-{ex.hull_containment_pct.max():.2f}% and
occupied-bin containment {ex.bin_containment_pct.min():.2f}-
{ex.bin_containment_pct.max():.2f}%. Gross out-of-range extrapolation alone is
therefore unlikely to explain the very high screening rate. **Local and
conditional distribution shift remain possible and are not excluded by these
measures**, and approximately {100-ex.bin_containment_pct.max():.0f}-
{100-ex.bin_containment_pct.min():.0f}% of HK rows fall outside occupied
training cells.

### 5. Missing-input breakdown (verified separately per feature)

Verified by direct tabulation, never inferred from equality of total exclusion
counts:

| Population | Eligible | irr_meas missing | power_W missing | Both | Excluded |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| {q['population']} | {q['raw_eligible']:,} | {q['missing_irr_meas']:,} | "
    f"{q['missing_power_W']:,} | {q['missing_both']:,} | "
    f"{q['excluded_missing_inputs']:,} |" for q in POPS) + f"""

## Conclusion

The two-feature Brazil configuration demonstrated substantial grouped
source-domain Normal/non-Normal discrimination, but direct transfer to HK
produced a high, Shadowing-concentrated classifier-screening rate during the
clean-sensor/no-known-fault population. The pattern was stable across
source-fold models and occurred predominantly within broad source support. These
findings provide evidence of limited direct cross-domain transferability under
the documented semantic, sensing and operating-distribution differences. HK
fault-type predictions are diagnostic outputs only and are not validated
physical fault assignments.
""", "transfer report")
(OUT / "objective3_transfer_report.md").write_text(REPORT)

with open(RUN_LOG, "a") as fh:
    fh.write(f"{datetime.now():%Y-%m-%dT%H:%M:%S} | objective3_transfer.py | "
             f"input: fault_dataset.csv, objective2_grid.csv, "
             f"extended_confusion_matrices.csv, extended_ablation_summary.csv, "
             f"objective3_binary_collapse.csv, residuals_full_arm_b/c.csv | "
             f"output: Objective3_Outputs/objective3_transfer/ | notes: direct transfer, "
             f"no target adaptation; step2 count-level checkpoint PASS; "
             f"clean-sensor screening rate {SCREEN_RATE*100:.2f}%\n")

print(f"\nWrote outputs to {OUT}")
print("OBJECTIVE 3 TRANSFER COMPLETE")
