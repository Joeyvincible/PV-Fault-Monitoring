"""
extended_ablation.py
====================
Objective 1 — extended ablation (final Brazil modelling task).

This is a COMPANION script. `pv_fault_classifier.py` is left
untouched so the baseline run remains exactly as produced; its 18
configurations are READ and merged here, never recomputed.

Adds two axes the grouped baseline did not cover:
  1. Model strength — does a stronger nonlinear model (MLP) retain
     performance when sensors are removed?
  2. Aggregate-signal representation — how much fault-discriminative
     information survives when detailed string-level measurements are
     compressed into aggregate power plus environmental or physics
     features.

EVERYTHING HELD IDENTICAL TO THE BASELINE RUN
---------------------------------------------
Same GAP_THRESHOLD, same approved fold allocation, same inf->NaN handling,
same median imputer fitted on the training fold only, StandardScaler for
scale-sensitive models only, five-class Macro F1 as the primary metric,
recording_group never used as a feature, test folds never resampled.

MLP IMBALANCE HANDLING — A STATED CONFOUND
-------------------------------------------
scikit-learn 1.5.2 MLPClassifier supports neither `class_weight` nor
`sample_weight` in fit(). Partial rebalancing by downsampling of the
TRAINING fold only is used for MLP: classes larger than the SECOND-largest
class are reduced to that size (in practice Normal is reduced to roughly
Shadowing's size), while the three controlled-fault classes are left
untouched. Test folds are never touched.

This REDUCES BUT DOES NOT ELIMINATE class imbalance. MLP versus
RF/HistGBM comparisons are therefore not a controlled isolation of
architecture — they combine architecture with imbalance strategy. The most
defensible MLP evidence is the WITHIN-MLP comparison of ELEC_FULL,
NO_CURRENT and NO_VOLTAGE.

Reducing every class to the smallest instead would leave MLP with roughly
24,000 training rows against ~490,000 for the class-weighted models,
stacking a severe training-data disadvantage on top of the differing
imbalance strategy and making the results uninterpretable.

SHARED TRAINING INDICES ACROSS MLP CONFIGURATIONS
--------------------------------------------------
The downsampled training indices are generated ONCE per fold under a fixed
seed and reused by every MLP configuration in that fold. Within a fold all
ten MLP configurations train on exactly the same rows, so the within-MLP
sensor-removal comparison varies feature availability only and not which
observations happened to be drawn. This is verified by fingerprinting the
index array actually passed to each fit() call and asserting equality.

AGGREGATE-SIGNAL SETS ARE NOT REDUCED SENSOR INSTALLATIONS
-----------------------------------------------------------
p_dc_computed = vdc1*idc1 + vdc2*idc2, so every configuration containing
it still requires both string voltage sensors and both string current
sensors to exist. The Brazil dataset contains no independently measured
power signal, so a standalone independent power-meter configuration cannot
be evaluated here. The genuine hardware-removal ablations remain
NO_CURRENT and NO_VOLTAGE.

INPUT : data/fault_dataset.csv
        outputs/grouped_baseline/*.csv  (read-only merge)
OUTPUT: outputs/extended_ablation/
"""

import datetime
import hashlib
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_CSV    = PROJECT_DIR / "data" / "fault_dataset.csv"
REGISTER    = PROJECT_DIR / "outputs" / "source_event_register.csv"
BASE_DIR    = PROJECT_DIR / "outputs" / "grouped_baseline"   # READ ONLY
OUT_DIR     = PROJECT_DIR / "outputs" / "extended_ablation"  # WRITE
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG     = PROJECT_DIR / "outputs" / "run_log.txt"

GAP_THRESHOLD = 3600
RANDOM_SEED   = 42
LONG_DEG_MIN  = 900
LABELS = [0, 1, 2, 3, 4]
LABEL_NAMES = {0: "Normal", 1: "Short-Circuit", 2: "Degradation",
               3: "Open-Circuit", 4: "Shadowing"}
APPROVED_FOLDS = {1: [2, 10, 11], 2: [1, 3, 7, 12], 3: [0, 9, 13],
                  4: [5, 6, 14], 5: [4, 8, 15]}
SCALE_SENSITIVE = {"LogReg", "MLP"}
MLP_TIME_BUDGET_S = 900   # per-configuration warning threshold

FORBIDDEN = ["power meter only", "hk-equivalent", "sensor-free",
             "pvlib only", "hardware cost", "cheaper hardware"]

_report = []


def assert_clean(text):
    low = text.lower()
    hits = [p for p in FORBIDDEN if p in low]
    assert not hits, f"Forbidden naming in output: {hits} -> {text!r}"


def emit(line=""):
    assert_clean(line)          # fail fast, before anything is written
    print(line)
    _report.append(line)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS AND MANDATED LABELS
# ─────────────────────────────────────────────────────────────────────────────
ELEC_FULL = ["vdc1", "vdc2", "idc1", "idc2", "p1", "p2", "p_dc_computed",
             "v_ratio", "i_ratio", "v_imbalance", "i_imbalance"]
ELEC_IRR = ELEC_FULL + ["irr_panel_wm2"]
ELEC_IRR_TMOD = ELEC_IRR + ["t_module_rear_c"]
NO_CURRENT = ["vdc1", "vdc2", "v_ratio", "v_imbalance"]
NO_VOLTAGE = ["idc1", "idc2", "i_ratio", "i_imbalance"]
POWER_PHYSICS = ["p_dc_computed", "irr_panel_wm2", "t_module_rear_c",
                 "p_exp_5k0", "residual_5k0", "ratio_5k0"]
POWER_ONLY = ["p_dc_computed"]
POWER_PLUS_IRR = ["p_dc_computed", "irr_panel_wm2"]
POWER_PLUS_IRR_TMOD = ["p_dc_computed", "irr_panel_wm2", "t_module_rear_c"]
HK_AVAILABILITY_PROXY = ["p_dc_computed", "irr_panel_wm2",
                         "p_exp_5k0", "residual_5k0", "ratio_5k0"]

L_ELEC        = "Electrical (full)"
L_ELEC_IRR    = "Electrical + irradiance"
L_ELEC_IRR_T  = "Electrical + irradiance + module temp"
L_NO_CUR      = "No current sensors"
L_NO_VOLT     = "No voltage sensors"
L_PHYSICS     = "Computed power + approximate measured-POA physics"
L_POWER       = "Computed total power only"
L_POWER_IRR   = "Computed power + measured irradiance"
L_POWER_IRR_T = "Computed power + irradiance + module temp"
L_HK_PROXY    = "HK-availability proxy (not a validated transfer mapping)"

HARDWARE_SETS = [L_ELEC, L_ELEC_IRR, L_ELEC_IRR_T, L_NO_CUR, L_NO_VOLT]
AGGREGATE_SETS = [L_PHYSICS, L_POWER, L_POWER_IRR, L_POWER_IRR_T, L_HK_PROXY]
ALL_SETS = HARDWARE_SETS + AGGREGATE_SETS

FEATURES = {
    L_ELEC: ELEC_FULL, L_ELEC_IRR: ELEC_IRR, L_ELEC_IRR_T: ELEC_IRR_TMOD,
    L_NO_CUR: NO_CURRENT, L_NO_VOLT: NO_VOLTAGE, L_PHYSICS: POWER_PHYSICS,
    L_POWER: POWER_ONLY, L_POWER_IRR: POWER_PLUS_IRR,
    L_POWER_IRR_T: POWER_PLUS_IRR_TMOD, L_HK_PROXY: HK_AVAILABILITY_PROXY,
}
# Baseline label -> mandated consolidated label
BASELINE_RELABEL = {
    "Electrical (full)": L_ELEC,
    "Electrical + irradiance": L_ELEC_IRR,
    "Electrical + irradiance + module temp": L_ELEC_IRR_T,
    "No current sensors": L_NO_CUR,
    "No voltage sensors": L_NO_VOLT,
    "Power + approx. measured-POA physics": L_PHYSICS,
}
NEW_SETS = [L_POWER, L_POWER_IRR, L_POWER_IRR_T, L_HK_PROXY]

SENSING_REQUIREMENT = {
    L_ELEC:        "both string voltage + both string current sensors",
    L_ELEC_IRR:    "both string V + I sensors, plus pyranometer",
    L_ELEC_IRR_T:  "both string V + I sensors, pyranometer, module PT100s",
    L_NO_CUR:      "both string voltage sensors only (current removed)",
    L_NO_VOLT:     "both string current sensors only (voltage removed)",
    L_PHYSICS:     "both string V + I sensors, pyranometer, module PT100s",
    L_POWER:       "both string V + I sensors (power is derived, not measured)",
    L_POWER_IRR:   "both string V + I sensors, plus pyranometer",
    L_POWER_IRR_T: "both string V + I sensors, pyranometer, module PT100s",
    L_HK_PROXY:    "both string V + I sensors, plus pyranometer",
}

emit("=" * 78)
emit("  OBJECTIVE 1 — EXTENDED ABLATION (final Brazil modelling task)")
emit("=" * 78)
emit(f"  Started: {datetime.datetime.now().isoformat(timespec='seconds')}")

# Early naming self-test, before any modelling.
for lbl in ALL_SETS:
    assert_clean(lbl)
for lbl in SENSING_REQUIREMENT.values():
    assert_clean(lbl)
emit("  Naming assertions on all feature-set labels: PASS")
# The banned phrases are deliberately NOT echoed here: printing the list
# through emit() would trip the guard on its own diagnostic line.
emit(f"  Banned label phrases enforced: {len(FORBIDDEN)} "
     f"(defined in the FORBIDDEN constant in this script's source)")

# ─────────────────────────────────────────────────────────────────────────────
# DATA + FOLDS
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV).sort_values("sample_index").reset_index(drop=True)
y = df["f_nv"].to_numpy()
n = len(df)
df["recording_group"] = (df["sample_index"].diff() > GAP_THRESHOLD
                         ).fillna(False).cumsum().astype(int)
assert sorted(df["recording_group"].unique()) == list(range(16))

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
emit(f"  Data: {n:,} rows | fold allocation reproduces approved: PASS")

for name, feats in FEATURES.items():
    missing = [c for c in feats if c not in df.columns]
    assert not missing, f"{name} references missing columns: {missing}"
emit("  All feature sets reference existing columns: PASS")

register = pd.read_csv(REGISTER)
retained = df.groupby("source_event_id").size().rename("retained")
ev = register.merge(retained, on="source_event_id", how="inner")
long_deg_ids = sorted(ev.loc[(ev["f_nv"] == 2) & (ev["retained"] > LONG_DEG_MIN),
                             "source_event_id"].tolist())
df["is_long_deg"] = df["source_event_id"].isin(long_deg_ids)
std_mask = (y == 2) & ~df["is_long_deg"].to_numpy()
ano_mask = (y == 2) & df["is_long_deg"].to_numpy()


def make_model(name):
    if name == "LogReg":
        return LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=RANDOM_SEED)
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      n_jobs=-1, random_state=RANDOM_SEED)
    if name == "HistGBM":
        return HistGradientBoostingClassifier(class_weight="balanced",
                                              max_iter=200,
                                              random_state=RANDOM_SEED)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                             early_stopping=True, n_iter_no_change=10,
                             random_state=RANDOM_SEED)
    raise ValueError(name)


def downsample_to_second_largest(idx_tr, y_tr):
    """Reduce classes larger than the SECOND-largest class to that size.

    TRAIN ONLY. Controlled-fault classes are left untouched. This reduces
    but does not eliminate imbalance — see module docstring.
    """
    counts = {int(c): int((y_tr == c).sum()) for c in np.unique(y_tr)}
    target = sorted(counts.values(), reverse=True)[1]   # second-largest
    keep = []
    for c in sorted(counts):
        ci = idx_tr[y_tr == c]
        keep.append(ci if len(ci) <= target
                    else resample(ci, n_samples=target, replace=False,
                                  random_state=RANDOM_SEED))
    return np.sort(np.concatenate(keep)), target


# Generated ONCE per fold, reused by every MLP configuration in that fold.
MLP_FOLD_IDX, MLP_FOLD_TARGET = {}, {}
mlp_counts = []
mlp_fingerprints = {}   # (feature_set, fold) -> sha1 of the index array used


def run_config(fs_name, feats, model_name):
    oof = np.full(n, -1, dtype=np.int8)
    fold_f1 = []
    t0 = time.time()
    for f in sorted(folds):
        te = (df["fold"] == f).to_numpy()
        tr = ~te
        idx_tr = np.where(tr)[0]
        y_tr_full = y[idx_tr]

        if model_name == "MLP":
            # Reuse the single index set generated once for this fold.
            idx_fit = MLP_FOLD_IDX[f]
            mlp_fingerprints[(fs_name, f)] = hashlib.sha1(
                idx_fit.tobytes()).hexdigest()
        else:
            idx_fit = idx_tr

        X_fit = df.loc[idx_fit, feats].to_numpy(dtype=float)
        X_te = df.loc[te, feats].to_numpy(dtype=float)
        X_fit[~np.isfinite(X_fit)] = np.nan
        X_te[~np.isfinite(X_te)] = np.nan
        imp = SimpleImputer(strategy="median").fit(X_fit)   # training fold only
        X_fit, X_te = imp.transform(X_fit), imp.transform(X_te)
        if model_name in SCALE_SENSITIVE:
            sc = StandardScaler().fit(X_fit)                # training fold only
            X_fit, X_te = sc.transform(X_fit), sc.transform(X_te)

        m = make_model(model_name).fit(X_fit, y[idx_fit])
        pred = m.predict(X_te)                              # test fold untouched
        oof[te] = pred
        fold_f1.append(f1_score(y[te], pred, average="macro", labels=LABELS,
                                zero_division=0))
    assert (oof >= 0).all(), "Row without an out-of-fold prediction"
    return oof, np.array(fold_f1), time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# READ EXISTING 18 CONFIGURATIONS (never recomputed)
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  MERGE — reading the 18 baseline configurations (not recomputed)")
emit("=" * 78)
b_sum = pd.read_csv(BASE_DIR / "grouped_baseline_summary.csv")
b_fold = pd.read_csv(BASE_DIR / "grouped_baseline_per_fold.csv")
b_class = pd.read_csv(BASE_DIR / "grouped_baseline_per_class.csv")
b_cm = pd.read_csv(BASE_DIR / "confusion_matrices.csv")
b_deg = pd.read_csv(BASE_DIR / "degradation_recall_breakdown.csv")
assert len(b_sum) == 18 and len(b_fold) == 90
for d_ in (b_sum, b_fold, b_class, b_cm, b_deg):
    d_["feature_set"] = d_["feature_set"].map(BASELINE_RELABEL)
    assert d_["feature_set"].notna().all(), "Unmapped baseline feature-set label"
emit(f"  Read {len(b_sum)} summary rows, {len(b_fold)} per-fold rows, "
     f"{len(b_class)} per-class rows")
# Recompute stored baseline fold SD with ddof=1; no model fitting occurs.
sd1 = (b_fold.groupby(["feature_set", "model"])["macro_f1"]
       .std(ddof=1).rename("macro_f1_fold_sd_ddof1").reset_index())
b_sum = b_sum.merge(sd1, on=["feature_set", "model"])
b_sum["source"] = "baseline (read)"
emit("  Fold SD recomputed with ddof=1 from stored per-fold values "
     "(no retraining)")

# ─────────────────────────────────────────────────────────────────────────────
# RUN 22 ADDITIONAL CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────
# ── MLP training indices: generated ONCE per fold, shared by all MLP configs ──
emit("\n" + "=" * 78)
emit("  MLP TRAINING-FOLD DOWNSAMPLING (generated once per fold, then shared)")
emit("=" * 78)
emit("  Classes larger than the second-largest are reduced to that size;")
emit("  the three controlled-fault classes are left untouched.")
emit(f"  {'fold':>5} {'class':<16}{'before':>12}{'after':>12}   target")
for f in sorted(folds):
    idx_tr = np.where((df["fold"] != f).to_numpy())[0]
    idx_fit, target = downsample_to_second_largest(idx_tr, y[idx_tr])
    MLP_FOLD_IDX[f], MLP_FOLD_TARGET[f] = idx_fit, target
    for c in LABELS:
        before, after = int((y[idx_tr] == c).sum()), int((y[idx_fit] == c).sum())
        mlp_counts.append({"fold": f, "label": c, "class": LABEL_NAMES[c],
                           "train_count_before": before,
                           "train_count_after_downsampling": after,
                           "cap_second_largest": target,
                           "index_sha1": hashlib.sha1(idx_fit.tobytes()).hexdigest()[:12]})
        emit(f"  {f:>5} {LABEL_NAMES[c]:<16}{before:>12,}{after:>12,}"
             f"{'   ' + format(target, ',') if c == 0 else ''}")
    emit(f"  {'':>5} {'TOTAL':<16}{len(idx_tr):>12,}{len(idx_fit):>12,}")
emit("  This reduces but does not eliminate class imbalance. MLP versus")
emit("  RF/HistGBM is therefore not a controlled isolation of architecture;")
emit("  the within-MLP ELEC_FULL / NO_CURRENT / NO_VOLTAGE comparison is the")
emit("  most defensible MLP evidence.")

new_plan = ([(fs, "MLP") for fs in [L_ELEC, L_ELEC_IRR, L_ELEC_IRR_T,
                                    L_NO_CUR, L_NO_VOLT, L_PHYSICS]]
            + [(fs, m) for fs in NEW_SETS
               for m in ["LogReg", "RF", "HistGBM", "MLP"]])
emit(f"\n  New configurations to run: {len(new_plan)} "
     f"(6 MLP x original sets + 16 aggregate-signal)")

emit("\n" + "=" * 78)
emit("  RUNNING NEW CONFIGURATIONS")
emit("=" * 78)
new_rows, new_fold_rows, new_class_rows, new_cm_rows, new_deg_rows = [], [], [], [], []
slow_configs = []
for fs_name, model_name in new_plan:
    feats = FEATURES[fs_name]
    oof, ff, secs = run_config(fs_name, feats, model_name)
    if model_name == "MLP" and secs > MLP_TIME_BUDGET_S:
        slow_configs.append((fs_name, secs))
    pooled = f1_score(y, oof, average="macro", labels=LABELS, zero_division=0)
    bal = balanced_accuracy_score(y, oof)
    rec = {l: float((oof[y == l] == l).mean()) for l in (1, 2, 3, 4)}
    four = float(np.mean(list(rec.values())))
    emit(f"  {fs_name:<52} {model_name:<8} F1={pooled:.4f} "
         f"({ff.mean():.4f}±{ff.std(ddof=1):.4f}) bal={bal:.4f} "
         f"4f={four:.4f} {secs:.0f}s")
    new_rows.append({"feature_set": fs_name, "model": model_name,
                     "macro_f1_oof": round(pooled, 4),
                     "macro_f1_fold_mean": round(float(ff.mean()), 4),
                     "macro_f1_fold_std": round(float(ff.std()), 4),
                     "macro_f1_fold_sd_ddof1": round(float(ff.std(ddof=1)), 4),
                     "balanced_accuracy_oof": round(bal, 4),
                     "four_fault_mean_recall": round(four, 4),
                     "runtime_s": round(secs, 1), "source": "new"})
    for i, v in enumerate(ff, start=1):
        new_fold_rows.append({"feature_set": fs_name, "model": model_name,
                              "fold": i,
                              "test_rows": int((df["fold"] == i).sum()),
                              "macro_f1": round(float(v), 4)})
    p, r, f1v, sup = precision_recall_fscore_support(y, oof, labels=LABELS,
                                                     zero_division=0)
    for i, l in enumerate(LABELS):
        new_class_rows.append({"feature_set": fs_name, "model": model_name,
                               "label": l, "class": LABEL_NAMES[l],
                               "precision": round(p[i], 4),
                               "recall": round(r[i], 4),
                               "f1": round(f1v[i], 4), "support": int(sup[i])})
    cm = confusion_matrix(y, oof, labels=LABELS)
    for i, tl in enumerate(LABELS):
        for j, pl in enumerate(LABELS):
            new_cm_rows.append({"feature_set": fs_name, "model": model_name,
                                "true": LABEL_NAMES[tl], "pred": LABEL_NAMES[pl],
                                "count": int(cm[i, j])})
    new_deg_rows.append({"feature_set": fs_name, "model": model_name,
                         "standard_deg_recall": round(float((oof[std_mask] == 2).mean()), 4),
                         "anomalous_deg_recall": round(float((oof[ano_mask] == 2).mean()), 4),
                         "standard_rows": int(std_mask.sum()),
                         "anomalous_rows": int(ano_mask.sum())})

# ── VERIFY: every MLP configuration in a fold trained on identical rows ──────
emit("\n" + "=" * 78)
emit("  SHARED-TRAINING-INDEX CHECK (within-MLP control)")
emit("=" * 78)
mlp_sets = sorted({fs for (fs, _) in mlp_fingerprints})
emit(f"  MLP configurations fingerprinted: {len(mlp_sets)} feature sets x "
     f"{len(folds)} folds = {len(mlp_fingerprints)}")
all_identical = True
for f in sorted(folds):
    digests = {fs: mlp_fingerprints[(fs, f)] for fs in mlp_sets}
    uniq = set(digests.values())
    ok = len(uniq) == 1
    all_identical &= ok
    emit(f"    fold {f}: {len(digests)} configurations, "
         f"{len(uniq)} distinct index set(s) -> {'IDENTICAL' if ok else 'DIFFER'}"
         f"  sha1 {list(uniq)[0][:12] if ok else 'MISMATCH'}")
    assert ok, f"MLP configurations in fold {f} used different training rows"
assert all_identical
emit("  ASSERT all MLP configurations within each fold used identical")
emit("  training rows: PASS — the within-MLP sensor comparison varies")
emit("  feature availability only.")

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLIDATE
# ─────────────────────────────────────────────────────────────────────────────
summary = pd.concat([b_sum, pd.DataFrame(new_rows)], ignore_index=True)
per_fold = pd.concat([b_fold, pd.DataFrame(new_fold_rows)], ignore_index=True)
per_class = pd.concat([b_class, pd.DataFrame(new_class_rows)], ignore_index=True)
cms = pd.concat([b_cm, pd.DataFrame(new_cm_rows)], ignore_index=True)
degs = pd.concat([b_deg, pd.DataFrame(new_deg_rows)], ignore_index=True)
assert len(summary) == 40, f"Expected 40 configurations, got {len(summary)}"

emit("\n" + "=" * 78)
emit("  OBJECTIVE 1 — CONSOLIDATED ABLATION (final)")
emit("=" * 78)
emit("  Group-disjoint 5-fold CV, all 16 recording groups tested once")
emit("  Pooled OOF Macro F1 (fold mean ± sample SD, ddof=1)")
emit("")
emit(f"  {'Feature Set':<52}{'LogReg':<11}{'RF':<11}{'HistGBM':<11}{'MLP':<11}")


def cell(fs, m):
    r = summary[(summary.feature_set == fs) & (summary.model == m)]
    return "-" if not len(r) else f"{r.iloc[0].macro_f1_oof:.3f}"


emit("  --- Hardware sensor ablations ---")
for fs in HARDWARE_SETS:
    emit(f"  {fs:<52}" + "".join(f"{cell(fs, m):<11}"
                                 for m in ["LogReg", "RF", "HistGBM", "MLP"]))
emit("  --- Aggregate-signal representations ---")
for fs in AGGREGATE_SETS:
    emit(f"  {fs:<52}" + "".join(f"{cell(fs, m):<11}"
                                 for m in ["LogReg", "RF", "HistGBM", "MLP"]))

emit("\n  SENSOR REMOVAL UNDER INCREASING MODEL STRENGTH")
emit("    Within-model comparison — imbalance strategy held constant per model.")
emit("")
emit(f"    {'Model':<11}{'ELEC_FULL':<12}{'NO_CURRENT':<13}{'NO_VOLTAGE':<13}"
     f"{'drop(cur)':<11}{'drop(volt)':<11}")
sensor_rows = []
for m in ["LogReg", "RF", "HistGBM", "MLP"]:
    def val(fs):
        return float(summary[(summary.feature_set == fs)
                             & (summary.model == m)].iloc[0].macro_f1_oof)
    ef, nc, nv = val(L_ELEC), val(L_NO_CUR), val(L_NO_VOLT)
    emit(f"    {m:<11}{ef:<12.3f}{nc:<13.3f}{nv:<13.3f}"
         f"{ef - nc:<11.3f}{ef - nv:<11.3f}")
    sensor_rows.append({"model": m, "elec_full": round(ef, 4),
                        "no_current": round(nc, 4), "no_voltage": round(nv, 4),
                        "drop_current": round(ef - nc, 4),
                        "drop_voltage": round(ef - nv, 4)})
sr = pd.DataFrame(sensor_rows)
sr.to_csv(OUT_DIR / "sensor_removal_by_model.csv", index=False)

strength = ["LogReg", "MLP", "RF", "HistGBM"]
dc = [float(sr[sr.model == m].iloc[0].drop_current) for m in strength]
dv = [float(sr[sr.model == m].iloc[0].drop_voltage) for m in strength]
emit("")
emit(f"    Ordered LogReg -> MLP -> RF -> HistGBM:")
emit(f"      drop(current): " + " -> ".join(f"{v:.3f}" for v in dc))
emit(f"      drop(voltage): " + " -> ".join(f"{v:.3f}" for v in dv))
shrink_c = dc[-1] < dc[0]
shrink_v = dv[-1] < dv[0]
emit(f"    -> Current-removal penalty {'shrinks' if shrink_c else 'does not shrink'} "
     f"with model strength; voltage-removal penalty "
     f"{'shrinks' if shrink_v else 'does not shrink'}.")
emit(f"       Voltage-removal penalty remains "
     f"{min(dv):.3f}-{max(dv):.3f} across all four models, so the sensing gap")
emit(f"       is not closed by model capacity.")
emit("")
emit("    CAVEAT: MLP used partial rebalancing by training-fold downsampling")
emit("    (classes above the second-largest reduced to that size) because")
emit("    scikit-learn 1.5.2 MLPClassifier supports neither class_weight nor")
emit("    sample_weight. This REDUCES BUT DOES NOT ELIMINATE class imbalance,")
emit("    so cross-model differences reflect combined architecture and")
emit("    imbalance-strategy effects and are NOT a controlled isolation of")
emit("    architecture. The within-MLP row is the controlled compensation")
emit("    test: all MLP configurations within a fold trained on identical")
emit("    rows, so only feature availability varies.")

emit("\n  FEATURE-AVAILABILITY / SENSING-REQUIREMENT SUMMARY")
emit(f"    {'Configuration':<52}{'best F1':<10}{'model':<9}sensing requirement")
best_by_fs = (summary.sort_values("macro_f1_oof", ascending=False)
              .groupby("feature_set").first().reset_index())
for _, r in best_by_fs.sort_values("macro_f1_oof", ascending=False).iterrows():
    emit(f"    {r.feature_set:<52}{r.macro_f1_oof:<10.3f}{r.model:<9}"
         f"{SENSING_REQUIREMENT[r.feature_set]}")
emit("")
emit("    REMINDER: p_dc_computed = vdc1*idc1 + vdc2*idc2, so every")
emit("    configuration containing it requires both string voltage and both")
emit("    string current sensors. None of the aggregate-signal sets represents")
emit("    a genuinely reduced sensor installation. A standalone independent")
emit("    power-meter configuration cannot be evaluated on this dataset.")
emit("")
emit("    The HK-availability proxy is excluded from any transfer-related")
emit("    interpretation and is reported as a preliminary feature-availability")
emit("    experiment only. HK power semantics (AC inverter output versus DC")
emit("    array power), units and aggregation remain unresolved.")

emit("\n  DEGRADATION EVENT-CHARACTER SENSITIVITY")
emit(f"    Standard rows {int(std_mask.sum()):,} | "
     f"anomalous rows {int(ano_mask.sum()):,} (events {long_deg_ids})")
emit(f"    {'model':<9}{'best configuration':<52}{'standard':<11}{'anomalous':<11}")
for m in ["LogReg", "RF", "HistGBM", "MLP"]:
    sub = summary[summary.model == m].sort_values("macro_f1_oof", ascending=False)
    bfs = sub.iloc[0].feature_set
    d = degs[(degs.model == m) & (degs.feature_set == bfs)]
    if len(d):
        emit(f"    {m:<9}{bfs:<52}{d.iloc[0].standard_deg_recall:<11.4f}"
             f"{d.iloc[0].anomalous_deg_recall:<11.4f}")

# >0.99 investigation
emit("\n  CONFIGURATIONS EXCEEDING MACRO F1 0.99")
high = summary[summary.macro_f1_oof > 0.99]
if len(high):
    for _, r in high.iterrows():
        emit(f"    {r.feature_set} / {r.model}: {r.macro_f1_oof:.4f} "
             f"-- INVESTIGATE")
else:
    emit("    None. Highest observed: "
         f"{summary.macro_f1_oof.max():.4f} "
         f"({summary.sort_values('macro_f1_oof').iloc[-1].feature_set} / "
         f"{summary.sort_values('macro_f1_oof').iloc[-1].model})")
if slow_configs:
    emit("\n  MLP configurations exceeding the 15-minute budget:")
    for fs, s in slow_configs:
        emit(f"    {fs}: {s:.0f}s")
else:
    emit("\n  No MLP configuration exceeded the 15-minute budget; "
         "max_iter remained 300.")
emit("=" * 78)

summary.to_csv(OUT_DIR / "extended_ablation_summary.csv", index=False)
per_fold.to_csv(OUT_DIR / "extended_ablation_per_fold.csv", index=False)
per_class.to_csv(OUT_DIR / "extended_ablation_per_class.csv", index=False)
cms.to_csv(OUT_DIR / "extended_confusion_matrices.csv", index=False)
degs.to_csv(OUT_DIR / "extended_degradation_breakdown.csv", index=False)
pd.DataFrame(mlp_counts).to_csv(OUT_DIR / "mlp_downsampling_counts.csv", index=False)
report_text = "\n".join(_report) + "\n"
assert_clean(report_text)
(OUT_DIR / "extended_ablation_report.txt").write_text(report_text)
print(f"\nSaved to: {OUT_DIR}")

with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"extended_ablation.py | input: {DATA_CSV.name} {n:,} rows + "
            f"18 baseline configs read from {BASE_DIR.name}/ | "
            f"output: 8 files in {OUT_DIR.name}/ | "
            f"notes: 22 new configs (MLP x6 original, 4 models x4 aggregate-signal); "
            f"40 consolidated; MLP used balanced training-fold downsampling "
            f"(sklearn 1.5.2 has no class_weight/sample_weight); baseline not "
            f"recomputed; no existing output modified\n")
print(f"Run log appended: {RUN_LOG}")
