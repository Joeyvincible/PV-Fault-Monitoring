"""
pv_fault_classifier.py
================================
Objective 1, Step 5 — Brazil classifier with group-disjoint
5-fold cross-validation over all 16 recording groups.

This is a NEW script. The historical `pv_fault_classifier.py` is left
untouched as the before-correction record, still reading the historical
`fault_dataset_cleaned.csv`. This script reads ONLY
`data/fault_dataset.csv`.

WHY GROUPED CV
--------------
The historical result (Macro F1 0.9975) used random row-level splitting on
1 Hz data. Adjacent seconds of the same physical fault event landed in both
train and test, so the model was largely retrieving memorised neighbours.
Here every fold's test set is a set of whole recording days, and no sample
from a test day appears in training.

EVENT STRUCTURE (descriptive metadata — f_nv is never modified)
---------------------------------------------------------------
Raw contiguous-run counts of 15 short-circuit / 19 degradation / 23
open-circuit runs are inflated by short label fragments at event edges.
Clustering nearby fragments shows 10 standard short-circuit, 10 standard
degradation and 10 standard open-circuit inductions across groups 11-15,
each approximately matching the paper's 10-minute (~600 sample) schedule.
Groups 8 and 9 additionally contain two longer degradation-labelled events
(1,779 and 2,695 samples) that do not follow the standard pattern.

This clustering is descriptive only. No row is removed, relabelled, or
excluded; every labelled row participates in training and evaluation. The
long degradation events are reported as a separate recall breakdown purely
as a diagnostic.

All controlled-fault events lie wholly within recording-group boundaries.
The only source events spanning reconstructed overnight boundaries are
Normal, so the grouped evaluation never splits a controlled physical
induction across train and test.

INPUT : data/fault_dataset.csv
OUTPUT: outputs/grouped_baseline/
"""

import datetime
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_CSV    = PROJECT_DIR / "data" / "fault_dataset.csv"
REGISTER    = PROJECT_DIR / "outputs" / "source_event_register.csv"
OUT_DIR     = PROJECT_DIR / "outputs" / "grouped_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG     = PROJECT_DIR / "outputs" / "run_log.txt"

GAP_THRESHOLD = 3600          # inside the established stable band (387-42,335)
RANDOM_SEED   = 42
NOMINAL_EVENT = 600           # paper: 10 min at 1 Hz
LONG_DEG_MIN  = 900           # > 1.5x nominal marks the anomalous long events

LABEL_NAMES = {0: "Normal", 1: "Short-Circuit", 2: "Degradation",
               3: "Open-Circuit", 4: "Shadowing"}
LABELS = [0, 1, 2, 3, 4]

# Fold allocation approved by the user (deterministic greedy, anchors 11-15).
APPROVED_FOLDS = {
    1: [2, 10, 11],
    2: [1, 3, 7, 12],
    3: [0, 9, 13],
    4: [5, 6, 14],
    5: [4, 8, 15],
}

_report = []


def emit(line=""):
    print(line)
    _report.append(line)


emit("=" * 78)
emit("  OBJECTIVE 1 — GROUP-DISJOINT CLASSIFIER, 5-FOLD CV")
emit("=" * 78)
emit(f"  Started: {datetime.datetime.now().isoformat(timespec='seconds')}")

# ─────────────────────────────────────────────────────────────────────────────
# 5a. LOAD + RECORDING GROUPS
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  5a. RECORDING GROUP RECONSTRUCTION")
emit("=" * 78)

df = pd.read_csv(DATA_CSV).sort_values("sample_index").reset_index(drop=True)
emit(f"  Loaded: {DATA_CSV.name}  {df.shape[0]:,} rows x {df.shape[1]} cols")

df["recording_group"] = (df["sample_index"].diff() > GAP_THRESHOLD
                         ).fillna(False).cumsum().astype(int)
groups = sorted(df["recording_group"].unique())
emit(f"  Gap threshold: {GAP_THRESHOLD:,} (inside stable band 387-42,335)")
emit(f"  Recording groups recovered: {len(groups)} -> {groups}")

assert len(groups) == 16, f"Expected 16 recording groups, got {len(groups)}"
assert groups == list(range(16)), "Groups must be numbered 0-15 in original order"
for g in [11, 12, 13, 14, 15]:
    present = set(df.loc[df["recording_group"] == g, "f_nv"].unique())
    assert {1, 2, 3}.issubset(present), f"Group {g} missing controlled labels"
emit("  ASSERT 16 groups numbered 0-15 in original order          : PASS")
emit("  ASSERT groups 11-15 each contain labels 1, 2 and 3        : PASS")
emit("  NOTE: recording_group is a split key only — never a model feature.")

# Event metadata for the degradation diagnostic breakdown.
register = pd.read_csv(REGISTER)
retained = df.groupby("source_event_id").size().rename("retained")
ev = register.merge(retained, on="source_event_id", how="inner")
long_deg_ids = sorted(ev.loc[(ev["f_nv"] == 2) & (ev["retained"] > LONG_DEG_MIN),
                             "source_event_id"].tolist())
emit(f"\n  Long anomalous degradation events (> {LONG_DEG_MIN} samples): "
     f"{long_deg_ids}")
for eid in long_deg_ids:
    r = ev[ev["source_event_id"] == eid].iloc[0]
    g = int(df.loc[df["source_event_id"] == eid, "recording_group"].iloc[0])
    emit(f"    event {eid}: {int(r['retained']):,} samples, group {g}")
df["is_long_deg"] = df["source_event_id"].isin(long_deg_ids)
emit(f"  Rows in long degradation events: {int(df['is_long_deg'].sum()):,} "
     f"(kept in training and evaluation — diagnostic split only)")

# ─────────────────────────────────────────────────────────────────────────────
# 5b. FOLD ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  5b. FOLD ASSIGNMENT")
emit("=" * 78)
emit("  Rule: anchors 11-15 seed folds 1-5. Remaining groups are assigned")
emit("  largest-first to whichever fold currently holds the fewest test rows")
emit("  (ties by lowest fold number), with groups 8 and 9 forced into")
emit("  different folds. Determined only from group size and the support")
emit("  table — never from model performance.")

sizes = df.groupby("recording_group").size()
anchors = [11, 12, 13, 14, 15]
folds = {i + 1: [a] for i, a in enumerate(anchors)}
totals = {i + 1: int(sizes[a]) for i, a in enumerate(anchors)}
for g in sorted([x for x in range(16) if x not in anchors],
                key=lambda x: (-int(sizes[x]), x)):
    for f in sorted(folds, key=lambda f: (totals[f], f)):
        if g == 8 and 9 in folds[f]:
            continue
        if g == 9 and 8 in folds[f]:
            continue
        folds[f].append(g)
        totals[f] += int(sizes[g])
        break
folds = {f: sorted(v) for f, v in folds.items()}

assert folds == APPROVED_FOLDS, (
    f"Fold allocation differs from the approved one.\n"
    f"  computed: {folds}\n  approved: {APPROVED_FOLDS}")
emit("  ASSERT computed allocation == user-approved allocation     : PASS")

df["fold"] = 0
for f, gs in folds.items():
    df.loc[df["recording_group"].isin(gs), "fold"] = f
assert (df["fold"] > 0).all(), "Some rows were not assigned to a fold"

big_ctrl = ev[(ev["f_nv"].isin([1, 2, 3])) & (ev["retained"] >= 300)]
grp_of_event = df.groupby("source_event_id")["recording_group"].first()
big_ctrl = big_ctrl.assign(grp=big_ctrl["source_event_id"].map(grp_of_event))

fold_rows = []
emit(f"\n  {'Fold':>4} {'Test groups':<18} {'Test rows':>10} "
     + " ".join(f"{LABEL_NAMES[l][:6]:>7}" for l in LABELS) + "  SC/Deg/OC ev")
for f in sorted(folds):
    m = df[df["fold"] == f]
    counts = {l: int((m["f_nv"] == l).sum()) for l in LABELS}
    evc = [int(big_ctrl[(big_ctrl["grp"].isin(folds[f]))
                        & (big_ctrl["f_nv"] == l)]["source_event_id"].nunique())
           for l in (1, 2, 3)]
    emit(f"  {f:>4} {str(folds[f]):<18} {len(m):>10,} "
         + " ".join(f"{counts[l]:>7,}" for l in LABELS)
         + f"  {evc[0]}/{evc[1]}/{evc[2]}")
    for l in LABELS:
        assert counts[l] > 0, f"Fold {f} has zero test support for label {l}"
    row = {"fold": f, "test_groups": " ".join(map(str, folds[f])),
           "test_rows": len(m), "n_long_deg_rows": int(m["is_long_deg"].sum())}
    row.update({f"label_{l}_{LABEL_NAMES[l]}": counts[l] for l in LABELS})
    row.update({"sc_events": evc[0], "deg_events": evc[1], "oc_events": evc[2]})
    fold_rows.append(row)
emit("  ASSERT every fold has non-zero test support for labels 0-4 : PASS")
pd.DataFrame(fold_rows).to_csv(OUT_DIR / "fold_assignments.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 5c. FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  5c. FEATURE SETS AND EXCLUSION ASSERTIONS")
emit("=" * 78)

ELEC_FULL = ["vdc1", "vdc2", "idc1", "idc2", "p1", "p2", "p_dc_computed",
             "v_ratio", "i_ratio", "v_imbalance", "i_imbalance"]
ELEC_IRR = ELEC_FULL + ["irr_panel_wm2"]
ELEC_IRR_TMOD = ELEC_IRR + ["t_module_rear_c"]
NO_CURRENT = ["vdc1", "vdc2", "v_ratio", "v_imbalance"]
NO_VOLTAGE = ["idc1", "idc2", "i_ratio", "i_imbalance"]
POWER_PHYSICS = ["p_dc_computed", "irr_panel_wm2", "t_module_rear_c",
                 "p_exp_5k0", "residual_5k0", "ratio_5k0"]

FEATURE_SETS = {
    "Electrical (full)": ELEC_FULL,
    "Electrical + irradiance": ELEC_IRR,
    "Electrical + irradiance + module temp": ELEC_IRR_TMOD,
    "No current sensors": NO_CURRENT,
    "No voltage sensors": NO_VOLTAGE,
    # NEVER "pvlib only" / "no sensor": this needs a pyranometer AND the
    # electrical measurements that power is computed from.
    "Power + approx. measured-POA physics": POWER_PHYSICS,
}

CURRENT_DERIVED = {"idc1", "idc2", "p1", "p2", "p_dc_computed", "i_ratio",
                   "i_imbalance", "p_exp_5k0", "residual_5k0", "ratio_5k0",
                   "p_exp_5k28", "residual_5k28", "ratio_5k28"}
VOLTAGE_DERIVED = {"vdc1", "vdc2", "p1", "p2", "p_dc_computed", "v_ratio",
                   "v_imbalance", "p_exp_5k0", "residual_5k0", "ratio_5k0",
                   "p_exp_5k28", "residual_5k28", "ratio_5k28"}

viol_c = sorted(set(NO_CURRENT) & CURRENT_DERIVED)
viol_v = sorted(set(NO_VOLTAGE) & VOLTAGE_DERIVED)
emit(f"  NO_CURRENT ∩ current-derived features : {viol_c if viol_c else 'EMPTY'}")
emit(f"  NO_VOLTAGE ∩ voltage-derived features : {viol_v if viol_v else 'EMPTY'}")
assert not viol_c, f"NO_CURRENT contains current-derived features: {viol_c}"
assert not viol_v, f"NO_VOLTAGE contains voltage-derived features: {viol_v}"
emit("  ASSERT NO_CURRENT excludes all current-derived features     : PASS")
emit("  ASSERT NO_VOLTAGE excludes all voltage-derived features     : PASS")
emit("  (physics features p_exp/residual/ratio require measured power,")
emit("   so they are excluded from BOTH hardware-removal sets.)")

emit("")
for name, feats in FEATURE_SETS.items():
    missing = [f for f in feats if f not in df.columns]
    assert not missing, f"{name} references missing columns: {missing}"
    emit(f"  {name:<40} {len(feats):>2} features")

# ─────────────────────────────────────────────────────────────────────────────
# 5d. NaN / INF AUDIT
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  5d. NaN / INF AUDIT (pre-imputation, whole dataset)")
emit("=" * 78)
emit("  Undefined ratios are imputed, never dropped: vdc2 -> 0 during")
emit("  open-circuit faults, exactly the class most needing evaluation.")
all_feats = sorted({f for v in FEATURE_SETS.values() for f in v}
                   | {"p_exp_5k28", "residual_5k28", "ratio_5k28"})
emit(f"\n  {'feature':<22} {'NaN':>10} {'+/-inf':>10} {'total bad':>11}")
nan_audit = []
for f in all_feats:
    col = df[f]
    n_nan = int(col.isna().sum())
    n_inf = int(np.isinf(col.to_numpy(dtype=float)).sum())
    if n_nan or n_inf:
        emit(f"  {f:<22} {n_nan:>10,} {n_inf:>10,} {n_nan + n_inf:>11,}")
    nan_audit.append({"feature": f, "n_nan": n_nan, "n_inf": n_inf})
if not any(a["n_nan"] or a["n_inf"] for a in nan_audit):
    emit("  (no NaN or inf present in any feature)")

# ─────────────────────────────────────────────────────────────────────────────
# 5e. MODELS
# ─────────────────────────────────────────────────────────────────────────────
import inspect
HGB_HAS_CW = "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters


def make_models():
    hgb_kw = {"max_iter": 200, "random_state": RANDOM_SEED}
    if HGB_HAS_CW:
        hgb_kw["class_weight"] = "balanced"
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=RANDOM_SEED),
        "RF": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     n_jobs=-1, random_state=RANDOM_SEED),
        "HistGBM": HistGradientBoostingClassifier(**hgb_kw),
    }


emit("\n" + "=" * 78)
emit("  5e. MODELS")
emit("=" * 78)
emit(f"  HistGradientBoostingClassifier supports class_weight: {HGB_HAS_CW}")
emit(f"  -> {'using class_weight=balanced' if HGB_HAS_CW else 'using balanced sample_weight'}")
emit("  No downsampling: all training observations are preserved.")
emit("  Imputer (and scaler for LogReg) fitted on the TRAINING fold only.")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
y_all = df["f_nv"].to_numpy()
n = len(df)


def run_config(feats, model_name, tag):
    """Group-disjoint CV for one feature set x model. Returns OOF preds."""
    oof = np.full(n, -1, dtype=np.int8)
    per_fold, imputed_total = [], 0
    t0 = time.time()
    for f in sorted(folds):
        te = (df["fold"] == f).to_numpy()
        tr = ~te
        X_tr = df.loc[tr, feats].to_numpy(dtype=float)
        X_te = df.loc[te, feats].to_numpy(dtype=float)
        X_tr[~np.isfinite(X_tr)] = np.nan
        X_te[~np.isfinite(X_te)] = np.nan
        imputed_total += int(np.isnan(X_tr).sum() + np.isnan(X_te).sum())

        imp = SimpleImputer(strategy="median").fit(X_tr)
        X_tr, X_te = imp.transform(X_tr), imp.transform(X_te)
        if model_name == "LogReg":
            sc = StandardScaler().fit(X_tr)
            X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

        y_tr = y_all[tr]
        model = make_models()[model_name]
        if model_name == "HistGBM" and not HGB_HAS_CW:
            model.fit(X_tr, y_tr,
                      sample_weight=compute_sample_weight("balanced", y_tr))
        else:
            model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        oof[te] = pred
        per_fold.append({
            "feature_set": tag, "model": model_name, "fold": f,
            "test_rows": int(te.sum()),
            "macro_f1": f1_score(y_all[te], pred, average="macro",
                                 labels=LABELS, zero_division=0),
        })
    assert (oof >= 0).all(), "Some rows received no out-of-fold prediction"
    return oof, per_fold, time.time() - t0, imputed_total


summary_rows, per_fold_rows, per_class_rows, cm_rows = [], [], [], []
oof_store = {}

emit("\n" + "=" * 78)
emit("  RUNNING 6 FEATURE SETS x 3 MODELS = 18 CONFIGURATIONS")
emit("=" * 78)

for fs_name, feats in FEATURE_SETS.items():
    emit(f"\n  {fs_name} ({len(feats)} features)")
    for model_name in ["LogReg", "RF", "HistGBM"]:
        oof, pf, secs, n_imp = run_config(feats, model_name, fs_name)
        oof_store[(fs_name, model_name)] = oof
        per_fold_rows.extend(pf)
        fold_f1 = np.array([p["macro_f1"] for p in pf])
        macro_oof = f1_score(y_all, oof, average="macro", labels=LABELS,
                             zero_division=0)
        bal_acc = balanced_accuracy_score(y_all, oof)
        rec = {l: float((oof[(y_all == l)] == l).mean()) for l in (1, 2, 3, 4)}
        four_fault = float(np.mean([rec[l] for l in (1, 2, 3, 4)]))
        emit(f"    {model_name:<8} MacroF1 OOF={macro_oof:.4f}  "
             f"folds={fold_f1.mean():.4f}+/-{fold_f1.std():.4f}  "
             f"balAcc={bal_acc:.4f}  4fault={four_fault:.4f}  "
             f"({secs:.0f}s, {n_imp:,} imputed)")
        summary_rows.append({
            "feature_set": fs_name, "model": model_name,
            "macro_f1_oof": round(macro_oof, 4),
            "macro_f1_fold_mean": round(float(fold_f1.mean()), 4),
            "macro_f1_fold_std": round(float(fold_f1.std()), 4),
            "balanced_accuracy_oof": round(bal_acc, 4),
            "four_fault_mean_recall": round(four_fault, 4),
            "runtime_s": round(secs, 1),
        })
        p, r, f1v, sup = precision_recall_fscore_support(
            y_all, oof, labels=LABELS, zero_division=0)
        for i, l in enumerate(LABELS):
            per_class_rows.append({
                "feature_set": fs_name, "model": model_name,
                "label": l, "class": LABEL_NAMES[l],
                "precision": round(p[i], 4), "recall": round(r[i], 4),
                "f1": round(f1v[i], 4), "support": int(sup[i])})
        cm = confusion_matrix(y_all, oof, labels=LABELS)
        for i, tl in enumerate(LABELS):
            for j, pl in enumerate(LABELS):
                cm_rows.append({"feature_set": fs_name, "model": model_name,
                                "true": LABEL_NAMES[tl], "pred": LABEL_NAMES[pl],
                                "count": int(cm[i, j])})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "grouped_baseline_summary.csv", index=False)
pd.DataFrame(per_fold_rows).to_csv(OUT_DIR / "grouped_baseline_per_fold.csv", index=False)
pd.DataFrame(per_class_rows).to_csv(OUT_DIR / "grouped_baseline_per_class.csv", index=False)
pd.DataFrame(cm_rows).to_csv(OUT_DIR / "confusion_matrices.csv", index=False)

oof_df = pd.DataFrame({"sample_index": df["sample_index"],
                       "recording_group": df["recording_group"],
                       "fold": df["fold"], "f_nv_true": y_all,
                       "is_long_deg": df["is_long_deg"].to_numpy()})
for (fs, mn), o in oof_store.items():
    oof_df[f"pred__{fs}__{mn}"] = o
oof_df.to_csv(OUT_DIR / "out_of_fold_predictions.csv", index=False)

best = summary.sort_values("macro_f1_oof", ascending=False).iloc[0]
best_fs, best_model = best["feature_set"], best["model"]

# ─────────────────────────────────────────────────────────────────────────────
# DEGRADATION RECALL BREAKDOWN (diagnostic only — no rows removed)
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  DEGRADATION RECALL BREAKDOWN (diagnostic only)")
emit("=" * 78)
emit("  Standard  = scheduled ~600-sample degradation inductions")
emit("  Anomalous = the two long degradation-labelled events in groups 8 and 9")
emit("  No sample was removed or relabelled; this is a reporting split only.")
is_deg = (y_all == 2)
is_long = df["is_long_deg"].to_numpy()
std_mask, ano_mask = is_deg & ~is_long, is_deg & is_long
emit(f"\n  Standard degradation rows : {int(std_mask.sum()):,}")
emit(f"  Anomalous degradation rows: {int(ano_mask.sum()):,} "
     f"(events {long_deg_ids})")
emit(f"\n  {'feature set':<40} {'model':<8} {'std recall':>11} {'anom recall':>12}")
deg_rows = []
for (fs, mn), o in oof_store.items():
    rs = float((o[std_mask] == 2).mean())
    ra = float((o[ano_mask] == 2).mean())
    deg_rows.append({"feature_set": fs, "model": mn,
                     "standard_deg_recall": round(rs, 4),
                     "anomalous_deg_recall": round(ra, 4),
                     "standard_rows": int(std_mask.sum()),
                     "anomalous_rows": int(ano_mask.sum())})
    if fs == best_fs:
        emit(f"  {fs:<40} {mn:<8} {rs:>11.4f} {ra:>12.4f}")
pd.DataFrame(deg_rows).to_csv(OUT_DIR / "degradation_recall_breakdown.csv", index=False)
emit("  (full table: degradation_recall_breakdown.csv)")

# ─────────────────────────────────────────────────────────────────────────────
# 5g. CAPACITY SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  5g. CAPACITY SENSITIVITY (POWER_PHYSICS only)")
emit("=" * 78)
PHYS_5K28 = ["p_dc_computed", "irr_panel_wm2", "t_module_rear_c",
             "p_exp_5k28", "residual_5k28", "ratio_5k28"]
phys_model = summary[summary["feature_set"] == "Power + approx. measured-POA physics"] \
    .sort_values("macro_f1_oof", ascending=False).iloc[0]["model"]
emit(f"  Best model for this feature set: {phys_model}")
cap_rows = []
for label, feats in [("5.00 kW (rated)", POWER_PHYSICS), ("5.28 kW (nameplate)", PHYS_5K28)]:
    oof, pf, secs, _ = run_config(feats, phys_model, "capacity")
    ff = np.array([p["macro_f1"] for p in pf])
    m = f1_score(y_all, oof, average="macro", labels=LABELS, zero_division=0)
    emit(f"  {label:<22} Macro F1 OOF={m:.4f}  folds={ff.mean():.4f}+/-{ff.std():.4f}")
    cap_rows.append({"capacity": label, "model": phys_model,
                     "macro_f1_oof": round(m, 4),
                     "macro_f1_fold_mean": round(float(ff.mean()), 4),
                     "macro_f1_fold_std": round(float(ff.std()), 4)})
pd.DataFrame(cap_rows).to_csv(OUT_DIR / "capacity_sensitivity.csv", index=False)
cap_diff = abs(cap_rows[0]["macro_f1_oof"] - cap_rows[1]["macro_f1_oof"])
emit(f"  Difference: {cap_diff:.4f} -> "
     f"{'INSENSITIVE' if cap_diff < 0.01 else 'SENSITIVE'} to the capacity ambiguity")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — TERMINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  OBJECTIVE 1 — GROUPED BASELINE")
emit("=" * 78)
emit("  Split: group-disjoint 5-fold CV over all 16 recording groups")
emit("  Each fold: 1 controlled-fault anchor day + background groups")
emit("  Every recording group tested exactly once")
emit("  Allocation rule: anchors 11-15 seed folds 1-5; remaining groups")
emit("    assigned largest-first to the fold with fewest test rows,")
emit("    groups 8 and 9 forced apart; sizes/support only, never performance")

emit("\n  Macro F1 — pooled out-of-fold (fold mean +/- std):")
emit(f"  {'Feature Set':<40} {'LogReg':<17} {'RF':<17} {'HistGBM':<17}")
for fs in FEATURE_SETS:
    cells = []
    for mn in ["LogReg", "RF", "HistGBM"]:
        r = summary[(summary.feature_set == fs) & (summary.model == mn)].iloc[0]
        cells.append(f"{r.macro_f1_oof:.3f} (±{r.macro_f1_fold_std:.3f})")
    emit(f"  {fs:<40} {cells[0]:<17} {cells[1]:<17} {cells[2]:<17}")

bf = summary.sort_values("macro_f1_oof", ascending=False).iloc[0]
bf_folds = pd.DataFrame(per_fold_rows)
bf_std = bf_folds[(bf_folds.feature_set == bf.feature_set)
                  & (bf_folds.model == bf.model)]["macro_f1"].std()
emit("\n  Four-fault mean recall — contextual comparison with Lazzaretti et al.")
emit("   (2020), not directly equivalent:")
emit(f"    Best configuration: {bf.four_fault_mean_recall:.3f} "
     f"({bf.feature_set} / {bf.model})")
emit("    Reference: Lazzaretti et al. (2020) ANN 95.44% average four-fault")
emit("      class accuracy | SC 99.35% | Deg 93.18% | OC 100% | Shad 89.26%")
emit("      NOTE: they trained on simulated data and tested a four-fault")
emit("      classifier on real data; ours is a five-class model trained and")
emit("      evaluated on real data with group-disjoint folds. Different")
emit("      protocols — do not subtract or rank these numbers.")

emit("\n  Capacity sensitivity (POWER_PHYSICS only):")
emit(f"    5.00 kW: Macro F1 {cap_rows[0]['macro_f1_oof']:.3f} "
     f"± {cap_rows[0]['macro_f1_fold_std']:.3f}")
emit(f"    5.28 kW: Macro F1 {cap_rows[1]['macro_f1_oof']:.3f} "
     f"± {cap_rows[1]['macro_f1_fold_std']:.3f}")
emit(f"    -> {'insensitive' if cap_diff < 0.01 else 'sensitive'} to capacity ambiguity")

emit("\n  VOLTAGE vs CURRENT:")
for fs in ["Electrical (full)", "No current sensors", "No voltage sensors"]:
    r = summary[summary.feature_set == fs].sort_values("macro_f1_oof",
                                                       ascending=False).iloc[0]
    base = summary[summary.feature_set == "Electrical (full)"] \
        .sort_values("macro_f1_oof", ascending=False).iloc[0]["macro_f1_oof"]
    drop = base - r.macro_f1_oof
    tail = "" if fs == "Electrical (full)" else f"  (drop: {drop:.3f})"
    emit(f"    {fs:<22} Macro F1: {r.macro_f1_oof:.3f}{tail}")
nc = summary[summary.feature_set == "No current sensors"]["macro_f1_oof"].max()
nv = summary[summary.feature_set == "No voltage sensors"]["macro_f1_oof"].max()
if abs(nc - nv) < 0.02:
    verdict = "is inconclusive"
elif nv > nc:
    verdict = "does not hold — removing voltage hurt LESS than removing current"
else:
    verdict = "holds — removing voltage hurt MORE than removing current"
emit(f"    -> Historical claim was that voltage sensors matter more than")
emit(f"       current. Under leakage-safe grouped validation this claim")
emit(f"       {verdict}.")

emit("\n  COMPARISON WITH HISTORICAL RESULT:")
emit("    Historical Macro F1 (random row split): 0.9975")
emit(f"    Grouped Macro F1 (best configuration):                    "
     f"{bf.macro_f1_oof:.3f} ± {bf_std:.3f}")
emit("    -> The historical result was inflated by temporal leakage: random")
emit("       row splitting on 1 Hz data placed adjacent seconds of the same")
emit("       physical fault event in both train and test.")

pf_df = pd.DataFrame(per_fold_rows)
worst = pf_df.groupby(["feature_set", "model"])["macro_f1"].std().sort_values()
emit("\n  Fold-to-fold variance:")
emit(f"    Highest std across folds: {worst.iloc[-1]:.3f} "
     f"(feature set: {worst.index[-1][0]}, model: {worst.index[-1][1]})")
emit(f"    Lowest  std across folds: {worst.iloc[0]:.3f} "
     f"({worst.index[0][0]}, {worst.index[0][1]})")
emit("    Per-fold class support is in fold_assignments.csv; folds 3 and 5")
emit("    carry the long degradation events, and every fold's SC/OC support")
emit("    rests on exactly 2 inductions (~1,200 samples).")
emit("=" * 78)

(OUT_DIR / "run_report.txt").write_text("\n".join(_report) + "\n")
print(f"\nSaved outputs to: {OUT_DIR}")

with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"pv_fault_classifier.py | "
            f"input: {DATA_CSV.name} {len(df):,} rows | "
            f"output: {OUT_DIR.name}/ 8 files | "
            f"notes: group-disjoint 5-fold CV over 16 groups, 18 configs; "
            f"best {bf.feature_set}/{bf.model} MacroF1 {bf.macro_f1_oof:.4f}; "
            f"historical 0.9975 was leakage-inflated; f_nv never modified\n")
print(f"Run log appended: {RUN_LOG}")
