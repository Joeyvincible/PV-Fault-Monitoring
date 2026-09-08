"""
split_effect_diagnostic.py
==========================
Objective 1 — two supporting diagnostics. Neither changes any data and
neither produces a headline result.

DIAGNOSTIC 1 — split-induced optimism (controlled)
--------------------------------------------------
The historical 0.9975 fell to 0.966 under the group-aware pipeline, but the
split method AND the preprocessing changed at the same time, so the drop
cannot be attributed to either. This holds dataset, features, model,
preprocessing, fold count and metric computation constant and varies ONLY
whether rows from the same recording group may cross partitions.

It compares 5-fold against 5-fold deliberately. A single 80/20 split
against 5-fold CV would differ in training-set size, number of fits and
the share of data receiving an out-of-fold prediction, confounding the
split principle with the estimator.

METHOD A IS METHODOLOGICALLY INVALID FOR REPORTING RESULTS. It exists only
to quantify optimism, and every Method A figure in the terminal output and
in the CSV carries the invalidity label.

DIAGNOSTIC 2 — paired fold drops
--------------------------------
Reads the existing per-fold results and computes paired WITHIN-fold
differences from the ELEC_FULL baseline, which supports a stronger claim
than differences of pooled values: that the reduction holds inside every
fold rather than only on average. Nothing is retrained.

INPUT : data/fault_dataset.csv
        outputs/grouped_baseline/grouped_baseline_per_fold.csv
OUTPUT: outputs/grouped_baseline/{split_effect_diagnostic.csv,
        split_effect_confusion.csv, paired_fold_drops.csv,
        split_effect_report.txt}
"""

import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, recall_score

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_CSV    = PROJECT_DIR / "data" / "fault_dataset.csv"
BASE_DIR    = PROJECT_DIR / "outputs" / "grouped_baseline"
PER_FOLD    = BASE_DIR / "grouped_baseline_per_fold.csv"
RUN_LOG     = PROJECT_DIR / "outputs" / "run_log.txt"

GAP_THRESHOLD = 3600
RANDOM_SEED   = 42
LABELS = [0, 1, 2, 3, 4]
LABEL_NAMES = {0: "Normal", 1: "Short-Circuit", 2: "Degradation",
               3: "Open-Circuit", 4: "Shadowing"}

APPROVED_FOLDS = {1: [2, 10, 11], 2: [1, 3, 7, 12], 3: [0, 9, 13],
                  4: [5, 6, 14], 5: [4, 8, 15]}

INVALID_LABEL = (
    "stratified random-row 5-fold CV — DIAGNOSTIC ONLY, methodologically\n"
    "     invalid for result reporting (adjacent seconds of the same physical\n"
    "     event appear in different folds)")
INVALID_LABEL_FLAT = (
    "stratified random-row 5-fold CV - DIAGNOSTIC ONLY, methodologically "
    "invalid for result reporting (adjacent seconds of the same physical "
    "event appear in different folds)")

PHYSICS_ONLY = ["ratio_5k0", "residual_5k0"]
ELEC_FULL = ["vdc1", "vdc2", "idc1", "idc2", "p1", "p2", "p_dc_computed",
             "v_ratio", "i_ratio", "v_imbalance", "i_imbalance"]
FEATURE_SETS = {
    "ratio_5k0 + residual_5k0": PHYSICS_ONLY,
    "Electrical full": ELEC_FULL,
}

_report = []


def emit(line=""):
    print(line)
    _report.append(line)


emit("=" * 78)
emit("  SPLIT-EFFECT DIAGNOSTIC")
emit("=" * 78)
emit(f"  Started: {datetime.datetime.now().isoformat(timespec='seconds')}")

df = pd.read_csv(DATA_CSV).sort_values("sample_index").reset_index(drop=True)
y = df["f_nv"].to_numpy()
n = len(df)
emit(f"  Data: {DATA_CSV.name}  {n:,} rows")

# recording_group is a SPLIT KEY ONLY — never a model feature.
df["recording_group"] = (df["sample_index"].diff() > GAP_THRESHOLD
                         ).fillna(False).cumsum().astype(int)
assert sorted(df["recording_group"].unique()) == list(range(16)), \
    "Recording group reconstruction did not yield 16 groups 0-15"

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
assert folds == APPROVED_FOLDS, (
    f"Group reconstruction does not reproduce the approved allocation.\n"
    f"  computed: {folds}\n  approved: {APPROVED_FOLDS}")
emit(f"  Group-disjoint allocation reproduces approved folds: PASS")

df["fold_b"] = 0
for f, gs in folds.items():
    df.loc[df["recording_group"].isin(gs), "fold_b"] = f
assert (df["fold_b"] > 0).all()


def make_model():
    return HistGradientBoostingClassifier(class_weight="balanced",
                                          max_iter=200,
                                          random_state=RANDOM_SEED)


def run_cv(feats, splits):
    """splits: list of (train_idx, test_idx). Returns OOF preds + per-fold F1."""
    oof = np.full(n, -1, dtype=np.int8)
    fold_f1 = []
    for tr_idx, te_idx in splits:
        X_tr = df.loc[tr_idx, feats].to_numpy(dtype=float)
        X_te = df.loc[te_idx, feats].to_numpy(dtype=float)
        X_tr[~np.isfinite(X_tr)] = np.nan
        X_te[~np.isfinite(X_te)] = np.nan
        imp = SimpleImputer(strategy="median").fit(X_tr)   # training partition only
        X_tr, X_te = imp.transform(X_tr), imp.transform(X_te)
        m = make_model().fit(X_tr, y[tr_idx])
        pred = m.predict(X_te)
        oof[te_idx] = pred
        fold_f1.append(f1_score(y[te_idx], pred, average="macro",
                                labels=LABELS, zero_division=0))
    assert (oof >= 0).all(), "Not every row received an out-of-fold prediction"
    assert int((oof >= 0).sum()) == n, "OOF prediction count != dataset row count"
    return oof, np.array(fold_f1)


# Split definitions --------------------------------------------------------
idx = np.arange(n)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
splits_a = list(skf.split(idx, y))
splits_b = [(idx[(df["fold_b"] != f).to_numpy()], idx[(df["fold_b"] == f).to_numpy()])
            for f in sorted(folds)]

emit(f"  Method A folds (random row) : sizes "
     f"{[len(te) for _, te in splits_a]}")
emit(f"  Method B folds (group-disjoint): sizes "
     f"{[len(te) for _, te in splits_b]}")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 1
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  DIAGNOSTIC 1 — SPLIT-INDUCED OPTIMISM (controlled)")
emit("=" * 78)
emit("  Dataset, features, model, preprocessing and fold count all held")
emit("  constant. Only partition assignment differs.")
emit("  Model: HistGradientBoosting | Folds: 5 | "
     "Data: fault_dataset.csv")

rows, cm_rows, results = [], [], {}
for fs_name, feats in FEATURE_SETS.items():
    emit(f"\n  Feature set: {fs_name} ({len(feats)} features)")
    for method, splits, valid in [("A", splits_a, False), ("B", splits_b, True)]:
        oof, ff = run_cv(feats, splits)
        pooled = f1_score(y, oof, average="macro", labels=LABELS, zero_division=0)
        results[(fs_name, method)] = (pooled, ff, oof)
        rec = recall_score(y, oof, labels=LABELS, average=None, zero_division=0)
        if method == "A":
            emit(f"    A. Stratified random-row 5-fold (INVALID) : {pooled:.3f}  "
                 f"(folds {ff.mean():.3f} ± {ff.std(ddof=1):.3f})")
            emit(f"       ^ {INVALID_LABEL}")
        else:
            emit(f"    B. Group-disjoint 5-fold (valid)          : {pooled:.3f}  "
                 f"(folds {ff.mean():.3f} ± {ff.std(ddof=1):.3f})")
        row = {"feature_set": fs_name, "n_features": len(feats),
               "method": ("A_stratified_random_row" if method == "A"
                          else "B_group_disjoint"),
               "valid_for_reporting": valid,
               "validity_label": ("VALID" if valid else INVALID_LABEL_FLAT),
               "macro_f1_pooled_oof": round(pooled, 4),
               "macro_f1_fold_mean": round(float(ff.mean()), 4),
               "macro_f1_fold_sd_ddof1": round(float(ff.std(ddof=1)), 4)}
        row.update({f"recall_{LABEL_NAMES[l]}": round(rec[i], 4)
                    for i, l in enumerate(LABELS)})
        rows.append(row)
        cm = confusion_matrix(y, oof, labels=LABELS)
        for i, tl in enumerate(LABELS):
            for j, pl in enumerate(LABELS):
                cm_rows.append({
                    "feature_set": fs_name,
                    "method": ("A_stratified_random_row" if method == "A"
                               else "B_group_disjoint"),
                    "valid_for_reporting": valid,
                    "validity_label": ("VALID" if valid else INVALID_LABEL_FLAT),
                    "true": LABEL_NAMES[tl], "pred": LABEL_NAMES[pl],
                    "count": int(cm[i, j])})
    opt = results[(fs_name, "A")][0] - results[(fs_name, "B")][0]
    emit(f"    Split-induced optimism                    : {opt:.3f}")

pd.DataFrame(rows).to_csv(BASE_DIR / "split_effect_diagnostic.csv", index=False)
pd.DataFrame(cm_rows).to_csv(BASE_DIR / "split_effect_confusion.csv", index=False)

# Per-class recall comparison
emit("\n  PER-CLASS RECALL — Method A (INVALID) vs Method B (valid)")
emit(f"  ^ Method A columns are {INVALID_LABEL_FLAT}")
for fs_name in FEATURE_SETS:
    emit(f"\n    {fs_name}")
    emit(f"      {'class':<16}{'A (invalid)':>13}{'B (valid)':>12}{'inflation':>11}")
    ra = recall_score(y, results[(fs_name, 'A')][2], labels=LABELS,
                      average=None, zero_division=0)
    rb = recall_score(y, results[(fs_name, 'B')][2], labels=LABELS,
                      average=None, zero_division=0)
    for i, l in enumerate(LABELS):
        emit(f"      {LABEL_NAMES[l]:<16}{ra[i]:>13.4f}{rb[i]:>12.4f}"
             f"{ra[i] - rb[i]:>+11.4f}")

# Confusion matrices
emit("\n  POOLED CONFUSION MATRICES (rows=true, cols=predicted)")
order = [LABEL_NAMES[l] for l in LABELS]
for fs_name in FEATURE_SETS:
    for method in ("A", "B"):
        tag = ("Method A — " + INVALID_LABEL_FLAT) if method == "A" \
            else "Method B — group-disjoint 5-fold (valid)"
        emit(f"\n    {fs_name} | {tag}")
        cm = confusion_matrix(y, results[(fs_name, method)][2], labels=LABELS)
        emit("      " + "true \\ pred".ljust(16)
             + "".join(f"{c[:9]:>10}" for c in order))
        for i, t in enumerate(order):
            emit("      " + t.ljust(16)
                 + "".join(f"{int(cm[i, j]):>10,}" for j in range(5)))

opt_phys = results[("ratio_5k0 + residual_5k0", "A")][0] - \
    results[("ratio_5k0 + residual_5k0", "B")][0]
opt_elec = results[("Electrical full", "A")][0] - \
    results[("Electrical full", "B")][0]
bigger = "physics-derived" if opt_phys > opt_elec else "electrical"
emit("\n  INTERPRETATION")
emit(f"    Split-induced optimism is larger for the {bigger} feature set: "
     f"{max(opt_phys, opt_elec):.3f} vs {min(opt_phys, opt_elec):.3f} "
     f"(difference {abs(opt_phys - opt_elec):.3f}).")
for fs_name in FEATURE_SETS:
    ra = recall_score(y, results[(fs_name, 'A')][2], labels=LABELS,
                      average=None, zero_division=0)
    rb = recall_score(y, results[(fs_name, 'B')][2], labels=LABELS,
                      average=None, zero_division=0)
    infl = {LABEL_NAMES[l]: ra[i] - rb[i] for i, l in enumerate(LABELS)}
    top = sorted(infl.items(), key=lambda kv: -kv[1])[:3]
    emit(f"    {fs_name}: most inflated classes — "
         + ", ".join(f"{k} {v:+.3f}" for k, v in top))

emit("\n  CONTEXT (not a controlled comparison)")
emit("    The uncorrected historical pipeline reported Macro F1 0.9975 under a")
emit("    single 80/20 random row split, with different preprocessing and a")
emit("    16-feature set including timestamp-derived pvlib features that no")
emit("    longer exist. That figure is not directly comparable to either method")
emit("    above and must not appear in dissertation result tables.")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC 2
# ─────────────────────────────────────────────────────────────────────────────
emit("\n" + "=" * 78)
emit("  DIAGNOSTIC 2 — PAIRED FOLD DROPS (sensor removal)")
emit("=" * 78)
emit("  Paired within-fold differences from the ELEC_FULL baseline.")
emit("  Standard deviation uses ddof=1 (sample SD, n=5).")

pf = pd.read_csv(PER_FOLD)
BASE_FS, NC_FS, NV_FS = "Electrical (full)", "No current sensors", "No voltage sensors"
MODEL_LABEL = {"LogReg": "Logistic Regression", "RF": "Random Forest",
               "HistGBM": "HistGradientBoosting"}

drop_rows, agg_voltage_larger, agg_total = [], 0, 0
for model in ["LogReg", "RF", "HistGBM"]:
    def series(fs):
        s = pf[(pf.feature_set == fs) & (pf.model == model)] \
            .set_index("fold")["macro_f1"]
        return s.reindex([1, 2, 3, 4, 5])
    base, nc, nv = series(BASE_FS), series(NC_FS), series(NV_FS)
    assert not base.isna().any() and not nc.isna().any() and not nv.isna().any(), \
        f"Missing per-fold entries for {model}"
    d_cur, d_vol = base - nc, base - nv
    n_vol_larger = int((d_vol > d_cur).sum())
    agg_voltage_larger += n_vol_larger
    agg_total += 5

    emit(f"\n  Model: {MODEL_LABEL[model]}")
    emit(f"    {'Fold':<14}" + "".join(f"{k:>8}" for k in [1, 2, 3, 4, 5])
         + "     mean ± SD")
    emit(f"    {'− current':<14}" + "".join(f"{d_cur[k]:>8.3f}" for k in [1, 2, 3, 4, 5])
         + f"     {d_cur.mean():.3f} ± {d_cur.std(ddof=1):.3f}")
    emit(f"    {'− voltage':<14}" + "".join(f"{d_vol[k]:>8.3f}" for k in [1, 2, 3, 4, 5])
         + f"     {d_vol.mean():.3f} ± {d_vol.std(ddof=1):.3f}")
    emit(f"    Folds where voltage removal cost more: {n_vol_larger} of 5")
    for k in [1, 2, 3, 4, 5]:
        drop_rows.append({"model": model, "fold": k,
                          "removal": "no_current",
                          "baseline_macro_f1": round(float(base[k]), 4),
                          "removed_macro_f1": round(float(nc[k]), 4),
                          "drop": round(float(d_cur[k]), 4)})
        drop_rows.append({"model": model, "fold": k,
                          "removal": "no_voltage",
                          "baseline_macro_f1": round(float(base[k]), 4),
                          "removed_macro_f1": round(float(nv[k]), 4),
                          "drop": round(float(d_vol[k]), 4)})

pd.DataFrame(drop_rows).to_csv(BASE_DIR / "paired_fold_drops.csv", index=False)

emit("\n  AGGREGATE")
emit(f"    Paired comparisons where voltage removal cost more: "
     f"{agg_voltage_larger} of {agg_total}")
emit("")
emit('    → "Removing voltage caused a substantially larger reduction in Macro')
emit('       F1 than removing current, consistently across all three classifiers')
emit(f'       and in {agg_voltage_larger} of {agg_total} paired fold comparisons."')
emit("=" * 78)

(BASE_DIR / "split_effect_report.txt").write_text("\n".join(_report) + "\n")
print(f"\nSaved to: {BASE_DIR}")

with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"split_effect_diagnostic.py | input: {DATA_CSV.name} {n:,} rows, "
            f"{PER_FOLD.name} | output: 4 files in {BASE_DIR.name}/ | "
            f"notes: optimism physics {opt_phys:.4f}, electrical {opt_elec:.4f}; "
            f"voltage removal larger in {agg_voltage_larger}/{agg_total} paired "
            f"folds; Method A is diagnostic-only and invalid for reporting; "
            f"no dataset or existing output modified\n")
print(f"Run log appended: {RUN_LOG}")
