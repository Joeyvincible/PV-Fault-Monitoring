"""
plot_objective1_results.py
==========================
Objective 1 — figure set for the group-disjoint results.

Reads existing grouped-validation CSVs and produces six figures. No modelling, no
retraining, no dataset access.

The historical figures and CSVs in outputs/ (ablation_summary*.csv,
cost_ablation_plot.png, per_class_f1_heatmap.png, ...) come from the
uncorrected pipeline. They are neither read nor overwritten here; they are
retained only as the historical record.

STRUCTURE
  constants -> load_grouped_sources -> validate_sources ->
  validate_terminology -> prepare_plot_tables -> plot_fig1..6 ->
  write_captions -> write_source_manifest

Each permitted CSV is read EXACTLY ONCE. Figure functions receive prepared
DataFrames and never reopen a source file, so two figures cannot silently
disagree by reading different files or applying different filters.
"""

import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EXT_DIR     = PROJECT_DIR / "outputs" / "extended_ablation"
BASE_DIR    = PROJECT_DIR / "outputs" / "grouped_baseline"
OUT_DIR     = PROJECT_DIR / "outputs" / "objective1_figures"
RUN_LOG     = PROJECT_DIR / "outputs" / "run_log.txt"

PERMITTED = {
    "summary":     EXT_DIR / "extended_ablation_summary.csv",
    "per_fold":    EXT_DIR / "extended_ablation_per_fold.csv",
    "per_class":   EXT_DIR / "extended_ablation_per_class.csv",
    "confusion":   EXT_DIR / "extended_confusion_matrices.csv",
    "degradation": EXT_DIR / "extended_degradation_breakdown.csv",
    "sensor":      EXT_DIR / "sensor_removal_by_model.csv",
    "split":       BASE_DIR / "split_effect_diagnostic.csv",
    "paired":      BASE_DIR / "paired_fold_drops.csv",
}
FORBIDDEN_INPUTS = ["ablation_summary.csv", "ablation_summary_v2.csv",
                    "ablation_per_class_detail.csv", "leakage_check.txt",
                    "ablation_summary_plot.png", "cost_ablation_plot.png",
                    "per_class_f1_heatmap.png", "confusion_matrix_best.png",
                    "feature_importance_ablation.png"]

MODELS = ["LogReg", "RF", "HistGBM", "MLP"]
MODEL_COLOURS = {"LogReg": "steelblue", "RF": "tomato",
                 "HistGBM": "seagreen", "MLP": "mediumpurple"}
CLASSES = ["Normal", "Short-Circuit", "Degradation", "Open-Circuit", "Shadowing"]

# CSV label -> figure display label
SECTION_1 = [
    ("Electrical (full)",                     "Electrical (full)"),
    ("Electrical + irradiance",               "Electrical\n+ irradiance"),
    ("Electrical + irradiance + module temp", "Electrical + irradiance\n+ module temp"),
    ("No current sensors",                    "No current sensors"),
    ("No voltage sensors",                    "No voltage sensors"),
]
SECTION_2 = [
    ("Computed power + approximate measured-POA physics",
     "Computed power +\napprox. measured-POA physics"),
    ("Computed total power only",             "Computed total power"),
    ("Computed power + measured irradiance",  "Computed power +\nmeasured irradiance"),
    ("Computed power + irradiance + module temp",
     "Computed power + irradiance\n+ module temp"),
    ("HK-availability proxy (not a validated transfer mapping)",
     "HK-availability proxy\n(preliminary)"),
]
ORDER = SECTION_1 + SECTION_2
CSV_LABELS = [c for c, _ in ORDER]
DISPLAY = dict(ORDER)

L_ELEC   = "Electrical (full)"
L_NO_CUR = "No current sensors"
L_NO_VOL = "No voltage sensors"
L_PHYS   = "Computed power + approximate measured-POA physics"
EXPECTED_BEST = ("Electrical + irradiance", "MLP")

SEC1_TITLE = "String-level electrical / sensor-availability configurations"
SEC2_TITLE = "Aggregate-signal / reduced-feature representations"

BANNED = ["cost ablation", "cheaper hardware", "hardware cost",
          "power meter only", "pvlib only", "sensor-free", "hk-equivalent",
          "model capacity", "increasing capacity"]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.labelsize": 11, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 9})

READ_LOG = []


# ─────────────────────────────────────────────────────────────────────────────
def load_grouped_sources():
    """Read each permitted CSV exactly once."""
    src = {}
    for key, path in PERMITTED.items():
        assert path.exists(), f"Missing permitted input: {path}"
        bad = [f for f in FORBIDDEN_INPUTS if path.name == f]
        assert not bad, f"Forbidden input file in read list: {bad}"
        src[key] = pd.read_csv(path)
        READ_LOG.append(str(path.relative_to(PROJECT_DIR)))
    return src


def validate_sources(src):
    need = {
        "summary": ["feature_set", "model", "macro_f1_oof", "macro_f1_fold_sd_ddof1"],
        "per_fold": ["feature_set", "model", "fold", "macro_f1"],
        "per_class": ["feature_set", "model", "class", "f1", "precision",
                      "recall", "support"],
        "confusion": ["feature_set", "model", "true", "pred", "count"],
        "degradation": ["feature_set", "model", "standard_deg_recall",
                        "anomalous_deg_recall"],
        "sensor": ["model", "drop_current", "drop_voltage"],
        "split": ["feature_set", "method", "macro_f1_pooled_oof",
                  "macro_f1_fold_sd_ddof1"],
        "paired": ["model", "fold", "removal", "drop"],
    }
    for k, cols in need.items():
        missing = [c for c in cols if c not in src[k].columns]
        assert not missing, f"{k}: missing columns {missing}"
    assert len(src["summary"]) == 40, "summary must hold 40 configurations"
    for lbl in CSV_LABELS:
        assert lbl in set(src["summary"].feature_set), f"absent feature set: {lbl}"
    return True


def validate_terminology(strings):
    """No banned phrase may appear in any figure or caption string."""
    violations = []
    for s in strings:
        low = str(s).lower()
        for b in BANNED:
            if b in low:
                violations.append((b, s))
    assert not violations, f"Banned terminology in figure text: {violations}"
    return True


def prepare_plot_tables(src):
    """Derive every plotting frame up front, from the single read."""
    t = {}
    s = src["summary"]
    t["fig1"] = (s[s.feature_set.isin(CSV_LABELS)]
                 .pivot(index="feature_set", columns="model",
                        values="macro_f1_oof").reindex(CSV_LABELS)[MODELS])
    t["fig1_sd"] = (s[s.feature_set.isin(CSV_LABELS)]
                    .pivot(index="feature_set", columns="model",
                           values="macro_f1_fold_sd_ddof1")
                    .reindex(CSV_LABELS)[MODELS])

    # Figure 2: paired within-fold drops. MLP is absent from paired_fold_drops
    # (that file predates MLP), so compute MLP from the per-fold table.
    pf = src["per_fold"]
    paired = src["paired"][["model", "fold", "removal", "drop"]].copy()
    mlp_rows = []
    for f in sorted(pf.fold.unique()):
        def v(fs):
            r = pf[(pf.feature_set == fs) & (pf.model == "MLP") & (pf.fold == f)]
            assert len(r) == 1, f"MLP/{fs}/fold {f} not uniquely present"
            return float(r.iloc[0].macro_f1)
        base = v(L_ELEC)
        mlp_rows += [{"model": "MLP", "fold": int(f), "removal": "no_current",
                      "drop": base - v(L_NO_CUR)},
                     {"model": "MLP", "fold": int(f), "removal": "no_voltage",
                      "drop": base - v(L_NO_VOL)}]
    paired = pd.concat([paired, pd.DataFrame(mlp_rows)], ignore_index=True)
    t["paired"] = paired

    wide = paired.pivot_table(index=["model", "fold"], columns="removal",
                              values="drop").reset_index()
    t["paired_wide"] = wide
    t["volt_larger"] = int((wide["no_voltage"] > wide["no_current"]).sum())
    t["paired_total"] = int(len(wide))
    t["fig2"] = (paired.groupby(["model", "removal"])["drop"]
                 .agg(["mean", lambda x: x.std(ddof=1)])
                 .rename(columns={"<lambda_0>": "sd"}).reset_index())

    t["fig3"] = src["split"]
    t["fig4_cm"] = src["confusion"]
    t["fig5"] = src["per_class"]
    t["fig6"] = src["degradation"]
    t["summary"] = s
    return t


# ─────────────────────────────────────────────────────────────────────────────
def save(fig, stem):
    for ext, kw in ((".png", {"dpi": 300}), (".pdf", {})):
        fig.savefig(OUT_DIR / f"{stem}{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def plot_fig1(t):
    f1, sd = t["fig1"], t["fig1_sd"]
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(CSV_LABELS))
    w = 0.2
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1.5) * w, f1[m].to_numpy(), w,
               yerr=sd[m].to_numpy(), capsize=2.5, label=m,
               color=MODEL_COLOURS[m], alpha=0.9,
               error_kw={"elinewidth": 0.8, "capthick": 0.8})
    ax.axvline(4.5, color="black", linestyle="--", linewidth=1.2)
    ax.text(2.0, 1.015, SEC1_TITLE, ha="center", fontsize=10, style="italic")
    ax.text(7.5, 1.015, SEC2_TITLE, ha="center", fontsize=10, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[c] for c in CSV_LABELS], rotation=35,
                       ha="right", fontsize=9)
    ax.set_ylabel("Pooled out-of-fold Macro F1")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(FIG1_TITLE)
    # Outside the axes: every in-axes region is occupied by bars.
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=True)
    ax.grid(alpha=0.3, axis="y")
    ax.text(0.5, -0.42, FIG1_FOOTNOTE, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.5, wrap=True,
            bbox=dict(facecolor="whitesmoke", edgecolor="grey", alpha=0.9))
    save(fig, "fig1_consolidated_results")


def plot_fig2(t):
    g = t["fig2"]
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(MODELS))
    for removal, colour, lbl in [
            ("no_current", "darkorange", "Current-removal penalty (ELEC_FULL − NO_CURRENT)"),
            ("no_voltage", "crimson", "Voltage-removal penalty (ELEC_FULL − NO_VOLTAGE)")]:
        sub = g[g.removal == removal].set_index("model").reindex(MODELS)
        ax.errorbar(x, sub["mean"].to_numpy(), yerr=sub["sd"].to_numpy(),
                    marker="o", capsize=4, linewidth=1.8, color=colour, label=lbl)
        for xi, (mu, sdv) in enumerate(zip(sub["mean"].to_numpy(),
                                           sub["sd"].to_numpy())):
            ax.annotate(f"{mu:.3f}", (xi, mu), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9, color=colour)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_xlabel("Model family")
    ax.set_ylabel("Macro F1 penalty (paired within-fold mean)")
    ax.set_ylim(0, 0.5)
    ax.set_title(FIG2_TITLE)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    ax.text(0.5, -0.20, FIG2_ANNOT, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.5,
            bbox=dict(facecolor="whitesmoke", edgecolor="grey", alpha=0.9))
    save(fig, "fig2_sensor_removal_by_model")


def plot_fig3(t):
    d = t["fig3"]
    sets = ["ratio_5k0 + residual_5k0", "Electrical full"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for ax, fs in zip(axes, sets):
        sub = d[d.feature_set == fs]
        a = sub[sub.method == "A_stratified_random_row"].iloc[0]
        b = sub[sub.method == "B_group_disjoint"].iloc[0]
        vals = [a.macro_f1_pooled_oof, b.macro_f1_pooled_oof]
        sds = [a.macro_f1_fold_sd_ddof1, b.macro_f1_fold_sd_ddof1]
        bars = ax.bar([0, 1], vals, yerr=sds, capsize=5, width=0.55,
                      color=["0.72", "steelblue"],
                      hatch=["//", ""], edgecolor="black", linewidth=0.8)
        for xi, (v, s_) in enumerate(zip(vals, sds)):
            ax.text(xi, v + s_ + 0.03, f"{v:.3f} ± {s_:.3f}", ha="center",
                    fontsize=10, fontweight="bold")
        # Opaque backing: the label sits over the hatched bar.
        ax.text(0, 0.06, "DIAGNOSTIC ONLY —\nmethodologically invalid",
                ha="center", va="center", fontsize=8.5, color="darkred",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="darkred",
                          alpha=0.95, boxstyle="round,pad=0.35"))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stratified\nrandom-row 5-fold",
                            "Group-disjoint\n5-fold"], fontsize=9)
        ax.set_title(f"{fs} ({int(sub.iloc[0].n_features)} features)", fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Pooled out-of-fold Macro F1")
    fig.suptitle(FIG3_TITLE, fontsize=12, y=0.99)
    fig.text(0.5, -0.04, FIG3_ANNOT, ha="center", va="top", fontsize=8.5,
             bbox=dict(facecolor="whitesmoke", edgecolor="grey", alpha=0.9))
    save(fig, "fig3_split_induced_optimism")


def plot_fig4(t, best_fs, best_model):
    cm = t["fig4_cm"]
    sub = cm[(cm.feature_set == best_fs) & (cm.model == best_model)]
    m = sub.pivot(index="true", columns="pred", values="count").reindex(
        index=CLASSES, columns=CLASSES).to_numpy(dtype=float)
    norm = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(FIG4_TITLE)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{norm[i, j]:.3f}\n({int(m[i, j]):,})",
                    ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised (recall)")
    save(fig, "fig4_confusion_matrix_best")


def plot_fig5(t):
    pc = t["fig5"]
    cols = [L_ELEC, L_NO_CUR, L_NO_VOL, L_PHYS]
    mat = np.zeros((5, 4))
    for j, fs in enumerate(cols):
        sub = pc[(pc.feature_set == fs) & (pc.model == "MLP")]
        for i, c in enumerate(CLASSES):
            r = sub[sub["class"] == c]
            assert len(r) == 1, f"missing per-class row: {fs}/MLP/{c}"
            mat[i, j] = float(r.iloc[0].f1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels([DISPLAY[c] for c in cols], rotation=25, ha="right",
                       fontsize=9)
    ax.set_yticks(range(5)); ax.set_yticklabels(CLASSES)
    ax.set_title(FIG5_TITLE)
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=10,
                    color="black" if 0.35 < mat[i, j] < 0.85 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Per-class F1")
    save(fig, "fig5_per_class_f1")


def plot_fig6(t, best_per_model):
    deg = t["fig6"]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    x = np.arange(len(MODELS)); w = 0.35
    std_v, ano_v = [], []
    for m in MODELS:
        fs = best_per_model[m]
        r = deg[(deg.model == m) & (deg.feature_set == fs)]
        assert len(r) == 1, f"degradation row missing for {m}/{fs}"
        std_v.append(float(r.iloc[0].standard_deg_recall))
        ano_v.append(float(r.iloc[0].anomalous_deg_recall))
    b1 = ax.bar(x - w/2, std_v, w, label="Standard ~600-sample scheduled inductions",
                color="steelblue", alpha=0.9)
    b2 = ax.bar(x + w/2, ano_v, w,
                label="Two long anomalous events (1,779 and 2,695 samples)",
                color="darkorange", alpha=0.9)
    for bars, vals in ((b1, std_v), (b2, ano_v)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_xlabel("Model family")
    ax.set_ylabel("Degradation recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(FIG6_TITLE)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    ax.text(0.5, -0.17, FIG6_ANNOT, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.5,
            bbox=dict(facecolor="whitesmoke", edgecolor="grey", alpha=0.9))
    save(fig, "fig6_degradation_event_character")


# ─────────────────────────────────────────────────────────────────────────────
PROTOCOL = ("Protocol: group-disjoint 5-fold cross-validation over all 16 "
            "reconstructed recording days; every row receives exactly one "
            "out-of-fold prediction. Uncertainty is the fold standard "
            "deviation (ddof=1, n=5).")
MLP_CAVEAT = ("MLP used balanced training-fold downsampling because "
              "scikit-learn 1.5.2 MLPClassifier supports neither class_weight "
              "nor sample_weight, whereas LogReg, RF and HistGBM used "
              "class weighting. Cross-model differences therefore combine "
              "classifier family with imbalance strategy and are not a "
              "controlled comparison of architecture.")

FIG1_TITLE = ("Objective 1 — group-disjoint results: 5-fold CV, "
              "all 16 recording days")
FIG1_FOOTNOTE = ("Error bars show fold standard deviation (ddof=1, n=5). "
                 "Aggregate-signal sets require the same string sensors as the "
                 "full electrical set — p_dc_computed = vdc1×idc1 + vdc2×idc2. "
                 "NO_CURRENT and NO_VOLTAGE are the genuine hardware-removal "
                 "ablations.")
FIG2_TITLE = "Sensor-removal penalty across model families"
FIG3_TITLE = ("Split-induced optimism: same data, same features, same model, "
              "same fold count")
FIG5_TITLE = ("Per-class F1 under sensor removal and aggregate representation "
              "— MLP")
FIG6_TITLE = ("Degradation recall by event character — generalisation to "
              "non-standard fault duration")
FIG6_ANNOT = ("The two long anomalous events occur in recording groups 8 and 9 "
              "and do not follow the paper's 10-minute induction schedule.\n"
              "All models recover standard inductions better.")


def write_captions(t, best_fs, best_model, volt_larger, total, sd_lo, sd_hi,
                   rr_lo, rr_hi, best_per_model):
    caps = []
    caps.append("FIGURE CAPTIONS — Objective 1 group-disjoint results\n" + "=" * 70)
    caps.append(
        f"\nFigure 1. Consolidated Objective 1 results.\n"
        f"Pooled out-of-fold five-class Macro F1 for ten feature configurations "
        f"across four model families, split into {SEC1_TITLE.lower()} (left) and "
        f"{SEC2_TITLE.lower()} (right). {PROTOCOL} Error bars show fold SD. "
        f"Every configuration in the right-hand section still requires both "
        f"string voltage and both string current sensors, because "
        f"p_dc_computed = vdc1×idc1 + vdc2×idc2; NO_CURRENT and NO_VOLTAGE are "
        f"the only genuine hardware-removal ablations. {MLP_CAVEAT}")
    caps.append(
        f"\nFigure 2. Sensor-removal penalty across model families.\n"
        f"Paired within-fold Macro F1 reduction when current sensing or voltage "
        f"sensing is removed from the full string-level electrical set, for each "
        f"model family. Points are means of the five paired within-fold "
        f"differences; error bars are their SD (ddof=1, n=5). {PROTOCOL} "
        f"Voltage removal cost more than current removal in {volt_larger} of "
        f"{total} paired model × fold comparisons. {MLP_CAVEAT}")
    caps.append(
        f"\nFigure 3. Split-induced optimism.\n"
        f"The same data, features, model and fold count evaluated under two "
        f"partitioning rules. The stratified random-row arm is a DIAGNOSTIC "
        f"ONLY: it is methodologically invalid for reporting results, because "
        f"adjacent seconds of the same physical fault event appear in different "
        f"folds, and it must not be read as a result. Random-row folding raises "
        f"the score and collapses fold SD from ±{sd_lo:.3f}–{sd_hi:.3f} to "
        f"±{rr_lo:.3f}–{rr_hi:.3f}, manufacturing an appearance of precision. "
        f"Uncertainty is fold SD (ddof=1, n=5).")
    caps.append(
        f"\nFigure 4. Pooled out-of-fold confusion matrix, best configuration "
        f"({best_fs}, {best_model}).\n"
        f"Row-normalised (recall orientation); each cell shows the normalised "
        f"value and the raw count. Pooled across all five group-disjoint folds, "
        f"so every row of the dataset is counted exactly once. {PROTOCOL} "
        f"Pooled Macro F1 for this configuration is "
        f"{float(t['summary'].query('feature_set == @best_fs and model == @best_model').iloc[0].macro_f1_oof):.3f} "
        f"± {float(t['summary'].query('feature_set == @best_fs and model == @best_model').iloc[0].macro_f1_fold_sd_ddof1):.3f}.")
    caps.append(
        f"\nFigure 5. Per-class F1 under sensor removal and aggregate "
        f"representation, MLP.\n"
        f"All four columns use MLP, holding classifier family and imbalance "
        f"strategy constant so differences reflect feature availability alone. "
        f"{PROTOCOL} Pooled out-of-fold per-class F1; fold-level spread for each "
        f"configuration is given in Figure 1.")
    caps.append(
        f"\nFigure 6. Degradation recall by event character.\n"
        f"Recall for degradation-labelled samples, split between the standard "
        f"~600-sample scheduled inductions and the two long anomalous events "
        f"(1,779 and 2,695 samples, recording groups 8 and 9). Each model uses "
        f"its best configuration ("
        f"{'; '.join(f'{m}: {best_per_model[m]}' for m in MODELS)}). {PROTOCOL} "
        f"The event-character split is a diagnostic breakdown: no sample was "
        f"removed or relabelled.")
    (OUT_DIR / "figure_captions.txt").write_text("\n".join(caps) + "\n")
    return caps


def write_source_manifest():
    lines = ["FIGURE SOURCE MANIFEST", "=" * 70,
             "\nFiles read (each exactly once):"]
    lines += [f"  {p}" for p in READ_LOG]
    lines += ["\nFigure -> source CSV:",
              "  fig1_consolidated_results        <- extended_ablation_summary.csv",
              "  fig2_sensor_removal_by_model     <- paired_fold_drops.csv (LogReg/RF/HistGBM)",
              "                                      + extended_ablation_per_fold.csv (MLP, computed)",
              "  fig3_split_induced_optimism      <- split_effect_diagnostic.csv",
              "  fig4_confusion_matrix_best       <- extended_confusion_matrices.csv",
              "                                      (best config from extended_ablation_summary.csv)",
              "  fig5_per_class_f1                <- extended_ablation_per_class.csv",
              "  fig6_degradation_event_character <- extended_degradation_breakdown.csv",
              "                                      (best config per model from summary)",
              "\nNot read: any historical uncorrected CSV or figure in the outputs/ root.",
              "  sensor_removal_by_model.csv was loaded and schema-validated but",
              "  Figure 2 uses the paired within-fold values, which carry the",
              "  fold-level spread that the pooled file does not."]
    (OUT_DIR / "figure_source_manifest.txt").write_text("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def main(dry_run=False):
    print("=" * 78)
    print("  OBJECTIVE 1 — GROUP-DISJOINT RESULTS FIGURES"
          + ("  [DRY RUN — no figure written]" if dry_run else ""))
    print("=" * 78)
    src = load_grouped_sources()
    print("\nFiles read (each exactly once):")
    for p in READ_LOG:
        print(f"  {p}")
    # Exact basename comparison. A suffix test would wrongly flag
    # "extended_ablation_summary.csv" for ending in "ablation_summary.csv".
    bad = [p for p in READ_LOG if Path(p).name in FORBIDDEN_INPUTS]
    assert not bad, f"Forbidden input read: {bad}"
    # Every read must also come from one of the two permitted directories,
    # never the historical outputs/ root.
    allowed_parents = {EXT_DIR.name, BASE_DIR.name}
    stray = [p for p in READ_LOG if Path(p).parent.name not in allowed_parents]
    assert not stray, f"Read outside the permitted directories: {stray}"
    print(f"  Forbidden-input assertion: PASS "
          f"({len(FORBIDDEN_INPUTS)} banned basenames checked, exact match)")
    print(f"  All reads confined to {sorted(allowed_parents)}: PASS")

    validate_sources(src)
    print("  Schema validation: PASS")

    t = prepare_plot_tables(src)

    # Figure 4 best configuration — derived, then asserted.
    s = t["summary"]
    best = s.sort_values("macro_f1_oof", ascending=False).iloc[0]
    best_fs, best_model = best.feature_set, best.model
    print(f"\nDetected best configuration: {best_fs} / {best_model} "
          f"(Macro F1 {best.macro_f1_oof:.4f})")
    assert (best_fs, best_model) == EXPECTED_BEST, (
        f"Best configuration {best_fs}/{best_model} != expected {EXPECTED_BEST}; "
        f"stopping rather than plotting.")
    print(f"  Assertion against expected {EXPECTED_BEST}: PASS")

    best_per_model = {m: s[s.model == m].sort_values("macro_f1_oof",
                                                     ascending=False).iloc[0].feature_set
                      for m in MODELS}
    print("  Best configuration per model (Figure 6):")
    for m in MODELS:
        print(f"    {m:<8} {best_per_model[m]}")

    vl, tot = t["volt_larger"], t["paired_total"]
    print(f"\nPaired within-fold comparisons where voltage removal cost more: "
          f"{vl} of {tot}")

    d = t["fig3"]
    gd = d[d.method == "B_group_disjoint"]["macro_f1_fold_sd_ddof1"]
    rr = d[d.method == "A_stratified_random_row"]["macro_f1_fold_sd_ddof1"]
    sd_lo, sd_hi = float(gd.min()), float(gd.max())
    rr_lo, rr_hi = float(rr.min()), float(rr.max())

    global FIG2_ANNOT, FIG3_ANNOT, FIG4_TITLE
    FIG2_ANNOT = ("The three nonlinear models show similar sensor-removal "
                  "penalties. Removing voltage produces a substantially larger\n"
                  f"Macro F1 reduction than removing current — in {vl} of {tot} "
                  "paired model × fold comparisons.")
    FIG3_ANNOT = ("Random-row folding inflates the score and collapses fold "
                  f"variance from ±{sd_lo:.3f}–{sd_hi:.3f} to "
                  f"±{rr_lo:.3f}–{rr_hi:.3f},\nmanufacturing an appearance of "
                  "precision.")
    FIG4_TITLE = ("Pooled out-of-fold confusion matrix — best configuration "
                  f"({best_fs}, {best_model})")

    figure_strings = [
        FIG1_TITLE, FIG1_FOOTNOTE, SEC1_TITLE, SEC2_TITLE,
        FIG2_TITLE, FIG2_ANNOT, FIG3_TITLE, FIG3_ANNOT, FIG4_TITLE,
        FIG5_TITLE, FIG6_TITLE, FIG6_ANNOT,
        "Pooled out-of-fold Macro F1", "Macro F1 penalty (paired within-fold mean)",
        "Model family", "Degradation recall", "Per-class F1", "Predicted", "Actual",
        "Row-normalised (recall)", "Model",
        "Current-removal penalty (ELEC_FULL − NO_CURRENT)",
        "Voltage-removal penalty (ELEC_FULL − NO_VOLTAGE)",
        "Standard ~600-sample scheduled inductions",
        "Two long anomalous events (1,779 and 2,695 samples)",
        "DIAGNOSTIC ONLY — methodologically invalid",
        "Stratified\nrandom-row 5-fold", "Group-disjoint\n5-fold",
        PROTOCOL, MLP_CAVEAT,
    ] + [d_ for d_ in DISPLAY.values()] + CLASSES + MODELS
    validate_terminology(figure_strings)
    print(f"\nTerminology assertion over {len(figure_strings)} figure strings: PASS")
    print(f"  Banned phrases enforced: {len(BANNED)}")
    print("\n  Checked strings:")
    for s_ in figure_strings:
        print(f"    | {str(s_)[:110].replace(chr(10), ' / ')}")

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"  exists already: {OUT_DIR.exists()}")
    targets = [f"{stem}{ext}" for stem in
               ["fig1_consolidated_results", "fig2_sensor_removal_by_model",
                "fig3_split_induced_optimism", "fig4_confusion_matrix_best",
                "fig5_per_class_f1", "fig6_degradation_event_character"]
               for ext in (".png", ".pdf")] + ["figure_captions.txt",
                                               "figure_source_manifest.txt"]
    clashes = [f for f in targets if (OUT_DIR / f).exists()]
    print(f"  write targets: {len(targets)} | already present: "
          f"{clashes if clashes else 'none'}")

    if dry_run:
        print("\nDRY RUN COMPLETE — all checks passed, nothing written.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_fig1(t); print("  saved fig1_consolidated_results")
    plot_fig2(t); print("  saved fig2_sensor_removal_by_model")
    plot_fig3(t); print("  saved fig3_split_induced_optimism")
    plot_fig4(t, best_fs, best_model); print("  saved fig4_confusion_matrix_best")
    plot_fig5(t); print("  saved fig5_per_class_f1")
    plot_fig6(t, best_per_model); print("  saved fig6_degradation_event_character")

    caps = write_captions(t, best_fs, best_model, vl, tot, sd_lo, sd_hi,
                          rr_lo, rr_hi, best_per_model)
    validate_terminology(caps)
    print("  Terminology assertion over captions: PASS")
    write_source_manifest()
    print(f"\nSaved to: {OUT_DIR}")

    with open(RUN_LOG, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
                f"plot_objective1_results.py | input: {len(READ_LOG)} grouped-validation "
                f"CSVs (no historical file read) | output: 6 figures x PNG+PDF "
                f"+ captions + manifest in {OUT_DIR.name}/ | "
                f"notes: best {best_fs}/{best_model}; voltage removal larger in "
                f"{vl}/{tot} paired comparisons; plots only, no modelling\n")
    print(f"Run log appended: {RUN_LOG}")


FIG2_ANNOT = ""   # set in main() once the paired count is computed
FIG3_ANNOT = ""
FIG4_TITLE = ""

if __name__ == "__main__":
    import sys
    main(dry_run="--verify" in sys.argv)
