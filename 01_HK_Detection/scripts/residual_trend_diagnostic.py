"""
residual_trend_diagnostic.py
============================
Objective 2 — read-only characterisation of the systematic negative residual
observed across 2023 in every arm.

READ ONLY. Nothing is retrained, no threshold is re-fitted, no model or
dataset is modified. Inputs are the per-timestamp residual series already
written by objective2_expected_power_arms.py.

PURPOSE AND SCOPE LIMIT
-----------------------
The purpose is to determine how Level 3 should be interpreted, and secondarily
whether an internal temporal trend exists within the training period. If the
cause remains ambiguous between degradation, soiling, post-event damage and
model drift, that is reported as ambiguous and the investigation stops.
Objective 2 does not require the cause to be known.

IN-SAMPLE vs OUT-OF-SAMPLE CONFOUND — applies to every 2021-22 vs 2023 comparison
--------------------------------------------------------------------------------
Residuals for 2021-2022 are predictions on data used to FIT the downstream
models; residuals for 2023 are held-out test predictions. A better-centred
training-period distribution therefore cannot be read as pure physical
degradation or distribution shift, because part of the difference is simply
in-sample versus out-of-sample prediction error. The training-period monthly
series is used primarily to assess whether an INTERNAL trend exists within
2021-2022, where both ends of that comparison are in-sample and the confound
applies equally.

Saola (Sept 1-15 2023) is a PROXY EVENT WINDOW, not fault ground truth.
"""

import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
HK          = PROJECT_DIR / "outputs" / "objective2_modeling"
OUT_DIR     = HK / "residual_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG     = PROJECT_DIR / "outputs" / "run_log.txt"

TZ = "Asia/Hong_Kong"
TRAIN_END = pd.Timestamp("2022-12-31", tz=TZ)
P0, P1 = pd.Timestamp("2023-09-01", tz=TZ), pd.Timestamp("2023-09-15", tz=TZ)
CLEAN_MONTHS = [1, 2, 11, 12]
BANNED = ["fault recall", "false positive rate", "false-positive rate",
          "fault precision", "fpr", "recovers a saola signal",
          "the screening layer works"]
ARMS = {"Reference (measured irradiance)": "reference",
        "Arm A (pvlib only)": "arm_a",
        "Arm B (ML estimate)": "arm_b",
        "Arm C (pvlib + ML estimate)": "arm_c"}
COLORS = {"Reference (measured irradiance)": "#444444",
          "Arm A (pvlib only)": "#1f77b4",
          "Arm B (ML estimate)": "#ff7f0e",
          "Arm C (pvlib + ML estimate)": "#2ca02c"}
PROXY_NOTE = ("Sept 1-15 2023 is a PROXY EVENT WINDOW, not fault ground truth")
INSAMPLE_NOTE = ("2021-22 residuals are IN-SAMPLE (training) predictions; 2023 "
                 "residuals are OUT-OF-SAMPLE. Part of any difference between "
                 "them is prediction-error type, not physical change.")

_rep = []


def emit(s=""):
    low = s.lower()
    hits = [b for b in BANNED if b in low]
    assert not hits, f"Banned phrase in output: {hits} -> {s!r}"
    print(s)
    _rep.append(s)


emit("# Objective 2 — Residual Trend Diagnostic")
emit("")
emit(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
emit("")
emit("Read-only. Nothing retrained, no threshold re-fitted, no model or dataset "
     "modified.")
emit("")
emit(f"> **{PROXY_NOTE}.**")
emit(">")
emit(f"> **{INSAMPLE_NOTE}**")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
summary = pd.read_csv(HK / "arm_training_summary.csv").set_index("arm")
grid = pd.read_csv(HK / "objective2_grid.csv", index_col=0,
                   usecols=["Unnamed: 0", "ghi_clear"])
grid.index = pd.to_datetime(grid.index, utc=True).tz_convert(TZ)

R = {}
for arm, tag in ARMS.items():
    d = pd.read_csv(HK / f"residuals_full_{tag}.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True).tz_convert(TZ)
    d["ghi_clear"] = grid["ghi_clear"].reindex(d.index)
    d["in_train"] = d.index <= TRAIN_END
    R[arm] = d
n = len(R["Arm A (pvlib only)"])
emit(f"Loaded full-period residuals: **{n:,}** rows per arm, "
     f"{R['Arm A (pvlib only)'].index.min().date()} to "
     f"{R['Arm A (pvlib only)'].index.max().date()}")
emit(f"- training-period rows: {int(R['Arm A (pvlib only)']['in_train'].sum()):,} "
     f"(in-sample) | test-period rows: "
     f"{int((~R['Arm A (pvlib only)']['in_train']).sum()):,} (out-of-sample)")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1
# ─────────────────────────────────────────────────────────────────────────────
emit("## Part 1 — Characterising the shift")
emit("")
emit("### 1.1 Monthly residual, 2021-2023")
emit("")
rows = []
for arm, d in R.items():
    g = d.groupby(d.index.to_period("M"))["resid"]
    for per, s in g:
        rows.append({"arm": arm, "month": str(per), "n": int(len(s)),
                     "mean_W": round(float(s.mean()), 1),
                     "median_W": round(float(s.median()), 1),
                     "q25_W": round(float(s.quantile(.25)), 1),
                     "q75_W": round(float(s.quantile(.75)), 1),
                     "in_sample": bool(d.loc[s.index, "in_train"].all())})
mon = pd.DataFrame(rows)
mon.to_csv(OUT_DIR / "residual_monthly_stats.csv", index=False)
emit(f"Full table in `residual_monthly_stats.csv` ({len(mon)} arm-months). "
     f"Yearly condensation below.")
emit("")

emit("### 1.2 Residual distribution by year")
emit("")
emit("| arm | year | sample | n | mean | median | SD | p5 | p25 | p75 | p95 |")
emit("|---|---|---|---|---|---|---|---|---|---|---|")
yr_rows = []
for arm, d in R.items():
    for y in [2021, 2022, 2023]:
        s = d[d.index.year == y]["resid"]
        if len(s) < 50:
            continue
        smp = "in-sample" if y < 2023 else "OUT-of-sample"
        emit(f"| {arm} | {y} | {smp} | {len(s):,} | {s.mean():,.0f} | "
             f"{s.median():,.0f} | {s.std():,.0f} | {s.quantile(.05):,.0f} | "
             f"{s.quantile(.25):,.0f} | {s.quantile(.75):,.0f} | "
             f"{s.quantile(.95):,.0f} |")
        yr_rows.append({"arm": arm, "year": y, "sample": smp, "n": int(len(s)),
                        "mean_W": round(float(s.mean()), 1),
                        "median_W": round(float(s.median()), 1),
                        "sd_W": round(float(s.std()), 1),
                        "p5": round(float(s.quantile(.05)), 1),
                        "p25": round(float(s.quantile(.25)), 1),
                        "p75": round(float(s.quantile(.75)), 1),
                        "p95": round(float(s.quantile(.95)), 1)})
pd.DataFrame(yr_rows).to_csv(OUT_DIR / "residual_by_year.csv", index=False)
emit("")
emit(f"**{INSAMPLE_NOTE}**")
emit("")

emit("### 1.3 Within-2023 breakdown")
emit("")
emit("| arm | Jan-Feb (clean) | Mar-Oct (sensor fault) | Nov-Dec (clean) | "
     "Sept 1-15 proxy window |")
emit("|---|---|---|---|---|")
for arm, d in R.items():
    d23 = d[d.index.year == 2023]
    jf = d23[d23.index.month.isin([1, 2])]["resid"]
    mo = d23[d23.index.month.isin(range(3, 11))]["resid"]
    nd = d23[d23.index.month.isin([11, 12])]["resid"]
    pw = d23[(d23.index >= P0) & (d23.index <= P1)]["resid"]
    emit(f"| {arm} | {jf.mean():,.0f} (n={len(jf):,}) | {mo.mean():,.0f} "
         f"(n={len(mo):,}) | {nd.mean():,.0f} (n={len(nd):,}) | "
         f"{pw.mean():,.0f} (n={len(pw):,}) |")
emit("")
emit("Reference-arm residuals during Mar-Oct are affected by its corrupted "
     "sensor input and should be read accordingly.")
emit("")

emit("### 1.4 Pre-Saola / Saola / post-Saola (8 weeks either side)")
emit("")
pre0, post1 = P0 - pd.Timedelta(weeks=8), P1 + pd.Timedelta(weeks=8)
emit("| arm | pre (8 wk) | proxy window | post (8 wk) |")
emit("|---|---|---|---|")
for arm, d in R.items():
    pre = d[(d.index >= pre0) & (d.index < P0)]["resid"]
    pw = d[(d.index >= P0) & (d.index <= P1)]["resid"]
    post = d[(d.index > P1) & (d.index <= post1)]["resid"]
    emit(f"| {arm} | {pre.mean():,.0f} (n={len(pre):,}) | {pw.mean():,.0f} "
         f"(n={len(pw):,}) | {post.mean():,.0f} (n={len(post):,}) |")
emit("")

emit("### 1.5 Do the arms shift together?")
emit("")
piv = mon.pivot(index="month", columns="arm", values="mean_W")
cc = piv.corr()
emit("Correlation of monthly mean residual between arms:")
emit("")
emit("| | " + " | ".join(a.split(" (")[0] for a in ARMS) + " |")
emit("|---" * (len(ARMS) + 1) + "|")
for a in ARMS:
    emit(f"| {a.split(' (')[0]} | " +
         " | ".join(f"{cc.loc[a, b]:.3f}" for b in ARMS) + " |")
offdiag = [cc.loc[a, b] for a in ARMS for b in ARMS if a != b]
emit("")
emit(f"Mean off-diagonal correlation: **{np.mean(offdiag):.3f}**. Arms moving "
     f"together points to a system- or site-level cause rather than to any one "
     f"arm's irradiance representation.")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2
# ─────────────────────────────────────────────────────────────────────────────
emit("## Part 2 — Candidate causes")
emit("")
emit("### 2.1 Internal trend WITHIN the training period (both ends in-sample)")
emit("")
emit("| arm | slope (W/month) | total drift over 2021-06 to 2022-12 | direction |")
emit("|---|---|---|---|")
trend_rows = []
for arm, d in R.items():
    tr = mon[(mon.arm == arm) & (mon.in_sample)].copy()
    tr["t"] = np.arange(len(tr))
    if len(tr) < 6:
        continue
    slope, intercept = np.polyfit(tr["t"], tr["mean_W"], 1)
    total = slope * (len(tr) - 1)
    emit(f"| {arm} | {slope:+,.1f} | {total:+,.0f} W | "
         f"{'downward' if slope < 0 else 'upward'} |")
    trend_rows.append({"arm": arm, "period": "2021-06..2022-12 (in-sample)",
                       "slope_W_per_month": round(float(slope), 2),
                       "total_drift_W": round(float(total), 1),
                       "n_months": int(len(tr))})
emit("")
emit("This is the cleanest available trend test: both ends are in-sample, so "
     "the in-sample/out-of-sample confound applies equally across it.")
emit("")

emit("### 2.2 Proportional or absolute?")
emit("")
emit("| arm | 2023 mean residual (W) | 2023 mean residual / mean predicted (%) | "
     "training mean residual/pred (%) |")
emit("|---|---|---|---|")
prop_rows = []
for arm, d in R.items():
    d23 = d[d.index.year == 2023]
    dtr = d[d["in_train"]]
    r23 = float(d23["resid"].mean())
    f23 = 100 * r23 / float(d23["pred"].mean())
    ftr = 100 * float(dtr["resid"].mean()) / float(dtr["pred"].mean())
    emit(f"| {arm} | {r23:,.0f} | {f23:+.1f}% | {ftr:+.1f}% |")
    prop_rows.append({"arm": arm, "resid_2023_W": round(r23, 1),
                      "frac_2023_pct": round(f23, 2),
                      "frac_train_pct": round(ftr, 2)})
emit("")

emit("### 2.3 Residual by clear-sky irradiance decile, 2022 vs 2023")
emit("")
emit(f"**{INSAMPLE_NOTE}** 2022 is in-sample and 2023 out-of-sample, so a "
     f"uniform vertical offset between the two curves is expected even with no "
     f"physical change. The informative feature is the SHAPE — whether the gap "
     f"widens with irradiance (proportional, consistent with degradation) or "
     f"stays flat (absolute offset).")
emit("")
bin_rows = []
for arm, d in R.items():
    dd = d[d["ghi_clear"].notna()].copy()
    dd["dec"] = pd.qcut(dd["ghi_clear"], 10, labels=False, duplicates="drop")
    for y in [2022, 2023]:
        sub = dd[dd.index.year == y]
        for dec, s in sub.groupby("dec"):
            bin_rows.append({"arm": arm, "year": y, "ghi_clear_decile": int(dec),
                             "n": int(len(s)),
                             "mean_ghi_clear": round(float(s["ghi_clear"].mean()), 1),
                             "mean_resid_W": round(float(s["resid"].mean()), 1),
                             "mean_pred_W": round(float(s["pred"].mean()), 1),
                             "sample": "in-sample" if y < 2023 else "out-of-sample"})
bins = pd.DataFrame(bin_rows)
bins.to_csv(OUT_DIR / "residual_by_irradiance_bin.csv", index=False)
emit("| arm | decile 0-2 gap (W) | decile 7-9 gap (W) | widens with irradiance? |")
emit("|---|---|---|---|")
for arm in ARMS:
    b = bins[bins.arm == arm]
    lo = (b[(b.year == 2023) & (b.ghi_clear_decile <= 2)]["mean_resid_W"].mean()
          - b[(b.year == 2022) & (b.ghi_clear_decile <= 2)]["mean_resid_W"].mean())
    hi = (b[(b.year == 2023) & (b.ghi_clear_decile >= 7)]["mean_resid_W"].mean()
          - b[(b.year == 2022) & (b.ghi_clear_decile >= 7)]["mean_resid_W"].mean())
    emit(f"| {arm} | {lo:,.0f} | {hi:,.0f} | "
         f"{'yes' if abs(hi) > abs(lo) * 1.5 else 'no / weakly'} |")
pd.DataFrame(prop_rows + trend_rows).to_csv(
    OUT_DIR / "residual_trend_summary.csv", index=False)
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — CAUSAL DETRENDING (DIAGNOSTIC ONLY)
# ─────────────────────────────────────────────────────────────────────────────
emit("## Part 3 — Causal detrended residual (DIAGNOSTIC ONLY)")
emit("")
emit("A trailing median over the preceding N days, using only timestamps "
     "strictly before the current one. This is **not leakage** — it consults no "
     "future information — but it changes what the detector means. It answers "
     "*is this unusual relative to recent behaviour* rather than *is this "
     "unusual relative to healthy operation*. A detector that continually "
     "re-centres on recent test-period behaviour can absorb sustained "
     "degradation as the new normal, the same failure mode that motivated "
     "removing lagged power.")
emit("")
emit("**These numbers are diagnostic. They are not adopted as the primary "
     "detector and are not Objective 2 results.**")
emit("")
emit("The same numeric training-fitted z threshold is reused; no threshold is "
     "re-fitted.")
emit("")
det_rows = []
for N in [7, 30, 90]:
    emit(f"### Trailing window N = {N} days")
    emit("")
    emit("| arm | mean z pre (8wk) | mean z proxy window | mean z post (8wk) | "
         "contrast vs pre | event-window rate | alert rate outside window |")
    emit("|---|---|---|---|---|---|---|")
    for arm, d in R.items():
        sig = float(summary.loc[arm, "resid_sigma_train_W"])
        thr = float(summary.loc[arm, "alert_z_threshold"])
        s = d["resid"].sort_index()
        trail = s.shift(1).rolling(f"{N}D").median()
        det = (s - trail) / sig
        dd = pd.DataFrame({"z": det}).dropna()
        pre = dd[(dd.index >= pre0) & (dd.index < P0)]["z"]
        pw = dd[(dd.index >= P0) & (dd.index <= P1)]["z"]
        post = dd[(dd.index > P1) & (dd.index <= post1)]["z"]
        below = (dd["z"] < thr).astype(int)
        grp = below.groupby((below != below.shift()).cumsum()).transform("sum")
        alert = ((below == 1) & (grp >= 4)).astype(int)
        inp = (dd.index >= P0) & (dd.index <= P1)
        in23 = dd.index > TRAIN_END
        outp = in23 & ~inp
        er = 100 * alert[inp].mean() if inp.sum() else np.nan
        ar = 100 * alert[outp].mean() if outp.sum() else np.nan
        emit(f"| {arm} | {pre.mean():+.3f} | {pw.mean():+.3f} | "
             f"{post.mean():+.3f} | {pw.mean()-pre.mean():+.3f} | {er:.1f}% | "
             f"{ar:.1f}% |")
        det_rows.append({"N_days": N, "arm": arm,
                         "z_pre": round(float(pre.mean()), 4),
                         "z_proxy": round(float(pw.mean()), 4),
                         "z_post": round(float(post.mean()), 4),
                         "contrast_vs_pre": round(float(pw.mean()-pre.mean()), 4),
                         "event_window_rate_pct": round(float(er), 2),
                         "alert_rate_outside_pct": round(float(ar), 2)})
    emit("")
pd.DataFrame(det_rows).to_csv(OUT_DIR / "detrended_diagnostic.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — IMPLICATION FOR LEVEL 3
# ─────────────────────────────────────────────────────────────────────────────
emit("## Part 4 — Implication for Level 3 as currently reported")
emit("")
emit("| arm | training-fitted z threshold | 2023 median z | mis-centring "
     "(median z, in σ) | 2023 alert slots | slots if 2023 median removed | "
     "attributable to year-level offset |")
emit("|---|---|---|---|---|---|---|")
off_rows = []
for arm, d in R.items():
    sig = float(summary.loc[arm, "resid_sigma_train_W"])
    thr = float(summary.loc[arm, "alert_z_threshold"])
    d23 = d[d.index.year == 2023].copy()
    z = d23["resid"] / sig
    med = float(z.median())

    def count_alerts(zs):
        below = (zs < thr).astype(int)
        grp = below.groupby((below != below.shift()).cumsum()).transform("sum")
        return int(((below == 1) & (grp >= 4)).sum())

    a_raw = count_alerts(z)
    a_dm = count_alerts(z - med)
    share = 100 * (a_raw - a_dm) / a_raw if a_raw else np.nan
    emit(f"| {arm} | {thr:.3f} | {med:+.3f} | {med:+.3f}σ | {a_raw:,} | "
         f"{a_dm:,} | {share:.0f}% |")
    off_rows.append({"arm": arm, "train_z_threshold": thr,
                     "median_z_2023": round(med, 4),
                     "alert_slots_2023": a_raw,
                     "alert_slots_after_median_removed": a_dm,
                     "pct_attributable_to_offset": round(float(share), 1)})
pd.DataFrame(off_rows).to_csv(OUT_DIR / "threshold_offset_analysis.csv",
                              index=False)
emit("")
emit("The decomposition subtracts the 2023 median z before applying the same "
     "unchanged threshold. It is a diagnostic attribution, not a re-fit.")
emit("")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    emit(f"- saved `{name}.png` / `.pdf`")


emit("## Figures")
emit("")
# 1 monthly series
fig, ax = plt.subplots(figsize=(14, 6))
for arm in ARMS:
    m = mon[mon.arm == arm].copy()
    x = pd.PeriodIndex(m["month"], freq="M").to_timestamp().to_numpy()
    ax.plot(x, m["mean_W"].to_numpy(), marker="o", ms=3, color=COLORS[arm],
            label=arm.split(" (")[0])
    ax.fill_between(x, m["q25_W"].to_numpy(), m["q75_W"].to_numpy(),
                    color=COLORS[arm], alpha=0.10)
ax.axvspan(pd.Timestamp("2023-03-01"), pd.Timestamp("2023-10-31"),
           color="gold", alpha=0.18)
ax.text(pd.Timestamp("2023-06-15"), ax.get_ylim()[1]*0.92,
        "sensor fault window\n(Mar–Oct 2023)", ha="center", fontsize=8)
ax.axvspan(pd.Timestamp("2023-09-01"), pd.Timestamp("2023-09-15"),
           color="red", alpha=0.25)
ax.axvline(pd.Timestamp("2023-01-01"), color="k", ls="--", lw=1)
ax.text(pd.Timestamp("2022-12-20"), ax.get_ylim()[0]*0.9,
        "in-sample ← | → out-of-sample", ha="right", fontsize=8)
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylabel("Monthly mean residual (W), bands = IQR")
ax.set_title("Monthly residual by arm, 2021–2023 — red band = Saola PROXY EVENT "
             "WINDOW, not fault ground truth")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.text(0.5, -0.03, INSAMPLE_NOTE, ha="center", fontsize=8,
         bbox=dict(fc="white", ec="grey"))
save(fig, "residual_monthly_series")

# 2 by-year distribution
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
for ax, arm in zip(axes, ARMS):
    d = R[arm]
    data = [d[d.index.year == y]["resid"].to_numpy() for y in (2021, 2022, 2023)]
    bp = ax.boxplot(data, labels=["2021\nin-sample", "2022\nin-sample",
                                  "2023\nOUT-of-sample"], showfliers=False,
                    patch_artist=True)
    for p in bp["boxes"]:
        p.set_facecolor(COLORS[arm])
        p.set_alpha(0.5)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(arm.split(" (")[0], fontsize=10)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("Residual (W)")
fig.suptitle("Residual distribution by year — 2021/22 in-sample vs 2023 "
             "out-of-sample (Saola is a proxy window, not fault ground truth)")
fig.text(0.5, -0.02, INSAMPLE_NOTE, ha="center", fontsize=8,
         bbox=dict(fc="white", ec="grey"))
save(fig, "residual_by_year_distribution")

# 3 irradiance decile
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
for ax, arm in zip(axes, ARMS):
    b = bins[bins.arm == arm]
    for y, ls, lab in [(2022, "-", "2022 (in-sample)"),
                       (2023, "--", "2023 (out-of-sample)")]:
        s = b[b.year == y].sort_values("ghi_clear_decile")
        ax.plot(s["ghi_clear_decile"].to_numpy(), s["mean_resid_W"].to_numpy(),
                ls, marker="o",
                color=COLORS[arm], alpha=1.0 if y == 2023 else 0.5, label=lab)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(arm.split(" (")[0], fontsize=10)
    ax.set_xlabel("clear-sky GHI decile")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Mean residual (W)")
fig.suptitle("Mean residual by clear-sky irradiance decile, 2022 vs 2023 — "
             "shape, not vertical offset, distinguishes proportional from absolute")
fig.text(0.5, -0.02, INSAMPLE_NOTE + " A uniform vertical gap is expected; the "
         "informative feature is whether it widens with irradiance.",
         ha="center", fontsize=8, bbox=dict(fc="white", ec="grey"))
save(fig, "residual_by_irradiance_bin")

# 4 detrended around Saola
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
for ax, arm in zip(axes, ARMS):
    d = R[arm]
    sig = float(summary.loc[arm, "resid_sigma_train_W"])
    s = d["resid"].sort_index()
    prim = (s / sig)
    det = ((s - s.shift(1).rolling("30D").median()) / sig)
    w = (s.index >= pre0) & (s.index <= post1)
    pr = prim[w].resample("D").mean()
    dt = det[w].resample("D").mean()
    ax.plot(pr.index.to_numpy(), pr.to_numpy(), color="grey", lw=1.2,
            label="primary z")
    ax.plot(dt.index.to_numpy(), dt.to_numpy(), color=COLORS[arm], lw=1.4,
            label="causal detrended z (DIAGNOSTIC)")
    ax.axvspan(P0, P1, color="red", alpha=0.22)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title(arm.split(" (")[0], fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
axes[0].set_ylabel("daily mean z")
fig.suptitle("Primary vs causal-detrended residual z around the Saola PROXY "
             "EVENT WINDOW (red) — detrended series is DIAGNOSTIC ONLY, not a "
             "detector and not an Objective 2 result")
save(fig, "residual_detrended_saola")

report = "\n".join(_rep) + "\n"
for b in BANNED:
    assert b not in report.lower(), f"banned phrase in report: {b}"
(OUT_DIR / "residual_diagnostic_report.md").write_text(report)
print(f"\nSaved to: {OUT_DIR}")
with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"residual_trend_diagnostic.py | input: residuals_full_*.csv "
            f"({n:,} rows/arm), grid, training summary | output: 5 CSVs + 4 "
            f"figures + report in objective2_modeling/residual_diagnostic/ | "
            f"notes: read-only, nothing retrained, no threshold re-fitted; "
            f"detrended series diagnostic only\n")
print(f"Run log appended: {RUN_LOG}")
