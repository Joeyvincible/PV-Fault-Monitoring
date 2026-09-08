"""
objective2_expected_power_arms.py
==================
Objective 2 — Stage 2: four-arm exogenous-only expected-power models,
evaluated at three levels across three slices.

BINDING CONSTRAINTS IMPLEMENTED HERE
------------------------------------
[1] EXOGENOUS-ONLY. Lagged PV power is not an input to any arm. power(t) is
    the prediction target and the residual-comparison signal, never a
    predictor. Rationale: during a sustained fault, recent depressed power
    would teach the model to expect depressed power, collapsing the residual
    — the Typhoon Saola case exactly. It also stops lagged power dominating
    the prediction and masking the differences between arms, which is what
    Objective 2 exists to measure.
    Expected cost: removing the strongest predictor should raise
    expected-power MAE materially against the historical ~25% nMAE. That is
    reported plainly as the price of a detector that cannot absorb a
    sustained fault.

[2] TYPHOON SAOLA IS A PROXY WINDOW, NOT FAULT GROUND TRUTH. HK has no
    per-timestep PV fault labels. Ordinary fault precision/recall/false-
    positive rate are NOT reported: they would imply every timestamp outside
    the window is fault-free, which is unknowable. Three evaluation levels
    are reported instead (irradiance accuracy; expected-power accuracy;
    anomaly screening). Banned metric names are blocked by assertion.

[3] THREE SLICES, each labelled wherever it appears.

[4] CONTINUOUS TIME GRID. Sequences are built positionally on the gap-free
    15-minute index and never on a compressed post-mask index. Each window's
    timestamps are asserted to be exactly 15 minutes apart, so no artificial
    adjacency is created across removed rows. Eligibility is applied to the
    TARGET timestamp only.
    Current-time exogenous information IS permitted: ghi_clear(t), solar
    geometry, ambient weather and the same-time estimate irr_est(t) are all
    legitimate inputs for predicting power(t). Only power(t) is barred.
    Excluding irr_est(t) would defeat the objective.
"""

import datetime
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT_DIR     = PROJECT_DIR / "outputs" / "objective2_modeling"
GRID_CSV    = OUT_DIR / "objective2_grid.csv"
RUN_LOG     = PROJECT_DIR / "outputs" / "run_log.txt"

TZ = "Asia/Hong_Kong"
SEQ_LEN = 16                 # 4 hours of exogenous context, inclusive of t
TRAIN_END = "2022-12-31"
TARGET = "power_W"
RANDOM_SEED = 42
EPOCHS, BATCH, LR, HIDDEN, LAYERS = 30, 256, 1e-3, 64, 2
CONSEC_SLOTS = 4             # persistence for an anomaly alert (1 hour)
ALERT_QUANTILE = 0.01        # threshold from the TRAINING residual distribution

PROXY_START, PROXY_END = "2023-09-01", "2023-09-15"
CLEAN_MONTHS = [1, 2, 11, 12]

BANNED = ["fault recall", "false positive rate", "false-positive rate",
          "fault precision", "fpr", "ground-truth label"]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
_rep = []


def emit(s=""):
    low = s.lower()
    hits = [b for b in BANNED if b in low]
    assert not hits, f"[2] Banned metric terminology in output: {hits} -> {s!r}"
    print(s)
    _rep.append(s)


emit("# Objective 2 — Stage 2: four-arm evaluation")
emit("")
emit(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
emit("")

df = pd.read_csv(GRID_CSV, index_col=0)
df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
emit(f"Grid: **{len(df):,}** rows, {df.index.min()} to {df.index.max()}")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS — all exogenous [1]
# ─────────────────────────────────────────────────────────────────────────────
# COMMON contains no irradiance representation. Each arm adds its specified
# representation; Arm C includes both clear-sky and estimated irradiance.
COMMON = ["zenith", "azimuth", "Temp (Degree Celsius)",
          "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year"]

# Retain gap and ratio for labelled analysis; they are not used in the primary
# Arm C. The model receives clear-sky and estimated irradiance directly.
df["irr_gap"] = df["irr_est"] - df["ghi_clear"]
df["irr_ratio"] = df["irr_est"] / df["ghi_clear"].clip(lower=1.0)

ARMS = {
    "Reference (measured irradiance)": COMMON + ["irr_meas"],
    "Arm A (pvlib only)":              COMMON + ["ghi_clear"],
    "Arm B (ML estimate)":             COMMON + ["irr_est"],
    "Arm C (pvlib + ML estimate)":     COMMON + ["ghi_clear", "irr_est"],
}
assert "ghi_clear" not in ARMS["Arm B (ML estimate)"], \
    "Arm B must not receive pvlib clear-sky"
assert "ghi_clear" not in ARMS["Reference (measured irradiance)"], \
    "Reference carries the measured representation only"

emit("")
emit("## Feature sets — exogenous only [1]")
emit("")
emit("| arm | features | count |")
emit("|---|---|---|")
for name, feats in ARMS.items():
    assert TARGET not in feats, f"[1] VIOLATION: target in features of {name}"
    assert not any("power" in f for f in feats), \
        f"[1] VIOLATION: a power-derived feature is in {name}"
    emit(f"| {name} | {', '.join(feats)} | {len(feats)} |")
emit("")
emit("**[1] assertion passed for all four arms — no power-derived column is an "
     "input anywhere.** `power_W` is the target only.")
emit("")
emit("Current-time exogenous values are inputs by design [4]: `ghi_clear(t)`, "
     "solar geometry, ambient temperature and `irr_est(t)` all carry the target "
     "timestamp. Only `power(t)` is barred. Excluding `irr_est(t)` would defeat "
     "the objective, which is precisely to test whether a same-time estimate can "
     "replace the physical measurement.")

# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE CONSTRUCTION ON THE CONTINUOUS GRID [4]
# ─────────────────────────────────────────────────────────────────────────────
emit("")
emit("## Sequence construction on the continuous grid [4]")
emit("")
step = pd.Timedelta("15min")
idx = df.index
spacing_ok = bool((pd.Series(idx).diff().dropna() == step).all())
emit(f"- Grid uniformly spaced at 15 min: **{spacing_ok}**")
assert spacing_ok, "Grid is not uniform; positional windows would be invalid"
emit(f"- Window: `t-{SEQ_LEN-1} … t` **inclusive of t** (exogenous only), "
     f"target `power(t)`")


def build(feats):
    """Positional windows on the continuous grid. Returns X, y, timestamps."""
    F = df[feats].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    elig = df["eligible"].to_numpy().astype(bool)
    n = len(df)
    Xs, ys, ts, bad_spacing = [], [], [], 0
    for i in range(SEQ_LEN - 1, n):
        if not elig[i] or not np.isfinite(y[i]):
            continue
        s = i - SEQ_LEN + 1
        # [4] strict contiguity: the window must be genuinely adjacent in time
        if idx[i] - idx[s] != step * (SEQ_LEN - 1):
            bad_spacing += 1
            continue
        w = F[s:i + 1]
        if not np.isfinite(w).all():
            continue
        Xs.append(w)
        ys.append(y[i])
        ts.append(idx[i])
    return (np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
            pd.DatetimeIndex(ts), bad_spacing)


class LSTMReg(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, HIDDEN, LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :]).squeeze(-1)


# CPU by default: these models are small (2-layer LSTM, hidden 64, 16 steps) and
# MPS on this machine was contended by another process. Set HK_USE_MPS=1 to try
# the GPU instead.
import os
_use_mps = os.environ.get("HK_USE_MPS") == "1" and torch.backends.mps.is_available()
device = torch.device("mps" if _use_mps else "cpu")
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
emit(f"- Device: {device}")

train_end = pd.Timestamp(TRAIN_END, tz=TZ)
p0, p1 = pd.Timestamp(PROXY_START, tz=TZ), pd.Timestamp(PROXY_END, tz=TZ)

results, per_slice, alerts_rows, resid_store = [], [], [], {}
emit("")
emit("## Training")
emit("")
for arm, feats in ARMS.items():
    t0 = time.time()
    X, y, ts, bad = build(feats)
    tr = ts <= train_end
    te = ts > train_end
    # verification: no input timestamp may equal or exceed the target for power
    assert TARGET not in feats
    emit(f"- **{arm}**: {len(X):,} sequences "
         f"(train {int(tr.sum()):,} / test {int(te.sum()):,}); "
         f"windows rejected for non-contiguous spacing: {bad}")

    nf = X.shape[2]
    sc = StandardScaler().fit(X[tr].reshape(-1, nf))          # training only
    Xs = sc.transform(X.reshape(-1, nf)).reshape(X.shape).astype(np.float32)
    ym, ysd = float(y[tr].mean()), float(y[tr].std())          # training only
    yz = (y - ym) / ysd

    model = LSTMReg(nf).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    Xtr = torch.tensor(Xs[tr]).to(device)
    ytr = torch.tensor(yz[tr]).to(device)
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred_z = np.concatenate([
            model(torch.tensor(Xs[i:i + 4096]).to(device)).cpu().numpy()
            for i in range(0, len(Xs), 4096)])
    pred = pred_z * ysd + ym
    resid = y - pred
    resid_store[arm] = pd.DataFrame(
        {"pred": pred, "actual": y, "resid": resid}, index=ts)

    # anomaly threshold from the TRAINING residual distribution only
    sigma = float(resid[tr].std())
    z = resid / sigma
    k = float(np.quantile(z[tr], ALERT_QUANTILE))
    secs = time.time() - t0
    results.append({"arm": arm, "n_features": nf, "sequences": int(len(X)),
                    "train_seq": int(tr.sum()), "test_seq": int(te.sum()),
                    "resid_sigma_train_W": round(sigma, 1),
                    "alert_z_threshold": round(k, 3),
                    "runtime_s": round(secs, 1)})
    resid_store[arm]["z"] = z
    emit(f"  trained in {secs:.0f}s | training residual σ = {sigma:,.0f} W | "
         f"alert threshold z < {k:.2f} (training {ALERT_QUANTILE:.0%} quantile)")

pd.DataFrame(results).to_csv(OUT_DIR / "arm_training_summary.csv", index=False)

# [C1] structural check: all four arms must share one evaluation population.
ref_ts = resid_store["Arm A (pvlib only)"].index
for arm, r in resid_store.items():
    assert r.index.equals(ref_ts), (
        f"[C1] VIOLATION: {arm} has a different evaluation population "
        f"({len(r):,} vs {len(ref_ts):,} rows). All arms must be scored on the "
        f"identical common sensor-free mask.")
emit("")
emit(f"**[C1] assertion passed — all four arms share one identical evaluation "
     f"population of {len(ref_ts):,} sequences**, so the head-to-head is "
     f"like-for-like. The Reference arm consumes the sensor as a feature but "
     f"does not get its own row set.")

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — EXPECTED-POWER PREDICTION ACCURACY, THREE SLICES [2][3]
# ─────────────────────────────────────────────────────────────────────────────
emit("")
emit("## Level 2 — expected-AC-power prediction accuracy [2]")
emit("")
SLICES = {
    "Slice 1 — Jan/Feb/Nov/Dec 2023 (clean sensor; NO known fault event; "
    "compares prediction accuracy, NOT fault-detection ability)":
        lambda t: (t > train_end) & pd.Index(t).month.isin(CLEAN_MONTHS),
    "Slice 2 — full 2023 (no-sensor deployment evaluation; Reference arm "
    "UNRELIABLE Mar–Oct)":
        lambda t: t > train_end,
    "Slice 3 — Sept 1–15 2023 Saola proxy window (case study only; Reference "
    "arm's sensor faulty or absent throughout)":
        lambda t: (t >= p0) & (t <= p1),
}
for sname, sel in SLICES.items():
    emit(f"### {sname}")
    emit("")
    emit("| arm | n | MAE (W) | RMSE (W) | nMAE (%) | mean residual (W) |")
    emit("|---|---|---|---|---|---|")
    for arm in ARMS:
        r = resid_store[arm]
        m = sel(r.index)
        sub = r[m]
        if len(sub) < 20:
            emit(f"| {arm} | {len(sub)} | – | – | – | – |")
            continue
        mae = mean_absolute_error(sub["actual"], sub["pred"])
        rmse = float(np.sqrt(mean_squared_error(sub["actual"], sub["pred"])))
        nmae = 100 * mae / sub["actual"].mean()
        emit(f"| {arm} | {len(sub):,} | {mae:,.0f} | {rmse:,.0f} | {nmae:.1f} | "
             f"{sub['resid'].mean():+,.0f} |")
        per_slice.append({"slice": sname.split(" —")[0], "arm": arm,
                          "n": int(len(sub)), "MAE_W": round(mae, 1),
                          "RMSE_W": round(rmse, 1), "nMAE_pct": round(nmae, 2),
                          "mean_residual_W": round(float(sub["resid"].mean()), 1)})
    emit("")
pd.DataFrame(per_slice).to_csv(OUT_DIR / "expected_power_accuracy.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3 — ANOMALY / POTENTIAL-FAULT SCREENING [2]
# ─────────────────────────────────────────────────────────────────────────────
emit("## Level 3 — anomaly / potential-fault screening [2]")
emit("")
emit("Reported as screening behaviour, not as labelled classification. The "
     "Saola window is a calendar proxy: timestamps outside it are **not** known "
     "to be fault-free, so metrics that presuppose labels are not computed.")
emit("")
emit("| arm | anomalies in proxy window | proxy-window slots | event-window "
     "detection rate | alert rate outside proxy window (2023) | total 2023 "
     "anomaly slots |")
emit("|---|---|---|---|---|---|")
for arm in ARMS:
    r = resid_store[arm]
    thr = [x for x in results if x["arm"] == arm][0]["alert_z_threshold"]
    below = (r["z"] < thr).astype(int)
    grp = below.groupby((below != below.shift()).cumsum()).transform("sum")
    alert = ((below == 1) & (grp >= CONSEC_SLOTS)).astype(int)
    r["alert"] = alert
    in_p = (r.index >= p0) & (r.index <= p1)
    in_23 = r.index > train_end
    out_p = in_23 & ~in_p
    det = 100 * alert[in_p].mean() if in_p.sum() else np.nan
    outr = 100 * alert[out_p].mean() if out_p.sum() else np.nan
    emit(f"| {arm} | {int(alert[in_p].sum()):,} | {int(in_p.sum()):,} | "
         f"{det:.1f}% | {outr:.1f}% | {int(alert[in_23].sum()):,} |")
    alerts_rows.append({"arm": arm, "proxy_anomaly_slots": int(alert[in_p].sum()),
                        "proxy_slots": int(in_p.sum()),
                        "event_window_detection_rate_pct": round(float(det), 2),
                        "alert_rate_outside_proxy_pct": round(float(outr), 2),
                        "total_2023_anomaly_slots": int(alert[in_23].sum()),
                        "alert_z_threshold": thr})
pd.DataFrame(alerts_rows).to_csv(OUT_DIR / "anomaly_screening.csv", index=False)

emit("")
emit("### Residual behaviour around the proxy window")
emit("")
emit("| arm | mean z, Aug 2023 | mean z, proxy window | mean z, Oct 2023 |")
emit("|---|---|---|---|")
for arm in ARMS:
    r = resid_store[arm]
    aug = r[(r.index >= pd.Timestamp("2023-08-01", tz=TZ)) & (r.index < p0)]["z"]
    pw = r[(r.index >= p0) & (r.index <= p1)]["z"]
    octb = r[(r.index >= pd.Timestamp("2023-10-01", tz=TZ))
             & (r.index < pd.Timestamp("2023-11-01", tz=TZ))]["z"]
    emit(f"| {arm} | {aug.mean():+.3f} | {pw.mean():+.3f} | {octb.mean():+.3f} |")
emit("")
emit("A more negative mean z inside the proxy window than either side indicates "
     "the arm's expected-power model saw output fall below environmental "
     "expectation during the storm period.")

for arm, r in resid_store.items():
    tag = arm.split(" (")[0].replace(" ", "_").lower()
    r[r.index > train_end].to_csv(OUT_DIR / f"residuals_2023_{tag}.csv")
    # Additive export only: the full 2021-2023 series, for the residual trend
    # diagnostic. This widens what is written and changes no model, parameter,
    # seed, feature set, threshold or training procedure.
    r.to_csv(OUT_DIR / f"residuals_full_{tag}.csv")

report = "\n".join(_rep) + "\n"
for b in BANNED:
    assert b not in report.lower(), f"[2] banned term in report: {b}"
(OUT_DIR / "arms_report.md").write_text(report)
print(f"\nSaved to: {OUT_DIR}")
with open(RUN_LOG, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"objective2_expected_power_arms.py | input: objective2_grid.csv | "
            f"output: 4 arms x 3 slices, screening + residual CSVs in objective2_modeling/ | "
            f"notes: exogenous-only (no lagged power) [1]; proxy-window screening "
            f"only, no label-presuming metrics [2]; 3 slices [3]; contiguous-grid "
            f"sequences with current-time exogenous permitted [4]\n")
print(f"Run log appended: {RUN_LOG}")
