"""
mat_to_csv.py
=============
Convert the Brazil fault dataset MAT files to CSV.

The MAT files contain NO time variable. Earlier revisions of this script
fabricated one: find_key's bare-suffix branch made the target "t" match the
key "pvt", so the ambient CSV's "time" column was a silent copy of module
temperature. Both the bare-suffix branch and the "t" target are now removed,
and no time column is emitted at all — only an explicit positional
sample_index, which is a row counter and not a clock.

Converted CSV output is written to raw_data/converted_csv/ while the source
MAT files remain untouched.
"""

import os
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_DIR / "raw_data"
CONVERTED_DIR = RAW_DATA_DIR / "converted_csv"
RUN_LOG = PROJECT_DIR / "outputs" / "run_log.txt"

def load_mat_any(filename: str):
    """
    Loads MATLAB .mat files.
    - For standard MAT files: uses scipy.io.loadmat
    - For v7.3 MAT files (HDF5): falls back to h5py
    """
    try:
        import scipy.io as sio
        return sio.loadmat(filename)
    except NotImplementedError:
        # likely v7.3
        import h5py
        data = {}
        with h5py.File(filename, "r") as f:
            def read_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    arr = np.array(obj)
                    data[name] = arr
            f.visititems(read_item)
        return data

def clean_vector(x):
    """Flatten MATLAB arrays into 1D python arrays where possible."""
    x = np.array(x)
    x = np.squeeze(x)
    return x

def find_key(mat_dict, target: str):
    """
    Find a key in mat_dict that matches:
    1) exact
    2) case-insensitive exact
    3) HDF5 path-suffix match (e.g., 'signals/vdc1' matches 'vdc1')

    The bare-suffix branch was removed. It matched any key ENDING in the
    target, so target "t" matched key "pvt".
    """
    # exact
    if target in mat_dict:
        return target

    # case-insensitive exact
    lower_map = {k.lower(): k for k in mat_dict.keys()}
    if target.lower() in lower_map:
        return lower_map[target.lower()]

    # path-suffix match for HDF5 paths only — must follow a "/" separator
    for k in mat_dict.keys():
        if k.lower().endswith("/" + target.lower()):
            return k

    return None

def pick_vars(mat_dict):
    """Try to pull the variables we care about from the .mat dict."""
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    print("Available keys:", keys)

    # "t" removed: the MAT files hold no time variable, and searching for it
    # is what triggered the pvt collision. "time"/"timestamp" are retained as
    # a diagnostic only — if either ever appears, it is reported, not written.
    wanted = ["vdc1", "vdc2", "idc1", "idc2", "irr", "pvt", "f_nv", "time", "timestamp"]
    found = {}
    for w in wanted:
        k = find_key(mat_dict, w)
        if k is not None:
            found[w] = clean_vector(mat_dict[k])
    return found

def align_lengths(found, cols):
        present = [c for c in cols if c in found]
        if not present:
            return found, None

        lengths = {c: len(found[c]) for c in present}
        min_len = min(lengths.values())
        max_len = max(lengths.values())

        if min_len != max_len:
            print("WARNING: length mismatch:", lengths, "-> trimming to", min_len)
            for c in present:
                found[c] = found[c][:min_len]

        return found, min_len

def to_dataframe(found):
    # ensure all signals align length-wise
    found, _ = align_lengths(found, ["vdc1","vdc2","idc1","idc2","irr","pvt","f_nv"])

    # If data spans two MAT files, convert each and combine the outputs.
    df = pd.DataFrame()
    # common signals
    for col in ["vdc1","vdc2","idc1","idc2","irr","pvt","f_nv"]:
        if col in found:
            df[col] = found[col]

    # No time column is emitted. Report any time-like variable rather than
    # writing it, so its provenance can be checked before use.
    present_time_vars = [c for c in ("time", "timestamp") if c in found]
    if present_time_vars:
        print("  NOTE: time-like variable(s) found in MAT:", present_time_vars,
              "— NOT written to CSV; verify provenance before any use.")
    else:
        print("  Confirmed: no time variable present in this MAT file.")

    # Positional row counter — an index, not a clock.
    df["sample_index"] = np.arange(len(df), dtype=np.int64)

    # DC power is intentionally NOT computed here. It is derived in
    # fault_preprocessing.py from the four string measurements.
    return df

if __name__ == "__main__":
    print("Raw data folder:", RAW_DATA_DIR)
    CONVERTED_DIR.mkdir(exist_ok=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    print("Converted CSV output folder:", CONVERTED_DIR)

    files = sorted(RAW_DATA_DIR.glob("*.mat"))
    if not files:
        raise SystemExit(f"No .mat files found in raw data folder: {RAW_DATA_DIR}")

    log_parts = []
    for fp in files:
        print("\n--- Loading:", fp.name)
        mat = load_mat_any(str(fp))
        found = pick_vars(mat)
        df = to_dataframe(found)

        # Historical CSVs beside the MAT files are never touched.
        out = CONVERTED_DIR / (fp.stem + ".csv")
        df.to_csv(out, index=False)
        print("Wrote:", out, "| shape:", df.shape)
        print("  Columns:", list(df.columns))

        # Verify sample_index is a 0..n-1 counter and differs from pvt.
        si = df["sample_index"]
        monotonic = bool(si.is_monotonic_increasing and si.iloc[0] == 0
                         and si.iloc[-1] == len(df) - 1
                         and si.diff().iloc[1:].eq(1).all())
        print(f"  sample_index monotonic 0..n-1: {monotonic}")
        if "pvt" in df.columns:
            contaminated = bool(np.allclose(si.to_numpy(dtype=float),
                                            df["pvt"].to_numpy(dtype=float)))
            print(f"  sample_index equals pvt (defect present): {contaminated}")

        log_parts.append(f"{fp.name}->{out.name} rows={len(df):,} "
                         f"cols={list(df.columns)} monotonic={monotonic}")

    with open(RUN_LOG, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} | "
                f"mat_to_csv.py | input: {len(files)} MAT files in {RAW_DATA_DIR} | "
                f"output: {CONVERTED_DIR} | notes: {'; '.join(log_parts)}\n")
    print(f"\nRun log appended: {RUN_LOG}")
