"""
Front-half of the pipeline (ported from neyen-backend): raw SVFE .dat
posting file -> parsed transactions -> gap detection (rolling K=150
neighborhood, Bonferroni z, 3-sec floor) -> confidence scoring ->
feature rows shaped exactly like app.models.RawTransaction, so they can
go straight into load_raw_transactions() / the existing
/transactions/bulk and /transactions/upload-csv endpoints.

This fills the gap DSIP-BACKEND didn't have: everything upstream of
"already gap-labeled rows in RawTransaction shape". app/pipeline.py
(hourly windows + prediction) picks up from there -- unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

MIN_BODY_LENGTH = 928
K_NEIGHBORS = 150
TARGET_FALSE_POS_RATE = 0.001
TIMESTAMP_RESOLUTION_SEC = 3.0
ACTIVE_HOURS_START = 7   # ATM open, per handoff.md operating context
ACTIVE_HOURS_END = 20    # ATM close

FIELDS = [
    ("universal_transaction_number",  102, 110),
    ("reversal_flag",                 111, 111),
    ("svfe_trace_number",             112, 120),
    ("svfe_message_type",             121, 124),
    ("svfe_system_time",              126, 131),
    ("svfe_system_date",              132, 135),
    ("response_code",                 312, 313),
    ("svfe_transaction_type",         775, 777),
    ("svfe_response_code",            780, 782),
    ("completion_status",             879, 879),
    ("stood_in_for",                  880, 880),
    ("issuer_posted",                 881, 881),
]


def _parse_body_line(line: str) -> dict:
    return {name: line[start - 1:end].strip() for name, start, end in FIELDS}


def parse_raw_dat(input_path: str, target_date: str | None = None) -> pd.DataFrame:
    """Raw fixed-width .dat -> clean transactions DataFrame (string columns)."""
    body_rows = []
    with open(input_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            marker = line[0]
            if marker == "1" or marker == "@":
                continue
            if len(line) < MIN_BODY_LENGTH:
                continue
            body_rows.append(_parse_body_line(line))

    df = pd.DataFrame(body_rows)
    blank_utrn = df["universal_transaction_number"].isna() | (df["universal_transaction_number"] == "")
    df = df[~blank_utrn]
    df = df.drop_duplicates(subset=["svfe_system_date", "universal_transaction_number", "reversal_flag"])
    if target_date is not None:
        df = df[df["svfe_system_date"] == target_date]
    return df.reset_index(drop=True)


def _local_ceiling_and_expected(gap_durations: np.ndarray, k_neighbors: int = K_NEIGHBORS,
                                 timestamp_resolution_sec: float = TIMESTAMP_RESOLUTION_SEC,
                                 target_fp_rate: float = TARGET_FALSE_POS_RATE):
    n = len(gap_durations)
    neighborhood_size = 2 * k_neighbors - 1
    alpha = target_fp_rate / neighborhood_size
    z = norm.ppf(1 - alpha)

    cumsum = np.concatenate([[0.0], np.cumsum(gap_durations)])
    ceilings = np.full(n, np.nan)
    expected = np.full(n, np.nan)
    flagged = np.zeros(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - k_neighbors)
        hi = min(n, i + k_neighbors + 1)
        neighborhood_sum = (cumsum[hi] - cumsum[lo]) - gap_durations[i]
        neighborhood_count = (hi - lo) - 1
        if neighborhood_count < 10:
            continue
        m1 = neighborhood_sum / neighborhood_count
        st2 = m1 / np.sqrt(2 / np.pi)
        ceiling = max(m1 + z * st2, timestamp_resolution_sec)
        ceilings[i] = ceiling
        expected[i] = m1
        if gap_durations[i] > ceiling:
            flagged[i] = True

    return ceilings, expected, flagged, z


def _score_confidence_for_flagged(df: pd.DataFrame, flagged: np.ndarray) -> np.ndarray:
    """
    Exact cross-validation window from the original algorithm: for each
    flagged gap, look at every transaction within 1 sec of the gap's
    start/end (not just the two bordering rows -- a busy gap boundary can
    have several transactions within that window). Only run for flagged
    gaps since they're sparse (tens out of millions of rows), so this
    stays cheap despite being a per-gap scan.
    """
    n = len(df)
    tier = np.full(n, "N/A", dtype=object)
    flagged_idx = np.where(flagged)[0]
    if len(flagged_idx) == 0:
        return tier

    ts = df["ts"].values
    completion = df["completion_status"].values
    stood_in = df["stood_in_for"].values
    issuer_posted = df["issuer_posted"].values
    resp_code = df["svfe_response_code"].values
    one_sec = np.timedelta64(1, "s")

    for i in flagged_idx:
        gap_start, gap_end = ts[i], ts[i + 1]
        mask = (ts >= gap_start - one_sec) & (ts <= gap_end + one_sec)
        sig1 = np.any(completion[mask] == "0")
        sig2 = np.any(stood_in[mask] == "1")
        sig3 = np.any((issuer_posted[mask] == "0") & (stood_in[mask] == "1"))
        sig4 = np.any(np.isin(resp_code[mask], ["801", "802"]))
        score = int(sig1) + int(sig2) + int(sig3) + int(sig4)
        tier[i] = "Low" if score == 0 else ("Medium" if score <= 2 else "High")

    return tier


def build_raw_transaction_rows(transactions_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Runs gap detection + confidence scoring, then returns a DataFrame with
    exactly the columns app.models.RawTransaction expects -- ready for
    load_raw_transactions() or a straight .to_dict(orient="records").
    """
    df = transactions_df.copy()
    date_raw = df["svfe_system_date"].astype(str).str.strip().str.zfill(4)
    time_raw = df["svfe_system_time"].astype(str).str.strip().str.zfill(6)
    df["ts"] = pd.to_datetime(str(year) + date_raw + time_raw, format="%Y%m%d%H%M%S", errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(f"Only {len(df)} rows with valid timestamps -- not enough to run gap detection.")

    ts_array = df["ts"].values
    gap_durations = np.diff(ts_array).astype("timedelta64[ms]").astype(float) / 1000.0
    gap_durations = np.append(gap_durations, np.nan)  # last transaction has no following gap

    ceilings, expected, flagged, _ = _local_ceiling_and_expected(gap_durations[:-1])
    ceilings = np.append(ceilings, np.nan)
    expected = np.append(expected, np.nan)
    flagged = np.append(flagged, False)

    df["gap_to_next_sec"] = gap_durations
    df["local_ceiling_sec"] = ceilings
    df["local_expected_gap_sec"] = expected
    with np.errstate(divide="ignore", invalid="ignore"):
        df["duration_to_expected_ratio"] = np.where(
            (expected == 0) | np.isnan(expected), np.nan, gap_durations / expected
        )
    df["gap_label"] = np.where(flagged, "Candidate Downtime", "Normal Silence")

    df["hour_of_day"] = df["ts"].dt.hour.astype(float)
    df["day_of_week"] = df["ts"].dt.dayofweek.astype(float)
    df["is_active_hours"] = df["hour_of_day"].between(ACTIVE_HOURS_START, ACTIVE_HOURS_END - 1)
    minutes_of_day = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["minutes_from_open"] = minutes_of_day - ACTIVE_HOURS_START * 60
    df["minutes_from_close"] = ACTIVE_HOURS_END * 60 - minutes_of_day

    df["confidence_tier"] = _score_confidence_for_flagged(df, flagged)

    # Cast to the types RawTransaction expects.
    df["reversal_flag"] = pd.to_numeric(df["reversal_flag"], errors="coerce").astype("Int64")
    df["completion_status"] = pd.to_numeric(df["completion_status"], errors="coerce").astype("Int64")
    df["stood_in_for"] = pd.to_numeric(df["stood_in_for"], errors="coerce").astype("Int64")
    df["issuer_posted"] = pd.to_numeric(df["issuer_posted"], errors="coerce").astype("Int64")

    keep_cols = [
        "universal_transaction_number", "reversal_flag", "svfe_trace_number",
        "svfe_message_type", "response_code", "svfe_transaction_type",
        "svfe_response_code", "completion_status", "stood_in_for", "issuer_posted",
        "ts", "gap_to_next_sec", "gap_label", "local_expected_gap_sec",
        "local_ceiling_sec", "duration_to_expected_ratio", "hour_of_day",
        "day_of_week", "is_active_hours", "minutes_from_open", "minutes_from_close",
        "confidence_tier",
    ]
    return df[keep_cols].reset_index(drop=True)
