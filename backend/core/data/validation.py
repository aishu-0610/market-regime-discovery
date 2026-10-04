import pandas as pd

PRICE_COLS = ["Open", "High", "Low", "Close"]
MAX_GAP_DAYS = 4  # Fri -> Tue (Monday holiday) is 4 days; more than that is suspicious


def validate(df: pd.DataFrame) -> dict:
    """Check a daily OHLCV DataFrame and return a report. Does not modify the data."""
    gaps = df.index.to_series().diff().dt.days
    return {
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "non_monotonic": not df.index.is_monotonic_increasing,
        "duplicate_dates": int(df.index.duplicated().sum()),
        "nan_rows": int(df[PRICE_COLS + ["Volume"]].isna().any(axis=1).sum()),
        "zero_volume_days": int((df["Volume"] == 0).sum()),
        "gaps": gaps[gaps > MAX_GAP_DAYS].astype(int).to_dict(),
    }