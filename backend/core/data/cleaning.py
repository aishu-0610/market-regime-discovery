import pandas as pd

KEEP_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of raw OHLCV data. The input is never modified."""
    out = df[KEEP_COLS].copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=PRICE_COLS)
    out["volume_missing"] = out["Volume"] == 0
    return out