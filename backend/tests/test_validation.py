import pandas as pd

from core.data.validation import validate


def make_df(dates, close=None, volume=None):
    """Build a tiny OHLCV frame with a chosen date index."""
    n = len(dates)
    close = close if close is not None else [100.0] * n
    volume = volume if volume is not None else [1000] * n
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": volume},
        index=pd.to_datetime(dates),
    )


def test_clean_data_reports_no_problems():
    df = make_df(["2024-01-02", "2024-01-03", "2024-01-04"])
    r = validate(df)
    assert r["rows"] == 3
    assert r["non_monotonic"] is False
    assert r["duplicate_dates"] == 0
    assert r["nan_rows"] == 0
    assert r["zero_volume_days"] == 0
    assert r["gaps"] == {}


def test_broken_data_is_flagged():
    df = make_df(
        ["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-10", "2024-01-09"],
        close=[None, 100.0, 100.0, 100.0, 100.0],
        volume=[1000, 0, 1000, 1000, 1000],
    )
    r = validate(df)
    assert r["non_monotonic"] is True
    assert r["duplicate_dates"] == 1
    assert r["nan_rows"] == 1
    assert r["zero_volume_days"] == 1
    assert list(r["gaps"].values()) == [7]