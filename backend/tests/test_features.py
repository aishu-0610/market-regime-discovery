import numpy as np
import pandas as pd

from core.features.engineering import log_return_1d, return_20d, annualized_volatility,distance_from_ma, volume_zscore, drawdown_from_high
from core.features.engineering import build_features

def make_prices():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    return pd.Series([100.0, 102.0, 101.0, 105.0, 107.0, 106.0], index=idx)


def test_first_value_is_nan():
    r = log_return_1d(make_prices())
    assert np.isnan(r.iloc[0])
    assert r.iloc[1:].notna().all()


def test_known_value():
    r = log_return_1d(make_prices())
    assert np.isclose(r.iloc[1], np.log(102 / 100))


def test_no_lookahead():
    p = make_prices()
    before = log_return_1d(p)
    p_changed = p.copy()
    p_changed.iloc[-1] = 999.0  # change only the LAST day
    after = log_return_1d(p_changed)
    # every earlier day must be unaffected by a future price change
    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])


def test_missing_value_only_affects_neighbors():
    p = make_prices()
    p.iloc[3] = np.nan  # one missing price
    r = log_return_1d(p)
    # the missing day and the day after it can't be computed; the rest can
    assert r.iloc[3:5].isna().all()
    assert r.iloc[1:3].notna().all()
    assert r.iloc[5:].notna().all()


def make_long_prices(n=40):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.linspace(100.0, 140.0, n), index=idx)


def test_return_20d_warmup_and_value():
    p = make_long_prices()
    r = return_20d(p)
    assert r.iloc[:20].isna().all()
    assert r.iloc[20:].notna().all()
    assert np.isclose(r.iloc[20], np.log(p.iloc[20] / p.iloc[0]))


def test_return_20d_no_lookahead():
    p = make_long_prices()
    before = return_20d(p)
    p_changed = p.copy()
    p_changed.iloc[-1] = 999.0
    after = return_20d(p_changed)
    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])


def make_noisy_prices(n=100):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def test_volatility_warmup():
    v = annualized_volatility(make_noisy_prices(), 20)
    # 1 NaN from the first return + 19 more until the window is full = 20
    assert v.iloc[:20].isna().all()
    assert v.iloc[20:].notna().all()


def test_volatility_known_value():
    p = make_noisy_prices()
    v = annualized_volatility(p, 20)
    daily = np.log(p / p.shift(1))
    expected = daily.iloc[1:21].std() * np.sqrt(252)
    assert np.isclose(v.iloc[20], expected)


def test_volatility_no_lookahead():
    p = make_noisy_prices()
    before = annualized_volatility(p, 60)
    p_changed = p.copy()
    p_changed.iloc[-1] = 999.0
    after = annualized_volatility(p_changed, 60)
    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])


def test_distance_from_ma_warmup():
    p = make_noisy_prices(100)
    d = distance_from_ma(p, 50)
    assert d.iloc[:49].isna().all()
    assert d.iloc[49:].notna().all()


def test_distance_from_ma_known_value():
    p = make_noisy_prices(100)
    d = distance_from_ma(p, 50)
    expected = p.iloc[49] / p.iloc[0:50].mean() - 1
    assert np.isclose(d.iloc[49], expected)


def test_distance_from_ma_no_lookahead():
    p = make_noisy_prices(100)
    before = distance_from_ma(p, 50)
    p_changed = p.copy()
    p_changed.iloc[-1] = 999.0
    after = distance_from_ma(p_changed, 50)
    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])


def make_volume(n=100):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(rng.integers(1_000_000, 5_000_000, n), index=idx)


def test_volume_zscore_warmup():
    z = volume_zscore(make_volume(100), 60)
    assert z.iloc[:59].isna().all()
    assert z.iloc[59:].notna().all()


def test_volume_zscore_known_value():
    v = make_volume(100)
    z = volume_zscore(v, 60)
    window = v.iloc[0:60].astype(float)
    expected = (window.iloc[-1] - window.mean()) / window.std()
    assert np.isclose(z.iloc[59], expected)


def test_volume_zscore_no_lookahead():
    v = make_volume(100)
    before = volume_zscore(v, 60)
    v_changed = v.copy()
    v_changed.iloc[-1] = 999_999_999
    after = volume_zscore(v_changed, 60)
    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])


def test_volume_zscore_respects_missing_flag():
    v = make_volume(100)
    missing = pd.Series(False, index=v.index)
    missing.iloc[70] = True
    z = volume_zscore(v, 60, missing)
    assert np.isnan(z.iloc[70])
    assert z.iloc[69] == z.iloc[69]  # day before the flagged day is still valid


def test_drawdown_warmup():
    p = make_noisy_prices(300)
    d = drawdown_from_high(p, 252)
    assert d.iloc[:251].isna().all()
    assert d.iloc[251:].notna().all()


def test_drawdown_never_positive():
    p = make_noisy_prices(300)
    d = drawdown_from_high(p, 252).dropna()
    assert (d <= 1e-12).all()


def test_drawdown_known_value():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    p = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0], index=idx)
    d = drawdown_from_high(p, 3)
    # window ending at index 4 is [90, 110, 80]; high = 110
    assert np.isclose(d.iloc[4], 80 / 110 - 1)
    # at index 2 the window is [100, 120, 90]; high = 120
    assert np.isclose(d.iloc[2], 90 / 120 - 1)


def test_drawdown_no_lookahead():
    p = make_noisy_prices(300)
    before = drawdown_from_high(p, 252)
    p_changed = p.copy()
    p_changed.iloc[-1] = 999.0
    after = drawdown_from_high(p_changed, 252)
    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])



def make_ohlcv(n=300):
    p = make_noisy_prices(n)
    return pd.DataFrame(
        {
            "Adj Close": p,
            "Volume": make_volume(n),
            "volume_missing": pd.Series(False, index=p.index),
        }
    )


def test_build_features_shape_and_columns():
    f = build_features(make_ohlcv(300))
    assert list(f.columns) == [
        "ret_1d", "ret_20d", "vol_20d", "vol_60d",
        "dist_ma50", "volume_z60", "drawdown_252d",
    ]
    assert len(f) == 300


def test_build_features_no_lookahead():
    df = make_ohlcv(300)
    before = build_features(df)
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("Adj Close")] = 999.0
    df2.iloc[-1, df2.columns.get_loc("Volume")] = 999_999_999
    after = build_features(df2)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])