import numpy as np
import pandas as pd

from core.features.dataset import make_dataset


def make_ohlcv_2015_2020():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", "2020-12-31")
    n = len(idx)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "Adj Close": price,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
            "volume_missing": False,
        },
        index=idx,
    )


def test_no_nans_and_no_overlap():
    d = make_dataset(make_ohlcv_2015_2020())
    for key in ["train", "test", "train_scaled", "test_scaled"]:
        assert not d[key].isna().any().any()
    assert d["train"].index.max() <= pd.Timestamp("2017-12-31")
    assert d["test"].index.min() >= pd.Timestamp("2018-01-01")
    assert d["train"].index.max() < d["test"].index.min()


def test_train_scaled_is_standardized():
    d = make_dataset(make_ohlcv_2015_2020())
    assert np.allclose(d["train_scaled"].mean(), 0, atol=1e-9)
    assert np.allclose(d["train_scaled"].std(), 1, atol=0.01)


def test_scaler_fit_on_train_only():
    d = make_dataset(make_ohlcv_2015_2020())
    assert np.allclose(d["scaler"].mean_, d["train"].mean().values)


def test_changing_test_data_does_not_change_scaler():
    df = make_ohlcv_2015_2020()
    before = make_dataset(df)
    df2 = df.copy()
    df2.loc["2020-06-01":, "Adj Close"] *= 3.0  # distort only late test-period prices
    after = make_dataset(df2)
    assert np.allclose(before["scaler"].mean_, after["scaler"].mean_)
    assert np.allclose(before["scaler"].scale_, after["scaler"].scale_)