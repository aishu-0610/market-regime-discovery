import numpy as np
import pandas as pd


def log_return_1d(price: pd.Series) -> pd.Series:
    """1-day log return: ln(price today / price yesterday)."""
    return np.log(price / price.shift(1))


def return_20d(price: pd.Series) -> pd.Series:
    """20-day log return: ln(price today / price 20 trading days ago)."""
    return np.log(price / price.shift(20))

def annualized_volatility(price: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of daily log returns, scaled to a yearly figure."""
    daily = log_return_1d(price)
    return daily.rolling(window=window, min_periods=window).std() * np.sqrt(252)

def distance_from_ma(price: pd.Series, window: int = 50) -> pd.Series:
    """How far price is above/below its rolling average, as a fraction of that average."""
    ma = price.rolling(window=window, min_periods=window).mean()
    return price / ma - 1

def volume_zscore(
    volume: pd.Series, window: int = 60, missing: pd.Series | None = None
) -> pd.Series:
    """How unusual today's volume is vs the last `window` days, in standard deviations."""
    vol = volume.astype(float)
    if missing is not None:
        vol = vol.where(~missing)  # flagged days become NaN
    mean = vol.rolling(window=window, min_periods=window).mean()
    std = vol.rolling(window=window, min_periods=window).std()
    return (vol - mean) / std

def drawdown_from_high(price: pd.Series, window: int = 252) -> pd.Series:
    """How far price is below its highest price of the last `window` days (0 = at the high)."""
    rolling_high = price.rolling(window=window, min_periods=window).max()
    return price / rolling_high - 1




def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the seven agreed features from a processed OHLCV DataFrame."""
    price = df["Adj Close"]
    missing = df["volume_missing"] if "volume_missing" in df.columns else None
    out = pd.DataFrame(index=df.index)
    out["ret_1d"] = log_return_1d(price)
    out["ret_20d"] = return_20d(price)
    out["vol_20d"] = annualized_volatility(price, 20)
    out["vol_60d"] = annualized_volatility(price, 60)
    out["dist_ma50"] = distance_from_ma(price, 50)
    out["volume_z60"] = volume_zscore(df["Volume"], 60, missing)
    out["drawdown_252d"] = drawdown_from_high(price, 252)
    return out