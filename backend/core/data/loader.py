from datetime import date
from core.data.cleaning import clean
import pandas as pd
import yfinance as yf
import storage

TICKERS = ["SPY", "QQQ", "^NSEI"]
START = "2005-01-01"


def download(ticker: str, start: str = START) -> pd.DataFrame:
    """Download daily bars from Yahoo. This is the only function that touches the network."""
    df = yf.Ticker(ticker).history(
        start=start,
        end=date.today().isoformat(),
        auto_adjust=False,
    )
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "Date"
    return df


def get_raw(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Return raw data, downloading only if it isn't cached (or refresh=True)."""
    if not refresh:
        cached = storage.load_raw(ticker)
        if cached is not None:
            return cached
    df = download(ticker)
    storage.save_raw(ticker, df)
    return df


def build_processed(ticker: str) -> pd.DataFrame:
    """Raw -> clean -> saved in data/processed. Returns the clean frame."""
    df = clean(get_raw(ticker))
    storage.save_processed(ticker, df)
    return df