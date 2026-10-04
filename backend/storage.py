from pathlib import Path
import pandas as pd

# backend/storage.py -> parents[0] is backend/, parents[1] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _ensure_dirs() -> None:
    """Create data folders if missing (git does not track empty folders)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _filename(ticker: str) -> str:
    """'^NSEI' -> 'NSEI.parquet' so the file name is safe on every OS."""
    return ticker.replace("^", "") + ".parquet"

def save_raw(ticker: str, df: pd.DataFrame) -> Path:
    """Save the untouched download for a ticker."""
    _ensure_dirs()
    path = RAW_DIR / _filename(ticker)
    df.to_parquet(path)
    return path


def load_raw(ticker: str) -> pd.DataFrame | None:
    """Return the cached raw data, or None if we never downloaded it."""
    path = RAW_DIR / _filename(ticker)
    return pd.read_parquet(path) if path.exists() else None


def save_processed(ticker: str, df: pd.DataFrame) -> Path:
    """Save the cleaned, validated data for a ticker."""
    _ensure_dirs()
    path = PROCESSED_DIR / _filename(ticker)
    df.to_parquet(path)
    return path


def load_processed(ticker: str) -> pd.DataFrame | None:
    """Return the cleaned data, or None if it doesn't exist yet."""
    path = PROCESSED_DIR / _filename(ticker)
    return pd.read_parquet(path) if path.exists() else None