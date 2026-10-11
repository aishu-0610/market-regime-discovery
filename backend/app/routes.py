import pandas as pd
from functools import lru_cache

from flask import Blueprint, jsonify, request

from core.features.dataset import TEST_START, TRAIN_END, make_dataset
from storage import load_processed

bp = Blueprint("api", __name__, url_prefix="/api")

# ^NSEI is left out for now: it has no reliable volume, so the volume feature
# can't be computed for it. We'll handle that when we build the asset selector.
ALLOWED_TICKERS = {"SPY", "QQQ"}


@lru_cache(maxsize=8)
def get_dataset(ticker: str):
    df = load_processed(ticker)
    if df is None:
        return None
    return make_dataset(df)


def error(message: str, status: int):
    return jsonify({"error": message}), status


@bp.get("/health")
def health():
    return {"status": "ok"}


@bp.get("/features/<ticker>")
def features(ticker):
    ticker = ticker.upper()
    if ticker not in ALLOWED_TICKERS:
        return error(f"Unsupported ticker '{ticker}'. Use one of {sorted(ALLOWED_TICKERS)}.", 404)

    split = request.args.get("split", "all")
    if split not in ("all", "train", "test"):
        return error("split must be one of: all, train, test", 400)

    d = get_dataset(ticker)
    if d is None:
        return error(f"No processed data cached for {ticker}.", 404)

    if split == "train":
        table = d["train"]
    elif split == "test":
        table = d["test"]
    else:
        table = pd.concat([d["train"], d["test"]])

    return jsonify(
        {
            "ticker": ticker,
            "split": split,
            "rows": len(table),
            "dates": [t.strftime("%Y-%m-%d") for t in table.index],
            "columns": list(table.columns),
            "data": {c: table[c].round(6).tolist() for c in table.columns},
        }
    )


@bp.get("/features/<ticker>/summary")
def features_summary(ticker):
    ticker = ticker.upper()
    if ticker not in ALLOWED_TICKERS:
        return error(f"Unsupported ticker '{ticker}'. Use one of {sorted(ALLOWED_TICKERS)}.", 404)

    d = get_dataset(ticker)
    if d is None:
        return error(f"No processed data cached for {ticker}.", 404)

    def describe(table):
        stats = table.agg(["count", "mean", "std", "min", "max"]).round(6)
        return {col: stats[col].to_dict() for col in stats.columns}

    scaler = d["scaler"]
    columns = list(d["train"].columns)
    return jsonify(
        {
            "ticker": ticker,
            "split": {
                "train_end": TRAIN_END,
                "test_start": TEST_START,
                "train_rows": len(d["train"]),
                "test_rows": len(d["test"]),
            },
            "train": describe(d["train"]),
            "test": describe(d["test"]),
            "scaler": {
                "fit_on": "train only",
                "mean": dict(zip(columns, scaler.mean_.round(6).tolist())),
                "scale": dict(zip(columns, scaler.scale_.round(6).tolist())),
            },
        }
    )