from core.data import loader
from core.data.validation import validate


def print_report(ticker: str) -> None:
    """Print a readable data-quality summary for one ticker's processed data."""
    df = loader.build_processed(ticker)
    r = validate(df)
    gaps = r["gaps"]

    print(f"=== {ticker} ===")
    print(f"Rows:            {r['rows']}")
    print(f"Date range:      {r['start'].date()} -> {r['end'].date()}")
    print(f"Gaps > 4 days:   {len(gaps)}")
    if gaps:
        worst_date = max(gaps, key=gaps.get)
        print(f"Longest gap:     {gaps[worst_date]} days, ending {worst_date.date()}")
    print(f"NaN rows:        {r['nan_rows']}")
    print(f"Duplicate dates: {r['duplicate_dates']}")
    print(f"Zero-volume:     {r['zero_volume_days']} ({r['zero_volume_days'] / r['rows']:.1%})")
    print()


if __name__ == "__main__":
    for t in loader.TICKERS:
        print_report(t)