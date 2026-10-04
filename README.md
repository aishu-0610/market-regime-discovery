# Market Regime Discovery

Markets don't behave the same way all the time: calm uptrends, volatile
selloffs, and sideways stretches each have different return and risk
characteristics. Most beginner models try to predict tomorrow's price, but
daily returns are mostly noise, so those models tend to overfit and fail on
new data. This project asks a more answerable question: what state is the
market in right now?

It applies unsupervised learning (K-Means and Gaussian Mixture Models) to
features built from daily OHLCV data for SPY, with prices adjusted for
splits and dividends. K-Means assigns each day to a regime, and the GMM
gives the probability that each day belongs to each regime. The regimes are
then interpreted from their average return, volatility, and volume, and
tested by whether they improve risk-adjusted performance in a backtest
against Buy & Hold, with models fitted only on earlier data to avoid
look-ahead leakage.

The project discovers and interprets regimes. It does not predict prices.

## Tech stack
Python, Flask, pandas, scikit-learn, yfinance · React, Vite, Tailwind CSS

## Data
Daily OHLCV for SPY, QQQ and the Nifty 50 index (^NSEI), downloaded once from
yfinance and cached locally as Parquet. Each dataset is validated (NaNs,
duplicate dates, gaps, zero volume) and cleaned before use. ^NSEI has no
reliable volume data, so those days are flagged and volume features skip them.

## Status
Work in progress (21-day MVP). Done so far: project skeleton (Flask API with a
health check, React frontend connected through a dev proxy) and the data layer
(loader, validation, caching, unit tests).