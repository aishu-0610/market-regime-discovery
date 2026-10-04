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

## Status
Work in progress. Day 1: project skeleton (Flask API with a health check,
React frontend connected through a dev proxy).