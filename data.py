"""Download and cache daily OHLCV data for the ETF universe.

Primary source: yfinance. Fallback: Stooq via pandas-datareader.
Raw data is cached as one CSV per ticker in data/ (gitignored). Nothing is ever
simulated: if both sources fail for a ticker, we stop with instructions.

Usage:  python data.py            # download anything not cached
        python data.py --refresh  # re-download everything
"""
import os
import sys
import time

import pandas as pd

import config

COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _path(ticker):
    return os.path.join(config.DATA_DIR, f"{ticker}.csv")


def _from_yfinance(ticker):
    import yfinance as yf
    # Keep yfinance's timezone cache inside the project; its default shared
    # SQLite cache can raise "database is locked".
    os.makedirs(os.path.join(config.DATA_DIR, ".yf_cache"), exist_ok=True)
    yf.set_tz_cache_location(os.path.join(config.DATA_DIR, ".yf_cache"))
    # auto_adjust=False: we want the traded High/Low/Close (split-adjusted only),
    # plus Adj Close separately for total-return volatility.
    df = yf.download(ticker, start=config.START_DATE, end=config.END_DATE,
                     auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError("empty response")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[COLS]


def _from_stooq(ticker):
    import pandas_datareader.data as web
    df = web.DataReader(f"{ticker}.US", "stooq", start=config.START_DATE,
                        end=config.END_DATE).sort_index()
    if df.empty:
        raise RuntimeError("empty response")
    df["Adj Close"] = df["Close"]  # Stooq prices are already adjusted
    return df[COLS]


def download(ticker, refresh=False):
    path = _path(ticker)
    if os.path.exists(path) and not refresh:
        return
    errors = []
    for name, fn in [("yfinance", _from_yfinance), ("stooq", _from_stooq)]:
        for attempt in range(3):
            try:
                df = fn(ticker)
                df.index.name = "Date"
                df.to_csv(path)
                print(f"  {ticker}: {len(df)} rows from {name}")
                return
            except Exception as e:  # noqa: BLE001 - we report every failure
                errors.append(f"{name} attempt {attempt + 1}: {e}")
                time.sleep(2 * (attempt + 1))
    sys.exit(
        f"Could not download {ticker}.\n  " + "\n  ".join(errors) +
        f"\nDownload the daily OHLCV CSV manually (e.g. from finance.yahoo.com) and "
        f"save it as {path} with columns Date,{','.join(COLS)}."
    )


def download_all(refresh=False):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    missing = [t for t in config.TICKERS if refresh or not os.path.exists(_path(t))]
    if missing:
        print(f"Downloading {len(missing)} tickers...")
    for t in missing:
        download(t, refresh=refresh)


def load(ticker):
    df = pd.read_csv(_path(ticker), index_col="Date", parse_dates=True)
    df = df[COLS].apply(pd.to_numeric, errors="coerce")
    # Drop days with missing prices or no trading (no range information).
    df = df.dropna(subset=["High", "Low", "Close"])
    df = df[(df["Volume"] > 0) & (df["High"] >= df["Low"]) & (df["Low"] > 0)]
    return df


def load_all():
    download_all()
    return {t: load(t) for t in config.TICKERS}


if __name__ == "__main__":
    download_all(refresh="--refresh" in sys.argv)
