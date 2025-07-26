from functools import lru_cache
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
import yfinance as yf


@lru_cache(maxsize=64)
def get_price_history(ticker: str, period: str = "max") -> pd.DataFrame:
    """Download historical OHLCV data for the given ticker.

    The result is cached in-memory to avoid repeated downloads during an app
    session. Set *period* to values accepted by yfinance (e.g. "1y", "5y",
    "max").
    """
    df = (
        yf.Ticker(ticker)
        .history(period=period, auto_adjust=False)
        .reset_index()
        .rename(columns={"Date": "date"})
    )
    df.sort_values("date", inplace=True)
    return df


def train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/test chronologically."""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]