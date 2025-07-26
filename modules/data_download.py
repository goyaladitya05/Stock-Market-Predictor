import yfinance as yf
import pandas as pd


def fetch_data(ticker, start='2023-01-01', end='2023-07-01', auto_adjust=True):
    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0]}_{ticker}" for col in df.columns]

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])

    # Rename 'Close_<ticker>' to 'Close'
    close_col = f"Close_{ticker}" if f"Close_{ticker}" in df.columns else "Close"
    df.rename(columns={close_col: "Close"}, inplace=True)

    return df
