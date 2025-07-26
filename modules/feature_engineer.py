import ta
import pandas as pd

def add_features(df, close_col='Close'):
    close_series = pd.Series(df[close_col].values.ravel(), index=df.index)

    df['MA5'] = close_series.rolling(5).mean()
    df['MA10'] = close_series.rolling(10).mean()
    df['MA20'] = close_series.rolling(20).mean()
    df['Return'] = close_series.pct_change()

    df['RSI14'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()

    macd = ta.trend.MACD(close=close_series)
    df['MACD'] = macd.macd().squeeze()
    df['MACD_signal'] = macd.macd_signal().squeeze()

    boll = ta.volatility.BollingerBands(close=close_series)
    df['Bollinger_Mavg'] = boll.bollinger_mavg().squeeze()
    df['Bollinger_High'] = boll.bollinger_hband().squeeze()
    df['Bollinger_Low'] = boll.bollinger_lband().squeeze()

    df['Target'] = (close_series.shift(-1) > close_series).astype(int)

    return df

def clean_data(df):
    tech_cols = ['MA5', 'MA10', 'MA20', 'Return', 'RSI14', 'MACD', 'MACD_signal',
                 'Bollinger_Mavg', 'Bollinger_High', 'Bollinger_Low']
    for col in tech_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='bfill')
    df['Return'] = df['Return'].fillna(0)
    return df.dropna(subset=['Target'])
