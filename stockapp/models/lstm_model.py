from typing import Optional

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

from .base_model import BaseModel


class LSTMModel(BaseModel):
    name = "LSTM (Keras)"

    def __init__(self, look_back: int = 60, epochs: int = 10):
        super().__init__(look_back=look_back, epochs=epochs)
        self.scaler: Optional[MinMaxScaler] = None
        self.model: Optional[Sequential] = None

    def _create_sequences(self, data: np.ndarray) -> tuple:
        X, y = [], []
        lb = self.params["look_back"]
        for i in range(lb, len(data)):
            X.append(data[i - lb : i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, train_df: pd.DataFrame):
        close_prices = train_df["Close"].values.reshape(-1, 1)
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(close_prices)

        X, y = self._create_sequences(scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        self.model = Sequential(
            [
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50),
                Dense(1),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(X, y, epochs=self.params["epochs"], verbose=0)
        self.fitted = True

    def predict(self, periods: int) -> pd.Series:
        raise NotImplementedError("Direct multi-step prediction not implemented; use forecast.")

    def forecast(self, test_df: pd.DataFrame) -> pd.Series:
        assert self.fitted
        data = np.concatenate([
            self.scaler.transform(test_df[["Close"]].values.reshape(-1, 1))
        ])
        # For simplicity, naive walk-forward (not updating weights)
        lb = self.params["look_back"]
        preds_scaled = []
        for i in range(lb, len(data)):
            seq = data[i - lb : i].reshape((1, lb, 1))
            pred = self.model.predict(seq, verbose=0)[0]
            preds_scaled.append(pred)
        preds_scaled = np.array(preds_scaled)
        preds = self.scaler.inverse_transform(preds_scaled)[:, 0]
        # align index
        pred_series = pd.Series(preds, index=test_df["date"].values[lb:])
        return pred_series