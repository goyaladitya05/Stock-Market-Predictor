from typing import Optional

import pandas as pd
import numpy as np
from pmdarima import auto_arima

from .base_model import BaseModel


class ARIMAModel(BaseModel):
    name = "ARIMA (auto_arima)"

    def __init__(self, seasonal: bool = False, m: int = 1):
        super().__init__(seasonal=seasonal, m=m)
        self.model: Optional[auto_arima] = None

    def fit(self, train_df: pd.DataFrame):
        y = train_df["Close"].values
        self.model = auto_arima(
            y,
            seasonal=self.params["seasonal"],
            m=self.params["m"],
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        self.fitted = True

    def predict(self, periods: int) -> pd.Series:
        assert self.fitted and self.model is not None
        forecast = self.model.predict(n_periods=periods)
        # index placeholder, actual index should be set by caller
        return pd.Series(forecast)

    def forecast(self, test_df: pd.DataFrame) -> pd.Series:
        preds = self.predict(len(test_df))
        preds.index = test_df["date"].values
        return preds