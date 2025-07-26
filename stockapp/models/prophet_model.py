from typing import Optional

import pandas as pd
from prophet import Prophet

from .base_model import BaseModel


class ProphetModel(BaseModel):
    name = "Facebook Prophet"

    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True):
        super().__init__(yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality)
        self.model: Optional[Prophet] = None

    def fit(self, train_df: pd.DataFrame):
        df = train_df.rename(columns={"date": "ds", "Close": "y"})[["ds", "y"]]
        self.model = Prophet(
            yearly_seasonality=self.params["yearly_seasonality"],
            weekly_seasonality=self.params["weekly_seasonality"],
            daily_seasonality=False,
        )
        self.model.fit(df)
        self.fitted = True

    def predict(self, periods: int) -> pd.Series:
        assert self.fitted and self.model is not None
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast[-periods:]["yhat"].reset_index(drop=True)

    def forecast(self, test_df: pd.DataFrame) -> pd.Series:
        df = test_df.rename(columns={"date": "ds"})[["ds"]]
        pred_df = self.model.predict(df)
        preds = pred_df["yhat"].copy()
        preds.index = test_df["date"].values
        return preds