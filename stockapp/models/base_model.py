from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class BaseModel(ABC):
    """Abstract wrapper so Streamlit UI can treat models uniformly."""

    name: str  # Human-readable

    def __init__(self, **params):
        self.params = params
        self.fitted = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame):
        """Fit the model using training dataframe with a `date` and `Close` columns."""

    @abstractmethod
    def predict(self, periods: int) -> pd.Series:
        """Generate `periods` predictions ahead, returning a Series indexed by date."""

    @abstractmethod
    def forecast(self, test_df: pd.DataFrame) -> pd.Series:
        """Forecast for the dates present in *test_df* (walk-forward)."""

    def get_params(self) -> Dict:
        return self.params

    # Optional hook for cleanup
    def release(self):
        pass