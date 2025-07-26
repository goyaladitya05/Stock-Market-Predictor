from typing import Dict, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Evaluator:
    """Evaluate prediction series vs ground truth and plot comparisons."""

    def __init__(self, test_df: pd.DataFrame, forecasts: Dict[str, pd.Series]):
        self.test_df = test_df
        self.forecasts = forecasts  # model_name -> pred Series

    def compute_metrics(self) -> pd.DataFrame:
        rows = []
        y_true = self.test_df.set_index("date")["Close"]
        for name, preds in self.forecasts.items():
            # align
            aligned = preds.reindex_like(y_true)
            mae = mean_absolute_error(y_true, aligned)
            rmse = mean_squared_error(y_true, aligned, squared=False)
            rows.append({"model": name, "MAE": mae, "RMSE": rmse})
        return pd.DataFrame(rows)

    def plot_forecasts(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.test_df["date"], self.test_df["Close"], label="Actual", linewidth=2)
        for name, preds in self.forecasts.items():
            plt.plot(preds.index, preds.values, label=name)
        plt.legend()
        plt.title("Actual vs Forecasted Prices")
        plt.tight_layout()
        return plt

    def plot_error_bar(self):
        df = self.compute_metrics()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df.melt(id_vars="model", var_name="metric", value_name="value"), x="model", y="value", hue="metric", ax=ax)
        ax.set_title("Error Metrics by Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig