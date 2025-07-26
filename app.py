import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px

from stockapp.nifty50_symbols import get_company_list, get_symbol
from stockapp.data_loader import get_price_history, train_test_split
from stockapp.models import ARIMAModel, ProphetModel, LSTMModel
from stockapp.evaluator import Evaluator

# -------------------- Sidebar --------------------
st.title("📈 NIFTY-50 Stock Predictor")

st.sidebar.header("Configuration")
company = st.sidebar.selectbox("Select Company", get_company_list())
model_name = st.sidebar.selectbox(
    "Prediction Model",
    (
        ARIMAModel.name,
        ProphetModel.name,
        LSTMModel.name,
    ),
)
train_ratio = st.sidebar.slider("Train Split %", 50, 95, 80)
period = st.sidebar.selectbox("History Period", ["1y", "2y", "5y", "max"], index=3)

# -------------------- Load Data --------------------

ticker = get_symbol(company)
with st.spinner("Downloading price history…"):
    df = get_price_history(ticker, period=period)

st.subheader(f"{company} ({ticker}) Closing Price")
fig = px.line(df, x="date", y="Close", title="Historical Close Price")
st.plotly_chart(fig, use_container_width=True)

# Split train/test
a_train, a_test = train_test_split(df, train_ratio / 100.0)

# -------------------- Model Instantiation --------------------
model_map = {
    ARIMAModel.name: ARIMAModel(seasonal=False),
    ProphetModel.name: ProphetModel(),
    LSTMModel.name: LSTMModel(epochs=5),
}
model = model_map[model_name]

if st.sidebar.button("Run Prediction"):
    with st.spinner("Training …"):
        model.fit(a_train)
    with st.spinner("Forecasting …"):
        preds = model.forecast(a_test)

    # Evaluate & Visualise
    forecasts = {model.name: preds}
    evaluator = Evaluator(a_test, forecasts)

    st.subheader("Predicted vs Actual")
    fig1 = evaluator.plot_forecasts()
    st.pyplot(fig1)

    st.subheader("Error Metrics")
    metrics_df = evaluator.compute_metrics()
    st.table(metrics_df)

    st.subheader("Error Bar")
    fig2 = evaluator.plot_error_bar()
    st.pyplot(fig2)

# -------------------- Multi-model Benchmarks --------------------
st.markdown("---")
st.header("🔬 Compare Multiple Models (Quick Benchmark)")
if st.button("Run All Models"):
    forecasts = {}
    for m in model_map.values():
        with st.spinner(f"Fitting {m.name} …"):
            m.fit(a_train)
            preds = m.forecast(a_test)
            forecasts[m.name] = preds
    evaluator = Evaluator(a_test, forecasts)
    st.pyplot(evaluator.plot_forecasts())
    st.pyplot(evaluator.plot_error_bar())
    st.table(evaluator.compute_metrics())