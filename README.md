# NIFTY-50 Stock Market Predictor 📈

An end-to-end, modular Python app for interactive stock-price prediction on NIFTY-50 equities.  The UI is powered by **Streamlit** and lets a user:

1. Pick any company from the NIFTY-50 list (auto-updated from Yahoo Finance tickers).
2. Configure hyper-parameters and choose among multiple prediction models (ARIMA, Prophet, LSTM, Random-Forest*)
3. Visualise historical prices as well as model forecasts, error metrics and side-by-side model comparisons.

> *Random-Forest & LSTM demos are provided as optional advanced notebooks – Prophet & ARIMA are fully wired in the Streamlit front-end.*

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run will download price histories via Yahoo Finance; subsequent runs are cached.

## Project Structure

```
.
├── app.py                # Streamlit UI
├── stockapp/
│   ├── __init__.py
│   ├── nifty50_symbols.py
│   ├── data_loader.py
│   ├── evaluator.py
│   └── models/
│       ├── __init__.py
│       ├── base_model.py
│       ├── arima_model.py
│       ├── prophet_model.py
│       └── lstm_model.py     # optional, heavy dep on TensorFlow
└── requirements.txt
```

## Screenshots
Add screenshots of the app after you explore it! 🌄

## Roadmap / Ideas
- Hyper-parameter search (optuna)
- Model ensemble benchmark
- Dockerfile + CI deploy to Streamlit Cloud

---
Created for my personal portfolio – feel free to fork & build upon it! ✨
