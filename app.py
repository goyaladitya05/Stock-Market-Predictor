import streamlit as st
import pandas as pd
from datetime import date
from modules.data_download import fetch_data
from modules.feature_engineer import add_features, clean_data
from modules.model import prepare_data, train_and_evaluate, get_model_options
from modules.visualization import plot_eda, plot_corr_heatmap, plot_moving_averages, plot_predictions
import matplotlib.pyplot as plt
import base64

# Streamlit Page Config
st.set_page_config(page_title="NIFTY50 Stock Dashboard", layout="wide")

# Set dark theme only
bg_color = "#111111"
text_color = "#ffffff"
tab_color = "#333333"
tab_selected_color = "#1f77b4"
selected_text = "#ffffff"

st.markdown(f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .main {{
            background-color: {bg_color};
        }}
        h1, h2, h3 {{
            color: {text_color};
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {tab_color};
            padding: 10px;
            border-radius: 6px;
            margin-right: 8px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {tab_selected_color};
            color: {selected_text};
        }}
        .logo-container {{
            text-align: center;
            margin-bottom: 10px;
        }}
        .logo-container img {{
            height: 60px;
        }}
    </style>
""", unsafe_allow_html=True)

# Load and display logo
logo_url = "https://upload.wikimedia.org/wikipedia/commons/1/1b/NSE_Logo.svg"
st.markdown(f'<div class="logo-container"><img src="{logo_url}" alt="Logo"></div>', unsafe_allow_html=True)

# App Title
st.title("📈 NIFTY50 Stock Price Movement Prediction Dashboard")

# Load NIFTY50 symbols
@st.cache_data
def load_nifty50_list():
    return pd.read_csv("data/nifty50_list.csv")['Symbol'].tolist()

symbols = load_nifty50_list()

# Search bar
search = st.sidebar.text_input("🔍 Search Stock", "")
filtered_symbols = [s for s in symbols if search.upper() in s.upper()]

if not filtered_symbols:
    st.sidebar.warning("No matching stocks found.")
    st.stop()

ticker = st.sidebar.selectbox("📌 Select NIFTY50 Stock", filtered_symbols, index=0)
max_date = date.today()
start_date = st.sidebar.date_input("📅 Start Date", pd.to_datetime("2023-01-01"), max_value=max_date)
end_date = st.sidebar.date_input("📅 End Date", pd.to_datetime("2023-07-01"), max_value=max_date)

# Load and process data
try:
    raw_data = fetch_data(ticker, start=start_date, end=end_date)
    if f"Close_{ticker}" in raw_data.columns:
        raw_data.rename(columns={f"Close_{ticker}": "Close"}, inplace=True)

    if 'Close' not in raw_data.columns:
        st.write("Data preview:")
        st.dataframe(raw_data.head())
        raise ValueError("Missing 'Close' column even after renaming.")

    data = add_features(raw_data.copy())
    data = clean_data(data)
except Exception as e:
    st.error(f"❌ Error loading or processing data: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "📈 EDA", "🤖 Predictions", "🏆 Top Performers"])

with tab1:
    st.subheader("🔍 Sample of Processed Data")
    st.dataframe(data.tail(10), use_container_width=True)

with tab2:
    st.subheader("📏 Price and Moving Averages")
    st.pyplot(plot_moving_averages(data))
    st.subheader("📌 Correlation Heatmap")
    plot_corr_heatmap(data)

with tab3:
    st.subheader("🤖 Model Training and Evaluation")
    X, y = prepare_data(data)
    model, (X_train, X_test, y_train, y_test), y_pred, metrics = train_and_evaluate(X, y)

    st.metric("🎯 Accuracy", f"{metrics['accuracy']:.2%}")
    with st.expander("📋 Classification Report"):
        st.json(metrics['report'])
    st.text("🧮 Confusion Matrix:")
    st.write(metrics['confusion'])

    st.subheader("📈 Close Price with Predicted 'Up' Days")
    st.pyplot(plot_predictions(data, y_test, y_pred))

    st.download_button(
        label="⬇️ Download Processed Data as CSV",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name=f"{ticker}_processed.csv",
        mime='text/csv'
    )

    # NLP-based layman-friendly summary
    st.subheader("🗣️ Summary in Simple Terms")
total = len(y_test)
correct = sum(y_test == y_pred)
acc_percent = metrics['accuracy'] * 100
up_days = sum((y_pred == 1))
summary = f"""
- ✅ The model correctly predicted the stock's next-day movement **{correct} out of {total}** times.
- 📊 That's an accuracy of **{acc_percent:.2f}%**.
- 📈 It predicted the stock would go *up* on **{up_days}** of those days.
- 📉 It predicted the stock would go *down* on **{total - up_days}** of those days.
"""

if acc_percent >= 80:
    summary += """
- 💬 **Interpretation:** The model shows strong predictive power.
- 🧠 **Confidence Tip:** You can use this prediction confidently, but still combine it with other indicators.
"""
elif acc_percent >= 60:
    summary += """
- 💬 **Interpretation:** The model performs reasonably well.
- 🧠 **Confidence Tip:** Treat predictions as helpful but not foolproof.
"""
else:
    summary += """
- ⚠️ **Interpretation:** The model’s predictions are not highly reliable.
- 🧠 **Confidence Tip:** Consider retraining with more data or tuning the model.
"""

# ➕ Add recent trend analysis
recent_data = data['Close'].tail(10)
trend_pct = ((recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]) * 100
volatility = recent_data.pct_change().std() * 100

if trend_pct > 2:
    trend_desc = "an upward trend"
elif trend_pct < -2:
    trend_desc = "a downward trend"
else:
    trend_desc = "a mostly sideways trend"

summary += f"""
- 📉 **Recent Trend Insight:** In the past 10 days, the stock has shown {trend_desc} with a net change of {trend_pct:.2f}%. Volatility has been {'high' if volatility > 2 else 'moderate' if volatility > 1 else 'low'}.
"""

st.markdown(summary)

# 🔹 Show sparkline of recent close trend with trendline and predicted up-day markers
fig, ax = plt.subplots(figsize=(6, 1.5))
ax.plot(recent_data.index, recent_data.values, color='white', marker='o', linewidth=1.5, label='Close')
ax.plot(recent_data.index, recent_data.rolling(window=3).mean(), color='cyan', linestyle='--', linewidth=1, label='Trend')

# Highlight predicted up days if in recent data
recent_dates = recent_data.index
pred_up_dates = data.iloc[-len(y_test):].index[y_pred == 1]
highlight_dates = [d for d in recent_dates if d in pred_up_dates]
highlight_prices = recent_data.loc[highlight_dates]
ax.scatter(highlight_dates, highlight_prices, color='lime', marker='^', s=50, label='Predicted Up')

ax.set_facecolor('#222222')
fig.patch.set_facecolor('#222222')
ax.tick_params(colors='white', labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('white')
ax.set_title("Recent 10-Day Close Price Trend", color='white', fontsize=10)
ax.legend(loc='upper left', fontsize=7, facecolor='#222222', edgecolor='white', labelcolor='white')
st.pyplot(fig)

with tab4:
    st.subheader("🏆 Top 5 Predicted Stocks by Accuracy")
    from tqdm import tqdm
    progress = st.progress(0)
    top_accuracies = []
    for i, symbol in enumerate(symbols[:10]):  # limit for faster UI
        try:
            df = fetch_data(symbol, start=start_date, end=end_date)
            if f"Close_{symbol}" in df.columns:
                df.rename(columns={f"Close_{symbol}": "Close"}, inplace=True)
            df = clean_data(add_features(df))
            X, y = prepare_data(df)
            _, _, _, met = train_and_evaluate(X, y)
            top_accuracies.append((symbol, met['accuracy']))
        except:
            continue
        progress.progress((i+1)/10)

    top_accuracies.sort(key=lambda x: x[1], reverse=True)
    st.write(pd.DataFrame(top_accuracies[:5], columns=["Stock", "Accuracy"]))
