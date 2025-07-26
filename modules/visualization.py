import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_corr_heatmap(data):
    """Displays a correlation heatmap of the numerical features."""
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)


def plot_moving_averages(data):
    """Plots closing price along with MA5, MA10, MA20."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data['Close'], label='Close', color='blue')
    ax.plot(data['MA5'], label='MA5', linestyle='--')
    ax.plot(data['MA10'], label='MA10', linestyle='--')
    ax.plot(data['MA20'], label='MA20', linestyle='--')
    ax.set_title('Price with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    return fig


def plot_predictions(data, y_test, y_pred):
    """Plots actual Close prices with predicted Up days."""
    fig, ax = plt.subplots(figsize=(12, 5))
    index = data.index[-len(y_test):]
    close_prices = data['Close'].iloc[-len(y_test):]
    ax.plot(index, close_prices, label='Close', color='blue')

    up_predicted = (y_pred == 1)
    ax.scatter(index[up_predicted], close_prices[up_predicted], color='green', marker='^', label='Predicted Up')

    ax.set_title("Predicted 'Up' Days on Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    return fig


def plot_eda(data):
    """Optional: placeholder for additional exploratory data analysis."""
    st.write("📊 EDA Placeholder: You can implement more visualizations here.")
