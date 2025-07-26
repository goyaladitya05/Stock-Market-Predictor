from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def prepare_data(df):
    feature_cols = ['MA5', 'MA10', 'MA20', 'Return', 'RSI14', 'MACD', 'MACD_signal',
                    'Bollinger_Mavg', 'Bollinger_High', 'Bollinger_Low']
    target_col = 'Target'
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def train_and_evaluate(X, y, model_name='RandomForest'):
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=500),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    model = models.get(model_name, RandomForestClassifier(random_state=42))
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred).tolist()
    }

    import pandas as pd

    # Add accuracy for visualization
    metrics['model_name'] = model_name
    metrics['accuracy_percent'] = metrics['accuracy'] * 100

    # Optional: store results in CSV or aggregate accuracy for multiple models
    if 'model_logs' not in globals():
        global model_logs
        model_logs = []
    model_logs.append({"Model": model_name, "Accuracy": metrics['accuracy_percent']})

    return model, (X_train, X_test, y_train, y_test), y_pred, metrics

def get_model_options():
    return ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'XGBoost']


# --- Optional frontend section for model comparison ---
def render_comparison_tab():
    import streamlit as st
    st.subheader("📊 Model Comparison")
    st.caption("Compare the accuracy of different models you have run so far.")
    st.divider()
    plot_model_comparison()


def plot_model_comparison():
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st

    if 'model_logs' not in globals() or len(model_logs) < 2:
        st.info("Run predictions with different models to see a comparison chart.")
        return

    df_logs = pd.DataFrame(model_logs).drop_duplicates(subset='Model', keep='last')
    df_logs = df_logs.sort_values(by='Accuracy', ascending=False)

    selected_models = st.multiselect("Select models to compare:", options=df_logs['Model'].tolist(), default=df_logs['Model'].tolist())
    if not selected_models:
        st.warning("Please select at least one model.")
        return

    df_logs = df_logs[df_logs['Model'].isin(selected_models)]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(df_logs['Model'], df_logs['Accuracy'], color='skyblue')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('📊 Model Accuracy Comparison')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)

    # Show comparison table below chart
    st.markdown("### 📋 Model Accuracy Table")
    st.dataframe(df_logs.reset_index(drop=True), use_container_width=True)
