from modules.data_download import fetch_data
from modules.feature_engineer import add_features, clean_data
from modules.visualization import plot_eda, plot_corr_heatmap
from modules.model import prepare_data, train_and_evaluate

def main():
    ticker = 'RELIANCE.NS'
    df = fetch_data(ticker, start='2023-01-01', end='2023-07-01')
    # Use the correct col name as identified after flattening
    close_col = [col for col in df.columns if 'Close' in col][0]
    df = add_features(df, close_col)
    df = clean_data(df)
    print(df.head())
    plot_eda(df, close_col)
    plot_corr_heatmap(df)
    X, y = prepare_data(df)
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
