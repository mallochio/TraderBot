import pandas as pd
import numpy as np
import yfinance as yf
from config import TRADING_CONFIG, DATA_DIR

def download_data():
    """Download historical data for symbols in config."""
    data = {}
    for symbol in TRADING_CONFIG['symbols']:
        # Convert ccxt style symbols to yfinance format
        ticker = symbol.replace('/', '-')
        df = yf.download(
            ticker, 
            start=TRADING_CONFIG.get('start_date', '2020-01-01'),
            end=TRADING_CONFIG.get('end_date', '2022-12-31'),
            interval=TRADING_CONFIG['timeframe']
        )
        df.to_csv(DATA_DIR / f"{ticker}.csv")
        data[symbol] = df
    return data

def add_technical_indicators(df):
    """Add technical indicators to dataframe."""
    # Example: Simple Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # Example: RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Add more indicators as needed
    return df

def prepare_features(df, seq_length=100):
    """Prepare feature sequences for the model."""
    # Normalize features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'SMA50', 'RSI']
    df_features = df[feature_cols].copy()
    
    # Simple normalization (can be improved)
    for col in feature_cols:
        if col in df_features.columns:
            df_features[col] = (df_features[col] - df_features[col].mean()) / df_features[col].std()
    
    # Create sequences
    sequences = []
    for i in range(len(df_features) - seq_length):
        seq = df_features.iloc[i:i+seq_length].values
        sequences.append(seq)
    
    return np.array(sequences)

if __name__ == "__main__":
    # Download and process data
    data = download_data()
    
    # Process each symbol's data
    for symbol, df in data.items():
        df = add_technical_indicators(df)
        features = prepare_features(df)
        np.save(DATA_DIR / f"{symbol.replace('/', '-')}_features.npy", features)
