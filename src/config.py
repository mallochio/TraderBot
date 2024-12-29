import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'saved_models'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    # Transformer architecture
    'n_encoder_layers': 4,
    'n_heads': 8,
    'd_model': 256,
    'dropout': 0.1,
    'd_ff': 1024,
    
    # Bayesian settings
    'prior_sigma': 0.1,
    'mc_samples_train': 5,
    'mc_samples_eval': 20,
    
    # Input/output dimensions
    'max_seq_len': 100,
    'n_features': 20,  # OHLCV, technical indicators, etc.
    'n_actions': 3,    # Buy, sell, hold
}

# Training configuration
TRAIN_CONFIG = {
    'lr': 3e-4,
    'batch_size': 64,
    'epochs': 100,
    'patience': 10,    # Early stopping
    'clip_grad_norm': 0.5,
    'weight_decay': 1e-5,
    'beta': 0.01,      # KL divergence weight for Bayesian layers
}

# RL configuration
RL_CONFIG = {
    'gamma': 0.99,
    'lambda_': 0.95,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.5,
    'ppo_epoch': 4,
    'num_mini_batch': 4,
    'clip_param': 0.2,
}

# Ray configuration for distributed training
RAY_CONFIG = {
    'num_workers': 4,
    'num_cpus_per_worker': 1,
    'num_gpus': 0,
    'num_gpus_per_worker': 0,
    'rollout_fragment_length': 200,
    'train_batch_size': 4000,
}

# Trading configuration
TRADING_CONFIG = {
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'symbols': ['SPY', 'AAPL', 'GOOGL', 'MSFT'],
    'timeframe': '1h',
    'initial_capital': 10000,
    'max_position_size': 0.2,  # Max 20% of capital in a single position
    'transaction_fee': 0.001,  # 0.1% per trade
    'risk_free_rate': 0.02,    # For Sharpe ratio calculation
    'stop_loss': 0.05,         # 5% stop loss
    'take_profit': 0.10,       # 10% take profit
}

# Backtesting configuration
BACKTEST_CONFIG = {
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'cash': 10000,
    'commission': 0.001,
}
