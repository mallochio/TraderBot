import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.tune.registry import register_env
from models.transformer import FinancialTransformer, UncertaintyAwareTrader
from models.bayesian import UncertaintyEvaluator
import sys
import os
import pandas as pd
from tqdm import tqdm

# Add the project root directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import TradingEnvironment
from config import MODEL_CONFIG, TRAIN_CONFIG, RL_CONFIG, RAY_CONFIG, DATA_DIR, MODEL_DIR

def create_model():
    """Create and initialize the financial transformer model."""
    model = FinancialTransformer(
        n_features=MODEL_CONFIG['n_features'],
        d_model=MODEL_CONFIG['d_model'],
        nhead=MODEL_CONFIG['n_heads'],
        num_layers=MODEL_CONFIG['n_encoder_layers'],
        dim_feedforward=MODEL_CONFIG['d_ff'],
        max_seq_len=MODEL_CONFIG['max_seq_len'],
        dropout=MODEL_CONFIG['dropout'],
        prior_sigma=MODEL_CONFIG['prior_sigma'],
        n_actions=MODEL_CONFIG['n_actions']
    )
    return model

def env_creator(env_config):
    """Environment creator function for registration with Ray."""
    # Pre-process the data if needed
    symbol = env_config.get("symbol", "AAPL")
    data = None
    
    # If features are pre-computed, load them
    if "data" in env_config:
        data = env_config["data"]
    else:
        import os
        from config import DATA_DIR
        
        # Try to load feature data
        feature_path = os.path.join(DATA_DIR, f"{symbol}_features.npy")
        csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        
        if os.path.exists(feature_path):
            data = np.load(feature_path)
        elif os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Could not find data for {symbol}")
    
    # Include data in the config
    env_config["data"] = data
    
    return TradingEnvironment(config=env_config)

def train_model():
    """Train the model using Ray RLlib."""
    # Initialize Ray
    ray.init(num_cpus=RAY_CONFIG['num_cpus_per_worker'] * RAY_CONFIG['num_workers'],
             num_gpus=RAY_CONFIG['num_gpus'])
    
    # Register the custom environment
    register_env("trading_env", env_creator)
    
    # Try to load data for environment initialization
    env_config = {
        "initial_capital": 10000,
        "transaction_fee": 0.001
    }
    
    # Try to load data if available
    try:
        data_path = os.path.join(DATA_DIR, "SPY_features.npy")
        if os.path.exists(data_path):
            # Load and verify the data
            data = np.load(data_path)
            print(f"Loaded data from {data_path}, shape: {data.shape}")
            
            # If data is 1D, create a synthetic OHLCV DataFrame
            if len(data.shape) == 1:
                print("Converting 1D data to OHLCV format")
                prices = data
                dates = pd.date_range("2020-01-01", periods=len(prices))
                data = pd.DataFrame({
                    'Date': dates,
                    'Open': prices,
                    'High': prices * 1.01,
                    'Low': prices * 0.99,
                    'Close': prices,
                    'Volume': np.ones_like(prices) * 1000
                })
                print(f"Created DataFrame with shape {data.shape}")
            
            env_config["data"] = data
            env_config["symbol"] = "SPY"
    except Exception as e:
        print(f"Warning: Could not load data: {e}")
        print("Will try to use default data from gym-anytrading")
    
    # Rest of the function remains the same...
    
    # In your train.py configuration builder, update the API stack call:
    config = (
        PPOConfig()
        .environment(
            env="trading_env",
            env_config=env_config
        )
        .framework("torch")
        .training(
            lr=TRAIN_CONFIG['lr'],
            gamma=RL_CONFIG['gamma'],
            lambda_=RL_CONFIG['lambda_'],
            entropy_coeff=RL_CONFIG['entropy_coef'],
            vf_loss_coeff=RL_CONFIG['value_loss_coef'],
            clip_param=RL_CONFIG['clip_param']
        )
        .resources(
            num_gpus=RAY_CONFIG['num_gpus']
        )
        .env_runners(
            num_env_runners=RAY_CONFIG['num_workers'],
            num_cpus_per_env_runner=RAY_CONFIG['num_cpus_per_worker'],
            num_gpus_per_env_runner=RAY_CONFIG['num_gpus_per_worker'],
            rollout_fragment_length=RAY_CONFIG['rollout_fragment_length']
        )
        # Disable the new API stack
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
    )

    # Build the algorithm - using build_algo() instead of deprecated build()
    algo = config.build_algo()

    # Train the algorithm
    for i in tqdm(range(TRAIN_CONFIG['epochs'])):
        try:
            result = algo.train()
            # Print the average reward for this iteration
            reward_mean = result.get("episode_reward_mean")
            if reward_mean is None:
                reward_mean = result.get("env_runners", {}).get("episode_reward_mean", "N/A")
            print(f"Iteration {i}: reward = {reward_mean}")
            # Save checkpoint periodically
            if i % 10 == 0:
                checkpoint_path = algo.save(MODEL_DIR)
                # print(f"Checkpoint saved at {checkpoint_path}")
                print(f"Checkpoint saved")
        except Exception as e:
            print(f"Error during training iteration {i}: {e}")
            break

    # Final save
    try:
        algo.save(MODEL_DIR)
    except Exception as e:
        print(f"Error saving final model: {e}")
        
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    train_model()