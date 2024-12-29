import numpy as np
import gymnasium as gym  # Use gymnasium
import pandas as pd
from config import TRADING_CONFIG, BACKTEST_CONFIG

class TradingEnvironment(gym.Env):
    """Trading environment wrapper for gym-anytrading."""
    
    def __init__(self, config=None):
        """Initialize the environment with config parameters."""
        super().__init__()
        
        # Extract config parameters or use defaults
        self.window_size = config.get('window_size', 100) if config else 100
        self.frame_bound = config.get('frame_bound', None)
        self.symbol = config.get('symbol', 'AAPL')
        self.initial_capital = config.get('initial_capital', 10000)
        
        # Load data
        self.data = self._load_data(config.get('data', None))
        
        # Create the gym-anytrading environment
        if self.frame_bound is None:
            self.frame_bound = (self.window_size, len(self.data))
        
        # Correct import path for StocksEnv
        from gym_anytrading.envs.stocks_env import StocksEnv
        
        # Create the environment directly instead of using gym.make
        self.env = StocksEnv(
            df=self.data,
            window_size=self.window_size,
            frame_bound=self.frame_bound
        )
        
        # Set spaces to match the gym-anytrading environment
        self.action_space = self.env.action_space  # 0: sell, 1: buy
        self.observation_space = self.env.observation_space
        
    def _load_data(self, data):
        """Load data from config or format the provided data."""
        if data is not None:
            # If data is already provided, check if format is compatible
            if isinstance(data, pd.DataFrame):
                # Make sure it has the required columns
                required_columns = ['Open', 'High', 'Low', 'Close']
                missing = [col for col in required_columns if col not in data.columns]
                if missing:
                    raise ValueError(f"Data is missing required columns: {missing}")
                return data
            elif isinstance(data, np.ndarray):
                # Handle 3D arrays (e.g. sequences of features)
                if data.ndim == 3:
                    print(f"Data is 3D with shape {data.shape}; taking the last time step from each sequence.")
                    # Take last time step along axis=1 so that new shape becomes (num_sequences, num_features)
                    data = data[:, -1, :]
                    
                # Handle 1D arrays
                if len(data.shape) == 1:
                    print(f"Converting 1D array of shape {data.shape} to 2D")
                    data = data.reshape(-1, 1)
                    
                # Now data should be at least 2D
                if len(data.shape) < 2:
                    raise ValueError(f"Data array must be at least 2D, got shape {data.shape}")
                    
                # Choose appropriate columns based on number of features
                if data.shape[1] >= 8:
                    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'SMA50', 'RSI']
                    df = pd.DataFrame(data[:, :8], columns=columns)
                    return df
                elif data.shape[1] >= 5:
                    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df = pd.DataFrame(data[:, :5], columns=columns)
                    return df
                else:
                    print(f"Warning: Not enough columns in data, shape={data.shape}. Creating synthetic OHLCV data.")
                    close_values = data[:, 0] if data.shape[1] > 0 else np.arange(len(data))
                    df = pd.DataFrame({
                        'Open': close_values,
                        'High': close_values * 1.01,
                        'Low': close_values * 0.99,
                        'Close': close_values,
                        'Volume': np.ones_like(close_values) * 1000
                    })
                    return df
        
        # If no data provided, try to load from data directory based on symbol
        from config import DATA_DIR
        import os
        
        csv_path = os.path.join(DATA_DIR, f"{self.symbol}.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Could not find data file for {self.symbol}")
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        # Check if the environment uses the new API
        if hasattr(self.env, 'reset') and callable(self.env.reset):
            try:
                # Try the new API first
                observation, info = self.env.reset(seed=seed)
                return observation, info
            except (TypeError, ValueError):
                # Fall back to old API if new one fails
                observation = self.env.reset()
                return observation, {}
        return self.env.reset(), {}
    
    def step(self, action):
        """Take a step in the environment."""
        # Map actions if necessary (if your system uses different action encoding)
        # gym-anytrading uses: 0=sell, 1=buy
        # Our system uses: 0=buy, 1=sell, 2=hold
        if action == 2:  # Hold
            # For hold, repeat the last action
            action = self._get_last_action()
        elif action == 0:  # Buy
            action = 1
        elif action == 1:  # Sell
            action = 0
        
        try:
            # Try the new Gymnasium API first (5 return values)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            return obs, reward, terminated, truncated, info
        except ValueError:
            # Fall back to old Gym API (4 return values)
            obs, reward, done, info = self.env.step(action)
            
            # Enhance info with custom metrics
            info.update({
                "portfolio_value": self.env.account_history[-1] if len(self.env.account_history) > 0 else self.initial_capital,
                "return": reward
            })
            
            return obs, reward, done, info
    
    def _get_last_action(self):
        """Get the last action to maintain position for HOLD."""
        if hasattr(self.env, 'history') and len(self.env.history) > 0:
            return self.env.history[-1][2]  # Last action in history
        return 1  # Default to buy if no history
    
    def render(self, mode='human'):
        """Render the environment."""
        try:
            return self.env.render(mode=mode)
        except TypeError:
            return self.env.render()  # For newer versions without mode parameter