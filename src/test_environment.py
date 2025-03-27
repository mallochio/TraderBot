import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from environment import TradingEnvironment

def main():
    """Test the trading environment."""
    print("Testing gym-anytrading integration")
    
    # Import sample data
    try:
        from gym_anytrading.datasets import STOCKS_GOOGL
        sample_data = STOCKS_GOOGL.copy()
        print("Successfully loaded STOCKS_GOOGL sample data")
    except ImportError as e:
        print(f"Error importing sample data: {e}")
        # Create fallback sample data
        print("Creating fallback sample data")
        # Generate random OHLCV data
        n_samples = 500
        dates = pd.date_range("2020-01-01", periods=n_samples)
        close = np.random.random(n_samples) * 100 + 100  # Random prices around 100-200
        
        # Calculate other values based on close
        high = close * (1 + np.random.random(n_samples) * 0.05)
        low = close * (1 - np.random.random(n_samples) * 0.05)
        open_price = low + np.random.random(n_samples) * (high - low)
        volume = np.random.random(n_samples) * 10000000
        
        # Create DataFrame
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    config = {
        'symbol': 'GOOGL',
        'window_size': 10,
        'frame_bound': (10, 100),  # Use a small subset for testing
        'data': sample_data,
    }
    
    print(f"Testing environment with sample data")
    print(f"Data shape: {sample_data.shape}")
    print(f"First few rows:\n{sample_data.head(3)}")
    
    # Initialize environment (with error handling)
    try:
        env = TradingEnvironment(config=config)
        print("Environment created successfully")
    except Exception as e:
        print(f"Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Reset environment
    try:
        reset_result = env.reset(seed=42)
        print("Reset successful")
        if isinstance(reset_result, tuple):
            observation, info = reset_result
            print("Using new Gymnasium API (observation, info)")
        else:
            observation = reset_result
            info = {}
            print("Using old Gym API (observation only)")
        
        print(f"Observation shape: {observation.shape}")
    except Exception as e:
        print(f"Error in reset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run a few steps
    done = False
    total_reward = 0
    step_count = 0
    max_steps = 80
    
    print("\nRunning trading simulation...")
    
    while not done and step_count < max_steps:
        # Sample random action (0=sell, 1=buy)
        action = np.random.choice([0, 1])  # Directly sample action
        
        try:
            # Execute step
            step_result = env.step(action)
            
            # Handle both API versions
            if len(step_result) == 5:
                observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                observation, reward, done, info = step_result
                terminated = truncated = done
                
            total_reward += reward
            step_count += 1
            
            if step_count % 10 == 0:  # Print status every 10 steps
                print(f"Step {step_count}: Action={action}, Reward={reward:.4f}, Portfolio=${info.get('portfolio_value', 'N/A')}")
        
        except Exception as e:
            print(f"Error in step {step_count}: {e}")
            break
    
    print(f"\nTest completed after {step_count} steps")
    print(f"Total Reward: {total_reward:.4f}")
    
    if 'portfolio_value' in info:
        print(f"Final Portfolio Value: ${info['portfolio_value']:.2f}")
    
    # Render final state
    try:
        plt.figure(figsize=(15, 6))
        env.render()
        plt.title(f"Trading Session: GOOGL Sample")
        plt.tight_layout()
        output_file = "test_trading_results.png"
        plt.savefig(output_file)
        print(f"Rendering saved to {output_file}")
        plt.show()
    except Exception as e:
        print(f"Error in rendering: {e}")

    # Output environment details
    print("\nEnvironment Details:")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

if __name__ == "__main__":
    main()