import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.transformer import FinancialTransformer, UncertaintyAwareTrader
from models.bayesian import UncertaintyEvaluator
from config import MODEL_CONFIG, BACKTEST_CONFIG, TRADING_CONFIG, DATA_DIR, MODEL_DIR, RESULTS_DIR
import backtrader as bt
from environment import TradingEnvironment

class BayesianTransformerStrategy(bt.Strategy):
    """Backtrader strategy using the Bayesian Transformer model."""
    
    params = {
        'model_path': None,
        'seq_length': 100,
        'uncertainty_threshold': 0.2,
    }
    
    def __init__(self):
        # Load the model
        self.model = FinancialTransformer(
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
        
        # Load trained weights if provided
        if self.p.model_path:
            self.model.load_state_dict(torch.load(self.p.model_path))
        
        self.model.eval()
        self.trader = UncertaintyAwareTrader(
            model=self.model,
            uncertainty_threshold=self.p.uncertainty_threshold
        )
        
        # Initialize gym-anytrading compatible environment for observation processing
        self.env_config = {
            'symbol': self.data._name,
            'window_size': self.p.seq_length
        }
        
        # Track positions and orders
        self.current_position = 0
        self.orders = {}
        
    def _get_observation_window(self):
        """Get the observation window in a format compatible with the model."""
        # Get current position in the data
        current_idx = len(self.data) - 1
        
        # Create a window of OHLCV data
        window = np.array([
            [self.data.open[i], self.data.high[i], self.data.low[i], 
             self.data.close[i], self.data.volume[i]] 
            for i in range(current_idx - self.p.seq_length + 1, current_idx + 1)
        ])
        
        # Process this window similar to how gym-anytrading would
        # Here we're manually creating a feature representation similar to gym-anytrading
        return window
        
    def next(self):
        # Only proceed if we have enough data for our sequence length
        if len(self.data) <= self.p.seq_length:
            return
            
        # Get observation window
        observation = self._get_observation_window()
        
        # Convert to tensor compatible with the model
        state = torch.FloatTensor(observation).unsqueeze(0)  # Add batch dimension
        
        # Get action from model
        with torch.no_grad():
            action, _, value, info = self.trader.select_action(state, n_samples=20)
            
            # Extract confidence
            confidence = info['confidence']
            
            # Map model actions to gym-anytrading actions:
            # model: 0=buy, 1=sell, 2=hold
            # gym-anytrading: 0=sell, 1=buy
            # Execute action with position sizing based on confidence
            if action == 0:  # Buy
                if not self.position:
                    # Size position based on confidence
                    size = self.broker.get_cash() * TRADING_CONFIG['max_position_size'] * confidence
                    self.orders[self.data._name] = self.buy(size=size / self.data.close[0])
                    
            elif action == 1:  # Sell
                if self.position:
                    self.orders[self.data._name] = self.sell(size=self.position.size)
                    
            # action 2 is hold, do nothing

def run_backtest():
    """Run backtest using the trained model."""
    # Create a cerebro instance
    cerebro = bt.Cerebro()
    
    # Add data
    for symbol in TRADING_CONFIG['symbols']:
        ticker = symbol.replace('/', '-')
        data_path = DATA_DIR / f"{ticker}.csv"
        if data_path.exists():
            data = bt.feeds.YahooFinanceCSVData(
                dataname=data_path,
                fromdate=pd.Timestamp(BACKTEST_CONFIG['start_date']),
                todate=pd.Timestamp(BACKTEST_CONFIG['end_date']),
                reverse=False
            )
            data._name = ticker
            cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(
        BayesianTransformerStrategy,
        model_path=MODEL_DIR / "best_model.pth",
        uncertainty_threshold=0.2
    )
    
    # Set initial cash
    cerebro.broker.setcash(BACKTEST_CONFIG['cash'])
    
    # Set commission
    cerebro.broker.setcommission(commission=BACKTEST_CONFIG['commission'])
    
    # Print starting conditions
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    
    # Run backtest
    results = cerebro.run()
    
    # Print final result
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
    
    # Plot results
    cerebro.plot()

if __name__ == "__main__":
    run_backtest()