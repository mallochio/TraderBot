import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .bayesian import BayesianLinear, BayesianDropout

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BayesianMultiHeadAttention(nn.Module):
    """Multi-head attention with Bayesian dropout."""
    
    def __init__(self, d_model, num_heads, dropout=0.1, prior_sigma=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Use Bayesian linear layers for query, key, value projections
        self.q_proj = BayesianLinear(d_model, d_model, prior_sigma)
        self.k_proj = BayesianLinear(d_model, d_model, prior_sigma)
        self.v_proj = BayesianLinear(d_model, d_model, prior_sigma)
        
        self.out_proj = BayesianLinear(d_model, d_model, prior_sigma)
        self.dropout = BayesianDropout(dropout)
        
    def forward(self, query, key, value, mask=None, sample=True):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self.q_proj(query, sample).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key, sample).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value, sample).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights, sample)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output, sample)
        
        # Calculate KL divergence
        kl_div = self.q_proj.get_kl_divergence() + self.k_proj.get_kl_divergence() + \
                 self.v_proj.get_kl_divergence() + self.out_proj.get_kl_divergence()
                 
        return output, kl_div


class BayesianTransformerEncoderLayer(nn.Module):
    """Bayesian Transformer encoder layer."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, prior_sigma=0.1):
        super().__init__()
        
        self.self_attn = BayesianMultiHeadAttention(d_model, nhead, dropout, prior_sigma)
        
        # Feed forward network
        self.ff1 = BayesianLinear(d_model, dim_feedforward, prior_sigma)
        self.ff2 = BayesianLinear(dim_feedforward, d_model, prior_sigma)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = BayesianDropout(dropout)
        self.dropout2 = BayesianDropout(dropout)
        
    def forward(self, src, src_mask=None, sample=True):
        # Self-attention block
        attn_output, kl_attn = self.self_attn(src, src, src, src_mask, sample)
        src = src + self.dropout1(attn_output, sample)
        src = self.norm1(src)
        
        # Feed forward block
        ff_output = self.ff1(src, sample)
        ff_output = F.relu(ff_output)
        ff_output = self.dropout2(ff_output, sample)
        ff_output = self.ff2(ff_output, sample)
        
        src = src + ff_output
        src = self.norm2(src)
        
        # Calculate KL divergence
        kl_div = kl_attn + self.ff1.get_kl_divergence() + self.ff2.get_kl_divergence()
        
        return src, kl_div


class BayesianTransformerEncoder(nn.Module):
    """Stack of transformer encoder layers with Bayesian components."""
    
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, prior_sigma=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BayesianTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, prior_sigma)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, mask=None, sample=True):
        output = src
        kl_div_total = 0
        
        for layer in self.layers:
            output, kl_div = layer(output, mask, sample)
            kl_div_total += kl_div
            
        return output, kl_div_total


class FinancialTransformer(nn.Module):
    """Transformer model for financial time series with uncertainty quantification."""
    
    def __init__(self, n_features, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, 
                 max_seq_len=100, dropout=0.1, prior_sigma=0.1, n_actions=3):
        super().__init__()
        
        # Feature embedding layer (convert raw features to model dimension)
        self.feature_embedding = BayesianLinear(n_features, d_model, prior_sigma)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        self.transformer_encoder = BayesianTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            prior_sigma=prior_sigma
        )
        
        # Output heads for RL (actor and critic)
        self.action_head = BayesianLinear(d_model, n_actions, prior_sigma)
        self.value_head = BayesianLinear(d_model, 1, prior_sigma)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize the parameters of the model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, mask=None, sample=True):
        """
        Forward pass through the model.
        
        Args:
            src: Tensor of shape [batch_size, seq_len, n_features]
            mask: Optional mask tensor
            sample: Whether to sample weights (True for training, False for deterministic inference)
            
        Returns:
            action_logits: Action logits tensor
            state_values: State value tensor
            kl_div: KL divergence for Bayesian layers
        """
        # Keep track of total KL divergence
        kl_div_total = 0
        
        # Embed input features
        x = self.feature_embedding(src, sample)
        kl_div_total += self.feature_embedding.get_kl_divergence()
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x, kl_div_encoder = self.transformer_encoder(x, mask, sample)
        kl_div_total += kl_div_encoder
        
        # Extract features from the last time step
        x = x[:, -1, :]
        
        # Actor head (policy)
        action_logits = self.action_head(x, sample)
        kl_div_total += self.action_head.get_kl_divergence()
        
        # Critic head (value function)
        state_values = self.value_head(x, sample)
        kl_div_total += self.value_head.get_kl_divergence()
        
        return action_logits, state_values, kl_div_total
    
    def predict_with_uncertainty(self, src, mask=None, n_samples=20):
        """
        Make predictions with uncertainty estimates.
        
        Args:
            src: Input tensor
            mask: Optional attention mask
            n_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            mean_action_logits: Mean action logits across samples
            action_uncertainty: Uncertainty in action logits (std deviation)
            mean_state_values: Mean state values across samples
            value_uncertainty: Uncertainty in state values (std deviation)
        """
        self.eval()
        
        action_samples = []
        value_samples = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                action_logits, state_values, _ = self.forward(src, mask, sample=True)
                action_samples.append(action_logits)
                value_samples.append(state_values)
                
        # Stack samples
        action_samples = torch.stack(action_samples)
        value_samples = torch.stack(value_samples)
        
        # Mean predictions
        mean_action_logits = torch.mean(action_samples, dim=0)
        mean_state_values = torch.mean(value_samples, dim=0)
        
        # Uncertainties (standard deviations)
        action_uncertainty = torch.std(action_samples, dim=0)
        value_uncertainty = torch.std(value_samples, dim=0)
        
        return mean_action_logits, action_uncertainty, mean_state_values, value_uncertainty
    
    def get_action_with_uncertainty(self, state, mask=None, n_samples=20):
        """
        Get action probabilities with uncertainty for trading.
        
        Args:
            state: Current state observation
            mask: Optional attention mask
            n_samples: Number of Monte Carlo samples
            
        Returns:
            action_probs: Action probabilities
            action_uncertainty: Uncertainty in action probabilities
            value: Expected state value
            value_uncertainty: Uncertainty in state value
        """
        mean_logits, logit_uncertainty, mean_value, value_uncertainty = self.predict_with_uncertainty(
            state, mask, n_samples
        )
        
        # Convert logits to probabilities
        action_probs = F.softmax(mean_logits, dim=-1)
        
        # Propagate uncertainty through softmax using first-order approximation
        # This is a simplified approach to uncertainty propagation
        action_uncertainty = logit_uncertainty * action_probs * (1 - action_probs)
        
        return action_probs, action_uncertainty, mean_value, value_uncertainty


class UncertaintyAwareTrader:
    """Trading policy that incorporates uncertainty estimates into decisions."""
    
    def __init__(self, model, uncertainty_threshold=0.2, confidence=0.95):
        """
        Initialize the trader.
        
        Args:
            model: FinancialTransformer model
            uncertainty_threshold: Threshold for max acceptable uncertainty
            confidence: Confidence level for decision making
        """
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence = confidence
    
    def select_action(self, state, mask=None, n_samples=20):
        """
        Select action based on model predictions and uncertainty.
        
        Returns:
            action: Selected action index
            action_probs: Action probabilities
            value: Expected value
            info: Dictionary with additional information
        """
        # Get predictions with uncertainty
        action_probs, action_uncertainty, value, value_uncertainty = self.model.get_action_with_uncertainty(
            state, mask, n_samples
        )
        
        # Risk-adjusted action selection
        if torch.max(action_uncertainty) > self.uncertainty_threshold:
            # High uncertainty case: default to "hold" action (typically index 2)
            action = torch.tensor(2)  # Hold action
            confidence_level = 0.0
        else:
            # Apply uncertainty-adjusted selection
            # Scale probabilities inversely to uncertainty
            adjusted_probs = action_probs / (1 + action_uncertainty)
            action = torch.argmax(adjusted_probs)
            confidence_level = 1.0 - action_uncertainty[action].item()
        
        info = {
            "action_uncertainty": action_uncertainty,
            "value_uncertainty": value_uncertainty,
            "confidence": confidence_level
        }
        
        return action, action_probs, value, info


class BayesianTemporalCNN(nn.Module):
    """Optional temporal CNN encoder for feature extraction before transformer."""
    
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], prior_sigma=0.1):
        super().__init__()
        
        # Multiple parallel convolutional branches with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                BayesianDropout(0.1)
            )
            for kernel_size in kernel_sizes
        ])
        
        # Output layer combining all branches
        self.output_layer = BayesianLinear(out_channels * len(kernel_sizes), out_channels, prior_sigma)
        
    def forward(self, x, sample=True):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, in_channels]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, out_channels]
        """
        # Transpose for Conv1d: [batch_size, in_channels, seq_len]
        x = x.transpose(1, 2)
        
        # Apply parallel convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
            
        # Concatenate along feature dimension
        combined = torch.cat(conv_outputs, dim=1)
        
        # Transpose back: [batch_size, seq_len, out_channels * len(kernel_sizes)]
        combined = combined.transpose(1, 2)
        
        # Apply output layer
        output = self.output_layer(combined, sample)
        
        return output, self.output_layer.get_kl_divergence()
