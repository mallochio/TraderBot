import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import numpy as np

class BayesianLinear(nn.Module):
    """Bayesian Linear layer with weight uncertainty."""
    
    def __init__(self, in_features, out_features, prior_sigma=0.1):
        super().__init__()
        
        # Weight parameters - mean and variance for each weight
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Prior distribution
        self.prior_sigma = prior_sigma
        self.weight_prior = Normal(0, prior_sigma)
        self.bias_prior = Normal(0, prior_sigma)
        
        # Initialize parameters
        self.reset_parameters()
        
        # KL divergence
        self.kl_divergence = 0
        
    def reset_parameters(self):
        # Initialize means
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        
        # Initialize rho (log of standard deviation)
        nn.init.constant_(self.weight_rho, -4)  # std = log(1 + exp(rho)) ≈ 0.02
        nn.init.constant_(self.bias_rho, -4)
    
    def forward(self, input, sample=True):
        if sample:
            # Sample weights using reparameterization trick
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
            
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
            
            # Calculate KL divergence for weights and biases
            weight_log_posterior = Normal(self.weight_mu, weight_sigma).log_prob(weight).sum()
            weight_log_prior = self.weight_prior.log_prob(weight).sum()
            bias_log_posterior = Normal(self.bias_mu, bias_sigma).log_prob(bias).sum()
            bias_log_prior = self.bias_prior.log_prob(bias).sum()
            
            # Store KL divergence
            self.kl_divergence = weight_log_posterior - weight_log_prior + bias_log_posterior - bias_log_prior
        else:
            # Use mean during inference without sampling
            weight = self.weight_mu
            bias = self.bias_mu
            self.kl_divergence = 0
        
        return F.linear(input, weight, bias)
        
    def get_kl_divergence(self):
        return self.kl_divergence


class BayesianMLP(nn.Module):
    """Multi-layer perceptron with Bayesian layers."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, prior_sigma=0.1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create layers
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        
        for i in range(len(layer_dims) - 1):
            layers.append(BayesianLinear(layer_dims[i], layer_dims[i+1], prior_sigma))
            if i < len(layer_dims) - 2:  # No activation/dropout on last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, sample=True):
        # Keep track of KL divergence
        kl_div = 0
        
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                x = layer(x, sample)
                kl_div += layer.get_kl_divergence()
            else:
                x = layer(x)
        
        return x, kl_div
    
    def predict_with_uncertainty(self, x, n_samples=20):
        """Make predictions with uncertainty estimates."""
        self.eval()
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                output, _ = self.forward(x, sample=True)
                predictions.append(output)
                
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_prediction = torch.mean(predictions, dim=0)
        
        # Uncertainty (standard deviation of predictions)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_prediction, uncertainty


class BayesianDropout(nn.Module):
    """Bayesian Dropout layer."""
    
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        
    def forward(self, x, sample=True):
        if self.training or sample:
            # Apply same dropout mask to all elements in the sequence
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.dropout_prob)) / (1 - self.dropout_prob)
            return x * mask
        else:
            return x


class UncertaintyEvaluator:
    """Utility class to evaluate and interpret uncertainty estimates."""
    
    @staticmethod
    def calculate_confidence_interval(mean, uncertainty, confidence=0.95):
        """Calculate confidence interval based on uncertainty."""
        # For normal distribution, 95% CI is approximately mean ± 1.96 * std
        z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        lower_bound = mean - z_score * uncertainty
        upper_bound = mean + z_score * uncertainty
        return lower_bound, upper_bound
    
    @staticmethod
    def uncertainty_to_risk_weight(uncertainty, sensitivity=1.0, max_weight=1.0):
        """Convert uncertainty to position sizing weight."""
        # Higher uncertainty leads to smaller position size
        weight = max_weight * torch.exp(-sensitivity * uncertainty)
        return torch.clamp(weight, min=0.1, max=max_weight)  # Always maintain at least 10% weight
