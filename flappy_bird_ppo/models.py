"""
Neural network models for PPO agent
Contains Actor (policy) and Critic (value) networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class Actor(nn.Module):
    """
    Actor network (policy) for PPO algorithm
    Outputs action probabilities given state observations
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[512, 512, 256]):
        super(Actor, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, obs):
        """
        Forward pass through the network
        
        Args:
            obs: State observations [batch_size, obs_dim]
            
        Returns:
            action_logits: Raw logits for each action [batch_size, action_dim]
        """
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        return self.network(obs)
    
    def get_action_and_log_prob(self, obs):
        """
        Get action and its log probability for given observation
        
        Args:
            obs: State observation
            
        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            entropy: Entropy of the action distribution
        """
        action_logits = self.forward(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Create categorical distribution
        dist = Categorical(action_probs)
        
        # Sample action
        action = dist.sample()
        
        # Get log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def get_action(self, obs):
        """
        Get action for given observation (for inference)
        
        Args:
            obs: State observation
            
        Returns:
            action: Selected action
        """
        with torch.no_grad():
            action_logits = self.forward(obs)
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Create categorical distribution
            dist = Categorical(action_probs)
            
            # Sample action
            action = dist.sample()
            
        return action
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for given observations (used in PPO updates)
        
        Args:
            obs: State observations [batch_size, obs_dim]
            actions: Actions taken [batch_size]
            
        Returns:
            log_probs: Log probabilities of the actions
            entropy: Entropy of the action distributions
        """
        action_logits = self.forward(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Create categorical distribution
        dist = Categorical(action_probs)
        
        # Get log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class Critic(nn.Module):
    """
    Critic network (value function) for PPO algorithm
    Outputs state value estimates
    """
    
    def __init__(self, obs_dim, hidden_sizes=[512, 512, 256]):
        super(Critic, self).__init__()
        
        self.obs_dim = obs_dim
        
        # Build network layers
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, obs):
        """
        Forward pass through the network
        
        Args:
            obs: State observations [batch_size, obs_dim]
            
        Returns:
            values: State value estimates [batch_size, 1]
        """
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        return self.network(obs).squeeze(-1)  # Remove last dimension for scalar values


class PPONetworks:
    """
    Container class for Actor and Critic networks with shared utilities
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[512, 512, 256], device='cpu'):
        self.device = device
        
        # Initialize networks
        self.actor = Actor(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic = Critic(obs_dim, hidden_sizes).to(device)
        
        # Store dimensions
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    
    def get_action_and_value(self, obs):
        """
        Get action and value for given observation
        
        Args:
            obs: State observation
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
            entropy: Entropy of action distribution
            value: State value estimate
        """
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        # Get action from actor
        action, log_prob, entropy = self.actor.get_action_and_log_prob(obs)
        
        # Get value from critic
        value = self.critic(obs)
        
        return action, log_prob, entropy, value
    
    def save_models(self, filepath):
        """Save both actor and critic models"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim
        }, filepath)
    
    def load_models(self, filepath):
        """Load both actor and critic models"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    def get_model_parameters(self):
        """Get combined parameters from both networks for optimizer"""
        return list(self.actor.parameters()) + list(self.critic.parameters())
    
    def set_train_mode(self):
        """Set both networks to training mode"""
        self.actor.train()
        self.critic.train()
    
    def set_eval_mode(self):
        """Set both networks to evaluation mode"""
        self.actor.eval()
        self.critic.eval()