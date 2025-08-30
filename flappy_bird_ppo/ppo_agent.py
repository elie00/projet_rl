"""
PPO (Proximal Policy Optimization) Agent Implementation
Contains the core PPO algorithm logic for training the Flappy Bird agent
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict
import gymnasium as gym

from models import PPONetworks
from config import Config


class PPOAgent:
    """
    PPO Agent that handles training and interaction with the environment
    """
    
    def __init__(self, env, config=None):
        self.config = config if config else Config()
        self.env = env
        
        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Initialize networks
        self.networks = PPONetworks(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.config.HIDDEN_SIZES,
            device=self.config.DEVICE
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.networks.get_model_parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        # Storage for rollout data
        self.reset_rollout_buffer()
        
        # Training statistics
        self.episode_rewards = []
        self.training_stats = defaultdict(list)
        
    def reset_rollout_buffer(self):
        """Reset the rollout buffer for new data collection"""
        self.rollout_buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': [],
            'entropies': []
        }
    
    def collect_rollout(self):
        """
        Collect a rollout of experiences from the environment
        
        Returns:
            rollout_data: Dictionary containing the collected experiences
            episode_rewards: List of completed episode rewards during rollout
        """
        self.networks.set_eval_mode()
        episode_rewards = []
        
        obs, _ = self.env.reset()
        episode_reward = 0
        
        for step in range(self.config.MAX_STEPS_PER_ROLLOUT):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.DEVICE)
            
            # Get action and value
            with torch.no_grad():
                action, log_prob, entropy, value = self.networks.get_action_and_value(obs_tensor)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action.cpu().item())
            done = terminated or truncated
            episode_reward += reward
            
            # Store experience
            self.rollout_buffer['observations'].append(obs)
            self.rollout_buffer['actions'].append(action.cpu().item())
            self.rollout_buffer['rewards'].append(reward)
            self.rollout_buffer['log_probs'].append(log_prob.cpu().item())
            self.rollout_buffer['values'].append(value.cpu().item())
            self.rollout_buffer['dones'].append(done)
            self.rollout_buffer['entropies'].append(entropy.cpu().item())
            
            obs = next_obs
            
            # Handle episode end
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                obs, _ = self.env.reset()
        
        # Bootstrap value for last state if episode didn't end
        if not self.rollout_buffer['dones'][-1]:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.DEVICE)
            with torch.no_grad():
                bootstrap_value = self.networks.critic(obs_tensor).cpu().item()
        else:
            bootstrap_value = 0.0
        
        # Convert to numpy arrays
        rollout_data = {
            'observations': np.array(self.rollout_buffer['observations']),
            'actions': np.array(self.rollout_buffer['actions']),
            'rewards': np.array(self.rollout_buffer['rewards']),
            'old_log_probs': np.array(self.rollout_buffer['log_probs']),
            'old_values': np.array(self.rollout_buffer['values']),
            'dones': np.array(self.rollout_buffer['dones']),
            'entropies': np.array(self.rollout_buffer['entropies']),
            'bootstrap_value': bootstrap_value
        }
        
        # Reset buffer
        self.reset_rollout_buffer()
        
        return rollout_data, episode_rewards
    
    def compute_gae_returns(self, rollout_data):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE)
        
        Args:
            rollout_data: Dictionary containing rollout experiences
            
        Returns:
            returns: Computed returns for each step
            advantages: Computed advantages for each step
        """
        rewards = rollout_data['rewards']
        values = rollout_data['old_values']
        dones = rollout_data['dones']
        bootstrap_value = rollout_data['bootstrap_value']
        
        # Compute returns using GAE
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        # Start from the end
        last_gae_lambda = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = bootstrap_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
            
            # Compute TD error
            delta = rewards[step] + self.config.GAMMA * next_value * next_non_terminal - values[step]
            
            # Compute GAE
            advantages[step] = last_gae_lambda = delta + self.config.GAMMA * 0.95 * next_non_terminal * last_gae_lambda
            
            # Compute returns
            returns[step] = advantages[step] + values[step]
        
        return returns, advantages
    
    def ppo_update(self, rollout_data):
        """
        Perform PPO update using collected rollout data
        
        Args:
            rollout_data: Dictionary containing rollout experiences
            
        Returns:
            training_stats: Dictionary containing training statistics
        """
        self.networks.set_train_mode()
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae_returns(rollout_data)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        observations = torch.FloatTensor(rollout_data['observations']).to(self.config.DEVICE)
        actions = torch.LongTensor(rollout_data['actions']).to(self.config.DEVICE)
        old_log_probs = torch.FloatTensor(rollout_data['old_log_probs']).to(self.config.DEVICE)
        returns_tensor = torch.FloatTensor(returns).to(self.config.DEVICE)
        advantages_tensor = torch.FloatTensor(advantages).to(self.config.DEVICE)
        
        # Training statistics
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_kl_div = 0.0
        
        # PPO epochs
        for epoch in range(self.config.PPO_EPOCHS):
            # Forward pass through networks
            new_log_probs, entropy = self.networks.actor.evaluate_actions(observations, actions)
            values = self.networks.critic(observations)
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratios * advantages_tensor
            surr2 = torch.clamp(ratios, 1.0 - self.config.CLIP_EPSILON, 
                              1.0 + self.config.CLIP_EPSILON) * advantages_tensor
            
            # Actor loss (policy loss)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss (value loss)
            critic_loss = F.mse_loss(values, returns_tensor)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = (actor_loss + 
                         self.config.VALUE_COEFF * critic_loss + 
                         self.config.ENTROPY_COEFF * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.networks.get_model_parameters(), 
                self.config.MAX_GRAD_NORM
            )
            
            self.optimizer.step()
            
            # Update statistics
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
            
            # Compute KL divergence for monitoring
            with torch.no_grad():
                kl_div = (old_log_probs - new_log_probs).mean().item()
                total_kl_div += kl_div
        
        # Average statistics over epochs
        training_stats = {
            'actor_loss': total_actor_loss / self.config.PPO_EPOCHS,
            'critic_loss': total_critic_loss / self.config.PPO_EPOCHS,
            'entropy': total_entropy / self.config.PPO_EPOCHS,
            'kl_divergence': total_kl_div / self.config.PPO_EPOCHS,
            'returns_mean': returns.mean(),
            'returns_std': returns.std(),
            'advantages_mean': advantages.mean(),
            'advantages_std': advantages.std()
        }
        
        return training_stats
    
    def train_step(self):
        """
        Perform one training step (rollout + update)
        
        Returns:
            episode_rewards: List of episode rewards from rollout
            training_stats: Dictionary of training statistics
        """
        # Collect rollout
        rollout_data, episode_rewards = self.collect_rollout()
        
        # Perform PPO update
        training_stats = self.ppo_update(rollout_data)
        
        # Update episode rewards
        self.episode_rewards.extend(episode_rewards)
        
        # Update training statistics
        for key, value in training_stats.items():
            self.training_stats[key].append(value)
        
        return episode_rewards, training_stats
    
    def get_action(self, obs):
        """
        Get action for a single observation (for evaluation)
        
        Args:
            obs: Single observation
            
        Returns:
            action: Selected action
        """
        self.networks.set_eval_mode()
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.DEVICE)
        
        with torch.no_grad():
            action = self.networks.actor.get_action(obs_tensor)
        
        return action.cpu().item()
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.networks.save_models(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.networks.load_models(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_training_stats(self):
        """Get training statistics"""
        return dict(self.training_stats)
    
    def get_episode_rewards(self):
        """Get episode rewards"""
        return self.episode_rewards.copy()