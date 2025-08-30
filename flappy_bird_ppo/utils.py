"""
Utility functions for PPO training and evaluation
Includes evaluation, logging, visualization, and video recording utilities
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
import os
import random
from collections import deque
from typing import List, Dict, Tuple
import wandb
from torch.utils.tensorboard import SummaryWriter

from config import Config


def set_seed(seed: int):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    """
    Logging utility that supports both Weights & Biases and TensorBoard
    """
    
    def __init__(self, config: Config, project_name: str = None, run_name: str = None):
        self.config = config
        self.use_wandb = config.USE_WANDB
        self.step_count = 0
        
        # Initialize TensorBoard
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=config.LOG_DIR)
        
        # Initialize Weights & Biases if enabled
        if self.use_wandb:
            try:
                wandb.init(
                    project=project_name or config.WANDB_PROJECT,
                    entity=config.WANDB_ENTITY,
                    name=run_name,
                    config=self._get_config_dict(),
                    reinit=True
                )
                print("Weights & Biases initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Weights & Biases: {e}")
                self.use_wandb = False
    
    def _get_config_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        config_dict = {}
        for attr_name in dir(self.config):
            if not attr_name.startswith('_') and not callable(getattr(self.config, attr_name)):
                value = getattr(self.config, attr_name)
                if isinstance(value, (int, float, str, bool, list)):
                    config_dict[attr_name] = value
        return config_dict
    
    def log_scalar(self, name: str, value: float, step: int = None):
        """Log scalar value"""
        step = step or self.step_count
        
        # Log to TensorBoard
        self.tb_writer.add_scalar(name, value, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb.log({name: value}, step=step)
    
    def log_scalars(self, scalar_dict: Dict[str, float], step: int = None):
        """Log multiple scalar values"""
        step = step or self.step_count
        
        for name, value in scalar_dict.items():
            self.log_scalar(name, value, step)
    
    def log_histogram(self, name: str, values: np.ndarray, step: int = None):
        """Log histogram of values"""
        step = step or self.step_count
        
        # Log to TensorBoard
        self.tb_writer.add_histogram(name, values, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb.log({f"{name}_histogram": wandb.Histogram(values)}, step=step)
    
    def log_video(self, video_path: str, step: int = None):
        """Log video file"""
        step = step or self.step_count
        
        if self.use_wandb and os.path.exists(video_path):
            try:
                wandb.log({"evaluation_video": wandb.Video(video_path)}, step=step)
            except Exception as e:
                print(f"Failed to log video to W&B: {e}")
    
    def increment_step(self):
        """Increment global step counter"""
        self.step_count += 1
    
    def close(self):
        """Close logging resources"""
        self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()


class EpisodeTracker:
    """
    Track episode statistics with rolling averages
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.total_episodes = 0
    
    def add_episode(self, reward: float, length: int):
        """Add episode data"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.total_episodes += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics"""
        if not self.episode_rewards:
            return {}
        
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        
        return {
            'episode_reward_mean': np.mean(rewards),
            'episode_reward_std': np.std(rewards),
            'episode_reward_min': np.min(rewards),
            'episode_reward_max': np.max(rewards),
            'episode_length_mean': np.mean(lengths),
            'total_episodes': self.total_episodes
        }


def evaluate_agent(agent, env, num_episodes: int = 10, render: bool = False, 
                  record_video: bool = False, video_path: str = None) -> Dict[str, float]:
    """
    Evaluate trained agent performance
    
    Args:
        agent: Trained PPO agent
        env: Environment to evaluate on
        num_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation
        record_video: Whether to record evaluation video
        video_path: Path to save evaluation video
        
    Returns:
        evaluation_stats: Dictionary with evaluation statistics
    """
    
    # Setup video recording if requested
    if record_video and video_path:
        env = RecordVideo(
            env, 
            video_folder=os.path.dirname(video_path),
            name_prefix=os.path.splitext(os.path.basename(video_path))[0],
            episode_trigger=lambda x: True  # Record all episodes
        )
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    agent.networks.set_eval_mode()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            if render:
                env.render()
            
            # Get action from agent
            action = agent.get_action(obs)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check success (getting more than threshold pipes)
        if episode_reward > agent.config.SUCCESS_THRESHOLD:
            success_count += 1
        
        print(f"Evaluation Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Close video recording
    if record_video:
        env.close()
    
    # Calculate statistics
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    
    evaluation_stats = {
        'eval_reward_mean': np.mean(rewards),
        'eval_reward_std': np.std(rewards),
        'eval_reward_min': np.min(rewards),
        'eval_reward_max': np.max(rewards),
        'eval_length_mean': np.mean(lengths),
        'eval_length_std': np.std(lengths),
        'success_rate': success_count / num_episodes,
        'num_episodes': num_episodes
    }
    
    return evaluation_stats


def evaluate_random_baseline(env, num_episodes: int = 100) -> Dict[str, float]:
    """
    Evaluate random baseline performance
    
    Args:
        env: Environment to evaluate on
        num_episodes: Number of episodes for evaluation
        
    Returns:
        baseline_stats: Dictionary with baseline statistics
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Take random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # Calculate statistics
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    
    baseline_stats = {
        'baseline_reward_mean': np.mean(rewards),
        'baseline_reward_std': np.std(rewards),
        'baseline_reward_min': np.min(rewards),
        'baseline_reward_max': np.max(rewards),
        'baseline_length_mean': np.mean(lengths),
        'baseline_length_std': np.std(lengths)
    }
    
    return baseline_stats


def plot_training_curves(episode_rewards: List[float], training_stats: Dict[str, List[float]], 
                        save_path: str = None, show: bool = True):
    """
    Plot training curves for visualization
    
    Args:
        episode_rewards: List of episode rewards
        training_stats: Dictionary with training statistics
        save_path: Path to save the plot
        show: Whether to show the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PPO Training Curves', fontsize=16)
    
    # Plot episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue')
    if len(episode_rewards) > 100:
        # Add moving average
        window_size = min(100, len(episode_rewards) // 10)
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0, 0].plot(range(window_size-1, len(episode_rewards)), moving_avg, color='red', linewidth=2)
    
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot actor loss
    if 'actor_loss' in training_stats:
        axes[0, 1].plot(training_stats['actor_loss'])
        axes[0, 1].set_title('Actor Loss')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot critic loss
    if 'critic_loss' in training_stats:
        axes[1, 0].plot(training_stats['critic_loss'])
        axes[1, 0].set_title('Critic Loss')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot entropy
    if 'entropy' in training_stats:
        axes[1, 1].plot(training_stats['entropy'])
        axes[1, 1].set_title('Policy Entropy')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_evaluation_report(agent_stats: Dict[str, float], baseline_stats: Dict[str, float]) -> str:
    """
    Create evaluation report comparing agent performance to baseline
    
    Args:
        agent_stats: Agent evaluation statistics
        baseline_stats: Random baseline statistics
        
    Returns:
        report: Formatted evaluation report string
    """
    report = "=" * 60 + "\n"
    report += "EVALUATION REPORT\n"
    report += "=" * 60 + "\n\n"
    
    # Agent performance
    report += "TRAINED AGENT PERFORMANCE:\n"
    report += f"  Average Reward: {agent_stats['eval_reward_mean']:.2f} ± {agent_stats['eval_reward_std']:.2f}\n"
    report += f"  Best Reward: {agent_stats['eval_reward_max']:.2f}\n"
    report += f"  Worst Reward: {agent_stats['eval_reward_min']:.2f}\n"
    report += f"  Success Rate: {agent_stats['success_rate']:.1%}\n"
    report += f"  Average Episode Length: {agent_stats['eval_length_mean']:.1f}\n\n"
    
    # Baseline performance
    report += "RANDOM BASELINE PERFORMANCE:\n"
    report += f"  Average Reward: {baseline_stats['baseline_reward_mean']:.2f} ± {baseline_stats['baseline_reward_std']:.2f}\n"
    report += f"  Best Reward: {baseline_stats['baseline_reward_max']:.2f}\n"
    report += f"  Worst Reward: {baseline_stats['baseline_reward_min']:.2f}\n"
    report += f"  Average Episode Length: {baseline_stats['baseline_length_mean']:.1f}\n\n"
    
    # Comparison
    reward_improvement = agent_stats['eval_reward_mean'] - baseline_stats['baseline_reward_mean']
    reward_improvement_pct = (reward_improvement / abs(baseline_stats['baseline_reward_mean'])) * 100
    
    report += "PERFORMANCE COMPARISON:\n"
    report += f"  Reward Improvement: {reward_improvement:+.2f} ({reward_improvement_pct:+.1f}%)\n"
    
    if agent_stats['eval_reward_mean'] > baseline_stats['baseline_reward_mean']:
        report += "  ✓ Agent outperforms random baseline!\n"
    else:
        report += "  ✗ Agent underperforms compared to baseline\n"
    
    if agent_stats.get('success_rate', 0) > 0:
        report += f"  ✓ Agent achieved {agent_stats['success_rate']:.1%} success rate!\n"
    else:
        report += "  ✗ Agent did not achieve success threshold\n"
    
    report += "\n" + "=" * 60
    
    return report


def save_model_checkpoint(agent, filepath: str, episode: int, stats: Dict[str, float]):
    """
    Save model checkpoint with metadata
    
    Args:
        agent: PPO agent to save
        filepath: Path to save checkpoint
        episode: Current episode number
        stats: Current training statistics
    """
    checkpoint = {
        'actor_state_dict': agent.networks.actor.state_dict(),
        'critic_state_dict': agent.networks.critic.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode': episode,
        'config': agent.config.__dict__,
        'stats': stats
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(agent, filepath: str) -> Tuple[int, Dict[str, float]]:
    """
    Load model checkpoint
    
    Args:
        agent: PPO agent to load into
        filepath: Path to checkpoint file
        
    Returns:
        episode: Episode number from checkpoint
        stats: Statistics from checkpoint
    """
    checkpoint = torch.load(filepath, map_location=agent.config.DEVICE, weights_only=False)
    
    agent.networks.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.networks.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    episode = checkpoint.get('episode', 0)
    stats = checkpoint.get('stats', {})
    
    print(f"Checkpoint loaded from {filepath}")
    return episode, stats