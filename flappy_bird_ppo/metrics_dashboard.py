"""
Advanced Metrics Dashboard for Flappy Bird PPO
Generates comprehensive training and evaluation metrics with visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MetricsDashboard:
    """
    Comprehensive metrics dashboard for PPO training analysis
    """
    
    def __init__(self, metrics_dir: str = "metrics", export_dir: str = "dashboard_exports"):
        self.metrics_dir = metrics_dir
        self.export_dir = export_dir
        self.metrics_data = {}
        self.training_history = []
        self.evaluation_history = []
        
        # Create directories
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)
        
    def collect_metrics(self, agent, episode: int, training_stats: Dict, 
                       evaluation_stats: Dict = None, environment_info: Dict = None):
        """
        Collect comprehensive metrics from training step
        
        Args:
            agent: PPO agent instance
            episode: Current episode number
            training_stats: Training statistics
            evaluation_stats: Optional evaluation statistics
            environment_info: Optional environment information
        """
        timestamp = datetime.now().isoformat()
        
        # Core training metrics
        metrics = {
            'timestamp': timestamp,
            'episode': episode,
            'training': training_stats.copy(),
            'episode_rewards': agent.get_episode_rewards()[-100:],  # Last 100 episodes
        }
        
        # Add evaluation metrics if available
        if evaluation_stats:
            metrics['evaluation'] = evaluation_stats
            
        # Add environment info if available
        if environment_info:
            metrics['environment'] = environment_info
            
        # Compute derived metrics
        metrics.update(self._compute_derived_metrics(agent, training_stats))
        
        self.training_history.append(metrics)
        
        # Save metrics
        self._save_metrics(metrics, episode)
        
    def _compute_derived_metrics(self, agent, training_stats: Dict) -> Dict:
        """Compute additional derived metrics"""
        episode_rewards = agent.get_episode_rewards()
        
        if len(episode_rewards) == 0:
            return {}
            
        recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
        
        derived = {
            'performance_metrics': {
                'reward_trend_slope': self._compute_trend_slope(recent_rewards),
                'reward_stability': np.std(recent_rewards) if len(recent_rewards) > 1 else 0,
                'learning_efficiency': self._compute_learning_efficiency(episode_rewards),
                'convergence_indicator': self._compute_convergence(recent_rewards)
            },
            'training_efficiency': {
                'gradient_norm': training_stats.get('gradient_norm', 0),
                'policy_change_rate': abs(training_stats.get('kl_divergence', 0)),
                'value_estimation_accuracy': 1.0 / (1.0 + training_stats.get('critic_loss', 1)),
            }
        }
        
        return derived
    
    def _compute_trend_slope(self, rewards: List[float]) -> float:
        """Compute trend slope of rewards"""
        if len(rewards) < 2:
            return 0.0
        x = np.arange(len(rewards))
        return np.polyfit(x, rewards, 1)[0]
    
    def _compute_learning_efficiency(self, rewards: List[float]) -> float:
        """Compute learning efficiency metric"""
        if len(rewards) < 10:
            return 0.0
        
        # Compare improvement rate vs episodes
        early_avg = np.mean(rewards[:len(rewards)//4]) if len(rewards) >= 4 else 0
        recent_avg = np.mean(rewards[-len(rewards)//4:]) if len(rewards) >= 4 else 0
        
        if early_avg == 0:
            return 0.0
        return (recent_avg - early_avg) / len(rewards)
    
    def _compute_convergence(self, rewards: List[float]) -> float:
        """Compute convergence indicator (lower variance = more convergent)"""
        if len(rewards) < 5:
            return 1.0
        
        recent_var = np.var(rewards[-20:]) if len(rewards) >= 20 else np.var(rewards)
        return 1.0 / (1.0 + recent_var)
    
    def _save_metrics(self, metrics: Dict, episode: int):
        """Save metrics to JSON file"""
        filename = os.path.join(self.metrics_dir, f"metrics_episode_{episode}.json")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        metrics_clean = convert_numpy_types(metrics)
        
        with open(filename, 'w') as f:
            json.dump(metrics_clean, f, indent=2)
    
    def load_training_history(self, metrics_dir: str = None) -> List[Dict]:
        """Load training history from saved metrics"""
        metrics_dir = metrics_dir or self.metrics_dir
        
        history = []
        for filename in sorted(os.listdir(metrics_dir)):
            if filename.startswith("metrics_episode_") and filename.endswith(".json"):
                filepath = os.path.join(metrics_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        metrics = json.load(f)
                    history.append(metrics)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return history
    
    def create_comprehensive_dashboard(self, save_individual: bool = True) -> str:
        """
        Create comprehensive dashboard with all metrics
        
        Args:
            save_individual: Whether to save individual plots
            
        Returns:
            Path to saved dashboard
        """
        if not self.training_history:
            self.training_history = self.load_training_history()
        
        if not self.training_history:
            print("No training history found!")
            return None
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('PPO Flappy Bird - Training Dashboard', fontsize=24, fontweight='bold')
        
        # Extract data for plotting
        episodes = [m['episode'] for m in self.training_history]
        
        # 1. Episode Rewards Over Time (Main plot)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_episode_rewards(ax1, episodes)
        
        # 2. Training Losses
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_training_losses(ax2, episodes)
        
        # 3. Performance Metrics
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_performance_metrics(ax3, episodes)
        
        # 4. Policy Evolution
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_policy_evolution(ax4, episodes)
        
        # 5. Learning Efficiency
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_learning_efficiency(ax5, episodes)
        
        # 6. Convergence Analysis
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_convergence_analysis(ax6, episodes)
        
        # 7. Statistical Summary
        ax7 = fig.add_subplot(gs[3, :2])
        self._plot_statistical_summary(ax7)
        
        # 8. Recent Performance Heatmap
        ax8 = fig.add_subplot(gs[3, 2:])
        self._plot_performance_heatmap(ax8)
        
        # Save dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dashboard_path = os.path.join(self.export_dir, f'ppo_dashboard_{timestamp}.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Dashboard saved to: {dashboard_path}")
        
        if save_individual:
            self._save_individual_plots()
        
        plt.close(fig)
        return dashboard_path
    
    def _plot_episode_rewards(self, ax, episodes):
        """Plot episode rewards with trend analysis"""
        # Get all episode rewards from history
        all_rewards = []
        episode_nums = []
        
        for i, metrics in enumerate(self.training_history):
            if 'episode_rewards' in metrics:
                rewards = metrics['episode_rewards']
                if rewards:
                    all_rewards.extend(rewards)
                    episode_nums.extend([metrics['episode']] * len(rewards))
        
        if not all_rewards:
            ax.text(0.5, 0.5, 'No episode rewards data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        # Plot raw rewards
        ax.plot(episode_nums, all_rewards, alpha=0.3, color='lightblue', linewidth=0.5, label='Raw Rewards')
        
        # Moving average
        if len(all_rewards) > 10:
            window_size = min(50, len(all_rewards) // 10)
            moving_avg = pd.Series(all_rewards).rolling(window=window_size, center=True).mean()
            ax.plot(episode_nums, moving_avg, color='blue', linewidth=2, label=f'Moving Average ({window_size})')
        
        # Trend line
        if len(all_rewards) > 5:
            z = np.polyfit(range(len(all_rewards)), all_rewards, 1)
            p = np.poly1d(z)
            ax.plot(episode_nums, p(range(len(all_rewards))), 
                   color='red', linewidth=2, linestyle='--', label='Trend')
        
        ax.set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_losses(self, ax, episodes):
        """Plot training losses"""
        actor_losses = []
        critic_losses = []
        
        for metrics in self.training_history:
            if 'training' in metrics:
                actor_losses.append(metrics['training'].get('actor_loss', 0))
                critic_losses.append(metrics['training'].get('critic_loss', 0))
        
        ax.plot(episodes, actor_losses, label='Actor Loss', linewidth=2)
        ax.plot(episodes, critic_losses, label='Critic Loss', linewidth=2)
        
        ax.set_title('Training Losses', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_metrics(self, ax, episodes):
        """Plot derived performance metrics"""
        trends = []
        stability = []
        efficiency = []
        
        for metrics in self.training_history:
            if 'performance_metrics' in metrics:
                perf = metrics['performance_metrics']
                trends.append(perf.get('reward_trend_slope', 0))
                stability.append(perf.get('reward_stability', 0))
                efficiency.append(perf.get('learning_efficiency', 0))
        
        if trends:
            ax.plot(episodes, trends, label='Reward Trend Slope', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(episodes, efficiency, label='Learning Efficiency', 
                    color='orange', linewidth=2, linestyle='--')
            
            ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Trend Slope')
            ax2.set_ylabel('Learning Efficiency')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    def _plot_policy_evolution(self, ax, episodes):
        """Plot policy evolution metrics"""
        entropy = []
        kl_div = []
        
        for metrics in self.training_history:
            if 'training' in metrics:
                entropy.append(metrics['training'].get('entropy', 0))
                kl_div.append(metrics['training'].get('kl_divergence', 0))
        
        ax.plot(episodes, entropy, label='Policy Entropy', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(episodes, kl_div, label='KL Divergence', 
                color='red', linewidth=2, linestyle='--')
        
        ax.set_title('Policy Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Entropy')
        ax2.set_ylabel('KL Divergence')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_efficiency(self, ax, episodes):
        """Plot learning efficiency over time"""
        efficiency = []
        convergence = []
        
        for metrics in self.training_history:
            if 'performance_metrics' in metrics:
                perf = metrics['performance_metrics']
                efficiency.append(perf.get('learning_efficiency', 0))
                convergence.append(perf.get('convergence_indicator', 0))
        
        if efficiency:
            ax.plot(episodes, efficiency, label='Learning Efficiency', linewidth=2)
            ax.plot(episodes, convergence, label='Convergence Indicator', linewidth=2)
            
            ax.set_title('Learning Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Metric Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_convergence_analysis(self, ax, episodes):
        """Plot convergence analysis"""
        returns_mean = []
        returns_std = []
        
        for metrics in self.training_history:
            if 'training' in metrics:
                returns_mean.append(metrics['training'].get('returns_mean', 0))
                returns_std.append(metrics['training'].get('returns_std', 0))
        
        if returns_mean:
            ax.plot(episodes, returns_mean, label='Returns Mean', linewidth=2)
            ax.fill_between(episodes, 
                           np.array(returns_mean) - np.array(returns_std),
                           np.array(returns_mean) + np.array(returns_std),
                           alpha=0.3, label='±1 Std Dev')
            
            ax.set_title('Returns Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Returns')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_statistical_summary(self, ax):
        """Plot statistical summary of latest metrics"""
        if not self.training_history:
            return
            
        latest = self.training_history[-1]
        
        # Create text summary
        summary_text = f"""
LATEST TRAINING STATISTICS
Episode: {latest.get('episode', 'N/A')}
Actor Loss: {latest.get('training', {}).get('actor_loss', 0):.4f}
Critic Loss: {latest.get('training', {}).get('critic_loss', 0):.4f}
Policy Entropy: {latest.get('training', {}).get('entropy', 0):.4f}
KL Divergence: {latest.get('training', {}).get('kl_divergence', 0):.4f}
        """
        
        if 'performance_metrics' in latest:
            perf = latest['performance_metrics']
            summary_text += f"""
Reward Trend: {perf.get('reward_trend_slope', 0):.4f}
Learning Efficiency: {perf.get('learning_efficiency', 0):.4f}
Convergence: {perf.get('convergence_indicator', 0):.4f}
            """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Current Statistics', fontsize=14, fontweight='bold')
    
    def _plot_performance_heatmap(self, ax):
        """Plot performance heatmap of recent episodes"""
        if len(self.training_history) < 10:
            ax.text(0.5, 0.5, 'Not enough data for heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get last 10 episodes of data
        recent_episodes = self.training_history[-10:]
        
        # Create heatmap data
        metrics_names = ['Actor Loss', 'Critic Loss', 'Entropy', 'KL Div']
        heatmap_data = []
        
        for metrics in recent_episodes:
            if 'training' in metrics:
                training = metrics['training']
                row = [
                    training.get('actor_loss', 0),
                    training.get('critic_loss', 0),
                    training.get('entropy', 0),
                    training.get('kl_divergence', 0)
                ]
                heatmap_data.append(row)
        
        if heatmap_data:
            heatmap_data = np.array(heatmap_data)
            # Normalize each column
            for i in range(heatmap_data.shape[1]):
                col = heatmap_data[:, i]
                if col.std() > 0:
                    heatmap_data[:, i] = (col - col.mean()) / col.std()
            
            im = ax.imshow(heatmap_data.T, cmap='RdYlBu_r', aspect='auto')
            
            ax.set_xticks(range(len(recent_episodes)))
            ax.set_xticklabels([f"Ep {m['episode']}" for m in recent_episodes])
            ax.set_yticks(range(len(metrics_names)))
            ax.set_yticklabels(metrics_names)
            ax.set_title('Recent Performance Heatmap', fontsize=14, fontweight='bold')
            
            plt.colorbar(im, ax=ax, shrink=0.6)
    
    def _save_individual_plots(self):
        """Save individual plots for detailed analysis"""
        episodes = [m['episode'] for m in self.training_history]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Individual reward plot
        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_episode_rewards(ax, episodes)
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, f'rewards_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual loss plot
        fig, ax = plt.subplots(figsize=(12, 8))
        self._plot_training_losses(ax, episodes)
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, f'losses_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual plots saved to {self.export_dir}")
    
    def export_metrics_summary(self) -> str:
        """Export comprehensive metrics summary"""
        if not self.training_history:
            return "No training history available"
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join(self.export_dir, f'metrics_summary_{timestamp}.txt')
        
        with open(summary_path, 'w') as f:
            f.write("PPO FLAPPY BIRD - TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Training Episodes: {len(self.training_history)}\n")
            
            if self.training_history:
                latest = self.training_history[-1]
                f.write(f"Latest Episode: {latest.get('episode', 'N/A')}\n")
                
                if 'training' in latest:
                    training = latest['training']
                    f.write(f"Final Actor Loss: {training.get('actor_loss', 0):.4f}\n")
                    f.write(f"Final Critic Loss: {training.get('critic_loss', 0):.4f}\n")
                    f.write(f"Final Policy Entropy: {training.get('entropy', 0):.4f}\n")
                
                # Performance analysis
                if 'performance_metrics' in latest:
                    perf = latest['performance_metrics']
                    f.write(f"\nPerformance Analysis:\n")
                    f.write(f"Reward Trend Slope: {perf.get('reward_trend_slope', 0):.4f}\n")
                    f.write(f"Learning Efficiency: {perf.get('learning_efficiency', 0):.4f}\n")
                    f.write(f"Convergence Indicator: {perf.get('convergence_indicator', 0):.4f}\n")
        
        return summary_path

# Integration function for easy use
def create_dashboard_from_agent(agent, config, episode: int, training_stats: Dict, 
                               evaluation_stats: Dict = None) -> str:
    """
    Convenience function to create dashboard from agent data
    
    Args:
        agent: PPO agent
        config: Training configuration
        episode: Current episode
        training_stats: Training statistics
        evaluation_stats: Optional evaluation statistics
        
    Returns:
        Path to generated dashboard
    """
    dashboard = MetricsDashboard()
    
    # Collect current metrics
    dashboard.collect_metrics(
        agent=agent,
        episode=episode,
        training_stats=training_stats,
        evaluation_stats=evaluation_stats,
        environment_info={'config': config.__dict__}
    )
    
    # Create dashboard
    return dashboard.create_comprehensive_dashboard()

if __name__ == "__main__":
    # Example usage
    print("Metrics Dashboard module loaded successfully!")
    dashboard = MetricsDashboard()
    print(f"Dashboard export directory: {dashboard.export_dir}")