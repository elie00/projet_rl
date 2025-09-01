#!/usr/bin/env python3
"""
Quick training demo with dashboard generation
"""
import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym
import flappy_bird_gymnasium
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models import PPONetworks
from ppo_agent import PPOAgent
from utils import set_seed, Logger, EpisodeTracker
from metrics_dashboard import MetricsDashboard

def quick_training_demo(max_episodes=100):
    """Run a quick training demo with dashboard generation"""
    
    print("🚀 Démarrage de l'entraînement PPO rapide...")
    
    # Configuration
    config = Config()
    config.TOTAL_EPISODES = max_episodes
    config.MAX_STEPS_PER_ROLLOUT = 2048  # Reduced for demo
    config.EVAL_FREQUENCY = 25  # More frequent evaluation
    config.SAVE_FREQUENCY = 25  # More frequent saves
    
    # Set seed
    set_seed(config.SEED)
    
    # Create directories
    config.create_directories()
    
    # Create environment
    env = gym.make(config.ENV_NAME, use_lidar=config.USE_LIDAR, render_mode="rgb_array")
    print(f"🎮 Environnement créé: {config.ENV_NAME}")
    
    # Create agent
    agent = PPOAgent(env, config)
    
    # Initialize dashboard
    dashboard = MetricsDashboard()
    
    # Initialize tracking
    episode_tracker = EpisodeTracker(window_size=50)
    
    print(f"📊 Entraînement sur {max_episodes} épisodes...")
    
    best_reward = float('-inf')
    
    try:
        for episode in range(max_episodes):
            episode_start_time = time.time()
            
            # Perform training step
            episode_rewards, training_stats = agent.train_step()
            
            # Update tracking
            for reward in episode_rewards:
                episode_tracker.add_episode(reward, 1)
            
            # Collect metrics for dashboard
            if episode % 5 == 0:  # Every 5 episodes
                dashboard.collect_metrics(
                    agent=agent,
                    episode=episode,
                    training_stats=training_stats,
                    environment_info={'config': config.__dict__}
                )
            
            # Print progress every 10 episodes
            if episode % 10 == 0 and episode > 0:
                episode_stats = episode_tracker.get_stats()
                episode_time = time.time() - episode_start_time
                
                print(f"Épisode {episode}/{max_episodes}")
                print(f"  Récompense Moy (50 derniers): {episode_stats.get('episode_reward_mean', 0):.2f}")
                print(f"  Récompense Max: {episode_stats.get('episode_reward_max', 0):.2f}")
                print(f"  Loss Acteur: {training_stats['actor_loss']:.4f}")
                print(f"  Loss Critique: {training_stats['critic_loss']:.4f}")
                print(f"  Temps par épisode: {episode_time:.2f}s")
                print("-" * 40)
            
            # Generate dashboard periodically
            if episode > 0 and episode % 25 == 0:
                print(f"📊 Génération du dashboard à l'épisode {episode}...")
                try:
                    dashboard_path = dashboard.create_comprehensive_dashboard()
                    print(f"✅ Dashboard généré: {dashboard_path}")
                except Exception as e:
                    print(f"❌ Erreur génération dashboard: {e}")
            
            # Track best reward
            if episode_rewards:
                current_reward = np.mean(episode_rewards)
                if current_reward > best_reward:
                    best_reward = current_reward
                    print(f"🎯 Nouveau meilleur score: {best_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Entraînement interrompu par l'utilisateur")
    
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Generate final dashboard
        print("\n🎨 Génération du dashboard final...")
        try:
            final_dashboard_path = dashboard.create_comprehensive_dashboard()
            summary_path = dashboard.export_metrics_summary()
            
            print(f"✅ Dashboard final: {final_dashboard_path}")
            print(f"📋 Résumé: {summary_path}")
            
            # Generate presentation slides
            print("🎯 Génération des slides de présentation...")
            from demo_dashboard import create_demo_dashboard
            
            # Use actual training data for final slides
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if dashboard.training_history:
                episodes = [m['episode'] for m in dashboard.training_history]
                
                # Quick slide generation
                import matplotlib.pyplot as plt
                
                # Single comprehensive slide
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('PPO Flappy Bird - Résultats d\'Entraînement Réel', fontsize=18, fontweight='bold')
                
                dashboard._plot_episode_rewards(ax1, episodes)
                dashboard._plot_training_losses(ax2, episodes)
                dashboard._plot_performance_metrics(ax3, episodes)
                dashboard._plot_policy_evolution(ax4, episodes)
                
                plt.tight_layout()
                final_slide_path = f'dashboard_exports/training_results_{timestamp}.png'
                plt.savefig(final_slide_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"🎯 Slide finale: {final_slide_path}")
            
        except Exception as e:
            print(f"❌ Erreur génération dashboard final: {e}")
        
        finally:
            env.close()
    
    print(f"\n🎉 Entraînement terminé! Meilleur score: {best_reward:.2f}")
    print(f"📁 Fichiers générés dans: dashboard_exports/")

if __name__ == "__main__":
    quick_training_demo(100)