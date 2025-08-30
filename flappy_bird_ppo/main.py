"""
Main training and evaluation script for Flappy Bird PPO
Supports training, evaluation, and rendering modes
"""
import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
import flappy_bird_gymnasium
from gymnasium.wrappers import RecordEpisodeStatistics

from config import Config
from models import PPONetworks
from ppo_agent import PPOAgent
from utils import (
    set_seed, Logger, EpisodeTracker, evaluate_agent, 
    evaluate_random_baseline, plot_training_curves, 
    create_evaluation_report, save_model_checkpoint
)


def create_environment(config: Config, record_stats: bool = True):
    """
    Create and configure the Flappy Bird environment
    
    Args:
        config: Configuration object
        record_stats: Whether to wrap with RecordEpisodeStatistics
        
    Returns:
        env: Configured environment
    """
    env = gym.make(config.ENV_NAME, use_lidar=config.USE_LIDAR, render_mode="rgb_array")
    
    if record_stats:
        env = RecordEpisodeStatistics(env)
    
    return env


def train_agent(config: Config, resume_from: str = None):
    """
    Train PPO agent on Flappy Bird
    
    Args:
        config: Configuration object
        resume_from: Path to checkpoint to resume from
    """
    print("Starting PPO training for Flappy Bird")
    print(f"Device: {config.DEVICE}")
    print(f"Total episodes: {config.TOTAL_EPISODES}")
    
    # Set random seed for reproducibility
    set_seed(config.SEED)
    
    # Create directories
    config.create_directories()
    
    # Create environment
    env = create_environment(config, record_stats=True)
    print(f"Environment: {config.ENV_NAME}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create agent
    agent = PPOAgent(env, config)
    
    # Initialize logging
    run_name = f"ppo_flappy_bird_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = Logger(config, run_name=run_name)
    episode_tracker = EpisodeTracker(window_size=100)
    
    # Resume from checkpoint if specified
    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        try:
            checkpoint = torch.load(resume_from, map_location=config.DEVICE, weights_only=False)
            agent.networks.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.networks.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint.get('episode', 0)
            print(f"Resumed from episode {start_episode}")
        except Exception as e:
            print(f"Failed to resume from checkpoint: {e}")
            print("Starting fresh training")
    
    # Evaluate random baseline
    print("\nEvaluating random baseline...")
    baseline_stats = evaluate_random_baseline(env, num_episodes=100)
    print(f"Random baseline: {baseline_stats['baseline_reward_mean']:.2f} ± {baseline_stats['baseline_reward_std']:.2f}")
    logger.log_scalars(baseline_stats, step=0)
    
    # Training loop
    print("\nStarting training...")
    best_reward = float('-inf')
    total_steps = 0
    
    try:
        for episode in range(start_episode, config.TOTAL_EPISODES):
            episode_start_time = time.time()
            
            # Perform training step
            episode_rewards, training_stats = agent.train_step()
            
            # Update tracking
            for reward in episode_rewards:
                episode_tracker.add_episode(reward, 1)  # Length not tracked in rollout
            
            total_steps += config.MAX_STEPS_PER_ROLLOUT
            
            # Logging
            if episode % 10 == 0 or episode_rewards:
                # Log episode statistics
                episode_stats = episode_tracker.get_stats()
                logger.log_scalars(episode_stats, step=episode)
                
                # Log training statistics
                training_log = {f"train/{k}": v for k, v in training_stats.items()}
                logger.log_scalars(training_log, step=episode)
                
                # Log recent episode rewards
                if episode_rewards:
                    recent_reward = np.mean(episode_rewards)
                    logger.log_scalar("recent_episode_reward", recent_reward, step=episode)
                
                logger.increment_step()
            
            # Print progress
            if episode % 50 == 0 and episode > 0:
                episode_stats = episode_tracker.get_stats()
                episode_time = time.time() - episode_start_time
                
                print(f"Episode {episode}/{config.TOTAL_EPISODES}")
                print(f"  Avg Reward (last 100): {episode_stats.get('episode_reward_mean', 0):.2f}")
                print(f"  Max Reward (last 100): {episode_stats.get('episode_reward_max', 0):.2f}")
                print(f"  Actor Loss: {training_stats['actor_loss']:.4f}")
                print(f"  Critic Loss: {training_stats['critic_loss']:.4f}")
                print(f"  Time per episode: {episode_time:.2f}s")
                print(f"  Total steps: {total_steps}")
                print("-" * 50)
            
            # Evaluation
            if episode > 0 and episode % config.EVAL_FREQUENCY == 0:
                print(f"\nEvaluating at episode {episode}...")
                
                video_path = config.get_video_path(episode) if config.RECORD_VIDEO else None
                eval_stats = evaluate_agent(
                    agent, env, 
                    num_episodes=config.EVAL_EPISODES,
                    render=config.RENDER_EVAL,
                    record_video=config.RECORD_VIDEO,
                    video_path=video_path
                )
                
                # Log evaluation results
                eval_log = {f"eval/{k}": v for k, v in eval_stats.items()}
                logger.log_scalars(eval_log, step=episode)
                
                # Log video if recorded
                if config.RECORD_VIDEO and video_path and os.path.exists(video_path):
                    logger.log_video(video_path, step=episode)
                
                print(f"Evaluation Results:")
                print(f"  Mean Reward: {eval_stats['eval_reward_mean']:.2f} ± {eval_stats['eval_reward_std']:.2f}")
                print(f"  Max Reward: {eval_stats['eval_reward_max']:.2f}")
                print(f"  Success Rate: {eval_stats['success_rate']:.1%}")
                
                # Save best model
                if eval_stats['eval_reward_mean'] > best_reward:
                    best_reward = eval_stats['eval_reward_mean']
                    best_model_path = os.path.join(config.MODEL_DIR, "best_model.pt")
                    agent.save_model(best_model_path)
                    print(f"New best model saved! Reward: {best_reward:.2f}")
            
            # Save periodic checkpoints
            if episode > 0 and episode % config.SAVE_FREQUENCY == 0:
                checkpoint_path = config.get_model_path(episode)
                episode_stats = episode_tracker.get_stats()
                save_model_checkpoint(agent, checkpoint_path, episode, episode_stats)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise e
    
    finally:
        # Final evaluation
        print("\nPerforming final evaluation...")
        final_eval_stats = evaluate_agent(
            agent, env, 
            num_episodes=config.EVAL_EPISODES * 2,  # More episodes for final eval
            render=False,
            record_video=True,
            video_path=os.path.join(config.VIDEO_DIR, "final_evaluation.mp4")
        )
        
        # Create evaluation report
        report = create_evaluation_report(final_eval_stats, baseline_stats)
        print(report)
        
        # Save final model
        final_model_path = os.path.join(config.MODEL_DIR, "final_model.pt")
        agent.save_model(final_model_path)
        
        # Plot training curves
        episode_rewards = agent.get_episode_rewards()
        training_stats = agent.get_training_stats()
        
        if episode_rewards:
            plot_path = os.path.join(config.LOG_DIR, "training_curves.png")
            plot_training_curves(episode_rewards, training_stats, save_path=plot_path, show=False)
        
        # Save final report
        report_path = os.path.join(config.LOG_DIR, "training_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
            f.write("\n\nTraining Configuration:\n")
            f.write(str(config))
        
        # Clean up
        env.close()
        logger.close()
        
        print(f"\nTraining completed!")
        print(f"Final model saved to: {final_model_path}")
        print(f"Training report saved to: {report_path}")


def evaluate_model(model_path: str, config: Config, num_episodes: int = 10, render: bool = False):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to trained model
        config: Configuration object
        num_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation
    """
    print(f"Evaluating model: {model_path}")
    
    # Set random seed
    set_seed(config.SEED)
    
    # Create environment
    env = create_environment(config, record_stats=False)
    
    # Create agent and load model
    agent = PPOAgent(env, config)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    agent.load_model(model_path)
    
    # Evaluate agent
    eval_stats = evaluate_agent(
        agent, env, 
        num_episodes=num_episodes,
        render=render,
        record_video=True,
        video_path=os.path.join(config.VIDEO_DIR, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    )
    
    # Evaluate baseline for comparison
    baseline_stats = evaluate_random_baseline(env, num_episodes=100)
    
    # Create and print report
    report = create_evaluation_report(eval_stats, baseline_stats)
    print(report)
    
    env.close()


def watch_agent(model_path: str, config: Config, num_episodes: int = 5):
    """
    Watch a trained agent play (with rendering)
    
    Args:
        model_path: Path to trained model
        config: Configuration object
        num_episodes: Number of episodes to watch
    """
    print(f"Watching agent play: {model_path}")
    
    # Set random seed
    set_seed(config.SEED)
    
    # Create environment with rendering
    env = create_environment(config, record_stats=False)
    
    # Create agent and load model
    agent = PPOAgent(env, config)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    agent.load_model(model_path)
    
    # Watch agent play
    print(f"Watching {num_episodes} episodes... (Close window to stop)")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while True:
            env.render()
            
            # Get action from agent
            action = agent.get_action(obs)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                print(f"Episode finished: Reward = {episode_reward:.2f}, Steps = {step_count}")
                time.sleep(1)  # Pause between episodes
                break
    
    env.close()


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="PPO Flappy Bird Training and Evaluation")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "watch"], 
                       default="train", help="Mode: train, eval, or watch")
    parser.add_argument("--model_path", type=str, help="Path to model file (for eval/watch modes)")
    parser.add_argument("--episodes", type=int, default=10, 
                       help="Number of episodes (for eval/watch modes)")
    parser.add_argument("--render", action="store_true", 
                       help="Render during evaluation")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file (not implemented)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    try:
        if args.mode == "train":
            train_agent(config, resume_from=args.resume)
        
        elif args.mode == "eval":
            if not args.model_path:
                print("Error: --model_path required for evaluation mode")
                sys.exit(1)
            evaluate_model(args.model_path, config, args.episodes, args.render)
        
        elif args.mode == "watch":
            if not args.model_path:
                print("Error: --model_path required for watch mode")
                sys.exit(1)
            watch_agent(args.model_path, config, args.episodes)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()