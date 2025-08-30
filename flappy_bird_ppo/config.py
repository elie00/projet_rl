"""
Configuration file for Flappy Bird PPO training
Contains all hyperparameters and training settings
"""
import torch
import os

class Config:
    # Environment settings
    ENV_NAME = "FlappyBird-v0"
    USE_LIDAR = False
    
    # PPO Hyperparameters
    LEARNING_RATE = 3e-4
    GAMMA = 0.99                    # Discount factor
    CLIP_EPSILON = 0.2              # PPO clipping parameter
    PPO_EPOCHS = 4                  # Number of optimization epochs per update
    ENTROPY_COEFF = 0.01            # Entropy regularization coefficient
    VALUE_COEFF = 0.5               # Value loss coefficient
    MAX_GRAD_NORM = 0.5             # Gradient clipping norm
    
    # Training settings
    MAX_STEPS_PER_ROLLOUT = 512     # Steps per rollout collection
    TOTAL_EPISODES = 10000          # Total training episodes
    EVAL_FREQUENCY = 100            # Evaluate every N episodes
    SAVE_FREQUENCY = 500            # Save model every N episodes
    LOG_FREQUENCY = 200             # Log metrics every N steps
    
    # Network architecture
    HIDDEN_SIZES = [512, 512, 256]  # Hidden layer sizes for both networks
    
    # Evaluation settings
    EVAL_EPISODES = 10              # Number of episodes for evaluation
    RENDER_EVAL = False             # Render during evaluation
    RECORD_VIDEO = True             # Record evaluation videos
    
    # Logging and saving
    USE_WANDB = True                # Use Weights & Biases logging
    WANDB_PROJECT = "flappy-bird-ppo"
    WANDB_ENTITY = None             # Set your wandb username if needed
    LOG_DIR = "logs"                # Directory for tensorboard logs
    MODEL_DIR = "models"            # Directory for saving models
    VIDEO_DIR = "videos"            # Directory for saving videos
    
    # Device settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reproducibility
    SEED = 42
    
    # Success criteria
    SUCCESS_THRESHOLD = 10          # Consider success if agent gets > 10 pipes
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for logging and saving"""
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.VIDEO_DIR, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, episode):
        """Get model save path for given episode"""
        return os.path.join(cls.MODEL_DIR, f"ppo_flappy_bird_{episode}.pt")
    
    @classmethod
    def get_video_path(cls, episode):
        """Get video save path for given episode"""
        return os.path.join(cls.VIDEO_DIR, f"eval_episode_{episode}.mp4")
    
    def __str__(self):
        """String representation of config for logging"""
        config_str = "PPO Configuration:\n"
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if not attr_name.startswith('get_') and not attr_name.startswith('create_'):
                    config_str += f"  {attr_name}: {attr_value}\n"
        return config_str