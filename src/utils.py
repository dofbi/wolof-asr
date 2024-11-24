import torch
import random
import numpy as np
import logging
import os

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging():
    """Setup logging configuration using config."""
    from src.config import LOGGING_CONFIG
    from logging.config import dictConfig
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(LOGGING_CONFIG['handlers']['file']['filename']), exist_ok=True)
    
    # Configure logging
    dictConfig(LOGGING_CONFIG)

def get_device():
    """Get appropriate device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_experiment_dir(base_dir="experiments"):
    """Create directory for experiment results."""
    os.makedirs(base_dir, exist_ok=True)
    experiment_id = len(os.listdir(base_dir))
    experiment_dir = os.path.join(base_dir, f"experiment_{experiment_id}")
    os.makedirs(experiment_dir)
    return experiment_dir
