import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def load_environment():
    """Load and validate environment variables."""
    # Load environment variables from .env file
    load_dotenv(override=True)

    # Required environment variables and their descriptions
    required_vars = {
        "WANDB_API_KEY": "Weights & Biases API key for experiment tracking",
    }

    # Check for missing variables
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")

    if missing_vars:
        error_msg = f"Missing required environment variables:\n" + "\n".join(missing_vars)
        logger.error(error_msg)
        raise EnvironmentError(error_msg)

    logger.info("Environment variables loaded successfully")

# Initialize environment
load_environment()

# Base paths with type hints for better IDE support
PROJECT_ROOT: Path = Path(__file__).parent.parent

# Initialize directory paths
DATA_DIR: Path = PROJECT_ROOT / "data"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
CACHE_DIR: Path = PROJECT_ROOT / "cache"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Create all required directories
required_dirs = [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    CACHE_DIR,
    CHECKPOINTS_DIR,
    EXPERIMENTS_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
]

for directory in required_dirs:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory initialized at: {directory}")

# Ensure the logs directory exists for the FileHandler
log_file_path = LOGS_DIR / "app.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)

# Wandb configuration
WANDB_PROJECT = "wolof-asr"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "default")

# Model configuration
DEFAULT_MODEL_CONFIG = {
    "model_name": "openai/whisper-small",
    "batch_size": 4,
    "epochs": 2,
    "learning_rate": 5e-5,
    "seed": 42,
    "max_samples": None,  # No limit by default
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "error.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file", "error_file"],
            "level": "INFO",
            "propagate": True,
        },
        "wolof_asr": {
            "handlers": ["console", "file", "error_file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}