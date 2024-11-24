"""
Main entry point for training and evaluating the Wolof ASR model.
This module handles the training pipeline and evaluation process.
"""

import os
import argparse
import logging
from typing import Any, Dict

# Local imports
from src.config import (
    DEFAULT_MODEL_CONFIG,
    PROJECT_ROOT,
    MODELS_DIR,
    WANDB_PROJECT,
)
from src.utils import setup_logging, set_seed, get_device, create_experiment_dir
from src.data_loader import create_data_loader
from src.model import WolofWhisperModel
from src.audio_processor import AudioPreprocessor
from src.trainer import Trainer
from src.evaluator import Evaluator

os.environ["WANDB_CONSOLE"] = "wrap"

def train_model(args: argparse.Namespace) -> None:
    """
    Train and evaluate the Wolof ASR model.

    Args:
        args: Namespace containing training arguments and configuration.

    Raises:
        RuntimeError: If training fails due to resource or configuration issues.
        ValueError: If input arguments are invalid.
    """
    try:
        if not os.getenv("WANDB_API_KEY"):
            raise ValueError("WANDB_API_KEY environment variable is not set")

        if args.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if args.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        # Setup configuration
        set_seed(args.seed)
        setup_logging()
        device = get_device()
        experiment_dir = create_experiment_dir(base_dir=str(MODELS_DIR))

        logging.info(f"Using device: {device}")
        logging.info(f"Experiment directory: {experiment_dir}")

        # Create dataloaders
        train_loader = create_data_loader("train", batch_size=args.batch_size)
        val_loader = create_data_loader("test", batch_size=args.batch_size)

        # Initialize model and processor
        model = WolofWhisperModel(pretrained_model=args.model_name)
        processor = AudioPreprocessor()

        # Initialize trainer with experiment directory
        trainer = Trainer(model, processor, str(device), experiment_dir)

        # Train model
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )

        # Evaluate
        evaluator = Evaluator(model, processor, str(device))
        metrics = evaluator.compute_metrics(val_loader)

        logging.info(f"Validation WER: {metrics['wer']}")
        logging.info("Sample predictions:")
        for ref, pred in metrics['samples']:
            logging.info(f"Reference: {ref}")
            logging.info(f"Prediction: {pred}")
            logging.info("-" * 50)

    except ValueError as ve:
        logging.error(f"Invalid configuration: {ve}")
        raise
    except RuntimeError as re:
        logging.error(f"Training failed due to runtime issues: {re}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise

def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train Wolof ASR model")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_CONFIG['model_name'],
                        help="Name of the pretrained model to use")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_MODEL_CONFIG['batch_size'],
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=DEFAULT_MODEL_CONFIG['epochs'],
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_MODEL_CONFIG['learning_rate'],
                        help="Learning rate for training")
    parser.add_argument("--seed", type=int, default=DEFAULT_MODEL_CONFIG['seed'],
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    main()
