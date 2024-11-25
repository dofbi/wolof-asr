import argparse
from train import main as train_main

def main():
    """Main entry point that reuses the training logic from train.py"""
    parser = argparse.ArgumentParser(description="Train Wolof ASR model")
    parser.add_argument("--model_name", default="openai/whisper-small", type=str,
                        help="Name of the pretrained model to use")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Interval for logging during training")
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str,
                        help="Directory to save checkpoints")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use for training")

    args = parser.parse_args()
    train_main(args)

if __name__ == "__main__":
    main()
