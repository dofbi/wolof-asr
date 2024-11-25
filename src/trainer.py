import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import psutil
import time
import wandb
from tqdm import tqdm
from src.checkpoint_manager import CheckpointManager
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, processor, device="cuda" if torch.cuda.is_available() else "cpu", experiment_dir=None):
        """Initialize trainer with model and processor."""
        self._setup_logging()

        self.model = model
        self.processor = processor
        self.device = device
        self.model.to(device)
        self.experiment_dir = experiment_dir

        if experiment_dir:
            self.checkpoint_manager = CheckpointManager(experiment_dir)

    def _setup_logging(self):
        """Configure logging for training."""
        logger.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)8s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler('training.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False
        logger.info("=== Logging setup completed ===")

    def evaluate(self, val_dataloader):
        """Evaluate the model on the validation dataset."""
        logger.info("Starting evaluation...")
        self.model.eval()  # Switch to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                # Prepare data
                processed_batch = self.processor.prepare_dataset(batch)
                input_features = processed_batch['input_features'].to(self.device)
                labels = processed_batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_features=input_features, labels=labels)
                val_loss += outputs.loss.item()

        val_loss /= len(val_dataloader)
        logger.info(f"Validation loss: {val_loss:.4f}")
        self.model.train()  # Switch back to training mode
        return val_loss

    def compute_accuracy(self, val_dataloader):
        """Compute accuracy on the validation dataset."""
        logger.info("Computing accuracy...")
        self.model.eval()  # Switch to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                # Prepare data
                processed_batch = self.processor.prepare_dataset(batch)
                input_features = processed_batch['input_features'].to(self.device)
                labels = processed_batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_features=input_features)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        logger.info(f"Validation accuracy: {accuracy:.4f}")
        self.model.train()  # Switch back to training mode
        return accuracy

    def train(self, train_dataloader, val_dataloader, num_epochs=10, learning_rate=5e-5):
        """Train the model with detailed logging."""
        print("\n" + "="*70)
        print("║" + " "*24 + "TRAINING START" + " "*24 + "║")
        print("="*70 + "\n")

        print(f"\n{'-'*30} Configuration {'-'*30}")
        print(f"• Device: {self.device}")
        print(f"• Architecture: {self.model.model.config._name_or_path}")
        print(f"• Number of epochs: {num_epochs}")
        print(f"• Batch size: {train_dataloader.batch_size}")
        print(f"• Learning rate: {learning_rate}")
        print(f"• Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"• Dataset size: {len(train_dataloader.dataset):,}")
        print(f"• Total steps: {num_epochs * len(train_dataloader):,}")

        run_config = {
            "experiment_dir": str(self.experiment_dir) if self.experiment_dir else "no_experiment_dir",
            "model_name": self.model.model.config._name_or_path,
            "batch_size": train_dataloader.batch_size,
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "device": self.device,
            "optimizer": "AdamW",
            "scheduler": "LinearLR",
            "total_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "total_steps": num_epochs * len(train_dataloader),
            "dataset_size": len(train_dataloader.dataset)
        }

        wandb.init(project="wolof-asr", config=run_config, tags=["training"])

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_epochs)

        if hasattr(self, 'checkpoint_manager'):
            resume_training = self.checkpoint_manager.prompt_resume_training()
            if resume_training:
                start_epoch, global_step, best_val_loss = self.checkpoint_manager.load_checkpoint(self.model, optimizer)
            else:
                start_epoch, global_step, best_val_loss = 0, 0, float('inf')
        else:
            resume_training = False
            start_epoch, global_step, best_val_loss = 0, 0, float('inf')

        try:
            logger.info(f"Training started with {num_epochs} epochs")

            for epoch in range(start_epoch, num_epochs):
                self.model.train()
                train_loss = 0
                train_steps = 0
                epoch_start_time = time.time()
                logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
                for batch in progress_bar:
                    optimizer.zero_grad()

                    # Prepare batch
                    processed_batch = self.processor.prepare_dataset(batch)
                    input_features = processed_batch['input_features'].to(self.device)
                    labels = processed_batch['labels'].to(self.device)

                    # Forward pass
                    outputs = self.model(input_features=input_features, labels=labels)
                    loss = outputs.loss
                    loss.backward()

                    optimizer.step()

                    train_loss += loss.item()
                    train_steps += 1
                    global_step += 1
                    progress_bar.set_postfix({'loss': loss.item()})

                    metrics = {
                        "loss": loss.item(),
                        "epoch": epoch + 1,
                        "step": global_step,
                        "progress": global_step / (len(train_dataloader) * num_epochs),
                    }

                    wandb.log(metrics)

                val_loss = self.evaluate(val_dataloader)
                accuracy = self.compute_accuracy(val_dataloader)

                val_metrics = {
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "accuracy": accuracy
                }
                wandb.log(val_metrics)

                # Save checkpoint
                if hasattr(self, 'checkpoint_manager') and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.checkpoint_manager.save_checkpoint(self.model, optimizer, epoch + 1, global_step, best_val_loss)

                scheduler.step()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if hasattr(self, 'checkpoint_manager'):
                self.checkpoint_manager.save_checkpoint(self.model, optimizer, epoch, global_step, best_val_loss)

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

        finally:
            wandb.finish()
