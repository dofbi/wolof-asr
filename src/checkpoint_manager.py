"""
Module de gestion des points de contrôle pour l'entraînement du modèle Wolof ASR.
"""

import logging
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import wandb
from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, experiment_dir: str):
        """Initialise le gestionnaire de points de contrôle."""
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint(self) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
        """Trouve le dernier point de contrôle et son état."""
        try:
            api = wandb.Api()
            runs = api.runs("wolof-asr")

            exp_runs = [run for run in runs if str(self.experiment_dir.name) in run.config.get("experiment_dir", "")]

            if not exp_runs:
                logger.info("Aucun point de contrôle trouvé")
                return None, None

            last_run = max(exp_runs, key=lambda r: r.summary.get("epoch", 0))
            checkpoint_epoch = last_run.summary.get("epoch", 0) - 1
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{checkpoint_epoch}.pt"

            if not checkpoint_path.exists():
                logger.warning(f"Fichier de point de contrôle non trouvé: {checkpoint_path}")
                return None, None

            state = {
                "epoch": checkpoint_epoch,
                "global_step": last_run.summary.get("global_step", 0),
                "best_val_loss": last_run.summary.get("best_val_loss", float("inf"))
            }

            return checkpoint_path, state

        except Exception as e:
            logger.error(f"Erreur lors de la recherche du dernier point de contrôle: {str(e)}")
            return None, None

    def prompt_resume_training(self) -> bool:
        """Demande à l'utilisateur s'il souhaite reprendre l'entraînement."""
        checkpoint_path, _ = self.find_latest_checkpoint()
        if checkpoint_path is None:
            print("Aucun checkpoint trouvé, l'entraînement commencera depuis le début.")
            return False

        response = input(f"Un point de contrôle a été trouvé à {checkpoint_path}. Voulez-vous reprendre l'entraînement à partir de ce checkpoint ? (y/n): ").strip().lower()
        return response == 'y'

    def save_checkpoint(self, model: Any, optimizer: Any, epoch: int, global_step: int, best_val_loss: float) -> None:
        """Sauvegarde un point de contrôle avec validation robuste."""
        try:
            # Validation approfondie du modèle
            if not hasattr(model, 'state_dict') or not callable(getattr(model, 'state_dict', None)):
                raise AttributeError(f"Modèle invalide : {type(model)} ne supporte pas state_dict()")

            if not hasattr(optimizer, 'state_dict') or not callable(getattr(optimizer, 'state_dict', None)):
                raise AttributeError("Optimiseur invalide")

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss
            }

            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Point de contrôle sauvegardé: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Échec sauvegarde point de contrôle : {str(e)}")

    def load_checkpoint(self, model: Any, optimizer: Any) -> Tuple[int, int, float]:
        """Charge le dernier point de contrôle."""
        checkpoint_path, _ = self.find_latest_checkpoint()

        if checkpoint_path is None:
            return 0, 0, float("inf")

        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            return (
                checkpoint["epoch"],
                checkpoint["global_step"],
                checkpoint["best_val_loss"]
            )

        except Exception as e:
            logger.error(f"Erreur chargement point de contrôle: {str(e)}")
            return 0, 0, float("inf")
