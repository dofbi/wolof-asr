"""
Module de gestion des points de contrôle pour l'entraînement du modèle Wolof ASR.
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import wandb
from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, experiment_dir: str):
        """
        Initialise le gestionnaire de points de contrôle.
        
        Args:
            experiment_dir: Chemin du répertoire d'expérience
        """
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def find_latest_checkpoint(self) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
        """
        Trouve le dernier point de contrôle et son état.
        
        Returns:
            Tuple contenant le chemin du checkpoint et son état
        """
        try:
            api = wandb.Api()
            runs = api.runs("wolof-asr")
            
            # Filtrer les runs pour ce répertoire d'expérience
            exp_runs = [run for run in runs if str(self.experiment_dir.name) in run.config.get("experiment_dir", "")]
            
            if not exp_runs:
                logger.info("Aucun point de contrôle trouvé")
                return None, None
                
            # Trouver le dernier run avec le plus grand nombre d'époques complétées
            last_run = max(exp_runs, key=lambda r: r.summary.get("epoch", 0))
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{last_run.summary.get('epoch', 0)}.pt"
            
            if not checkpoint_path.exists():
                logger.warning(f"Fichier de point de contrôle non trouvé: {checkpoint_path}")
                return None, None
                
            state = {
                "epoch": last_run.summary.get("epoch", 0),
                "global_step": last_run.summary.get("global_step", 0),
                "best_val_loss": last_run.summary.get("best_val_loss", float("inf"))
            }
            
            return checkpoint_path, state
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche du dernier point de contrôle: {str(e)}")
            return None, None
            
    def prompt_resume_training(self) -> bool:
        """
        Demande à l'utilisateur s'il souhaite reprendre l'entraînement.
        
        Returns:
            bool: True si l'utilisateur souhaite reprendre, False sinon
        """
        checkpoint_path, state = self.find_latest_checkpoint()
        
        if checkpoint_path is None:
            logger.info("Démarrage d'un nouvel entraînement")
            return False
            
        if checkpoint_path is None or state is None:
            logger.info("Démarrage d'un nouvel entraînement")
            return False

        logger.info("\n=== Point de contrôle trouvé ===")
        logger.info(f"Époque: {state['epoch']}")
        logger.info(f"Étape globale: {state['global_step']}")
        logger.info(f"Meilleure perte de validation: {state['best_val_loss']:.4f}")
        
        while True:
            response = input("\nVoulez-vous reprendre l'entraînement depuis ce point ? (oui/non): ").lower()
            if response in ["oui", "o", "yes", "y"]:
                return True
            elif response in ["non", "n", "no"]:
                return False
            print("Veuillez répondre par 'oui' ou 'non'")
            
    def save_checkpoint(self, model: Any, optimizer: Any, epoch: int, global_step: int,
                       best_val_loss: float) -> None:
        """
        Sauvegarde un point de contrôle.
        """
        try:
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
            logger.error(f"Erreur lors de la sauvegarde du point de contrôle: {str(e)}")
            
    def load_checkpoint(self, model: Any, optimizer: Any) -> Tuple[int, int, float]:
        """
        Charge le dernier point de contrôle.
        
        Returns:
            Tuple[int, int, float]: époque, étape globale, meilleure perte de validation
        """
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
            logger.error(f"Erreur lors du chargement du point de contrôle: {str(e)}")
            return 0, 0, float("inf")
