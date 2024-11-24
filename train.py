import os
import argparse
import logging
import psutil
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_scheduler

from src.data_loader import create_data_loader
from src.model import WolofWhisperModel
from src.trainer import Trainer
from src.utils import seed_everything

# Configuration des logs
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main(args):
    # Fixer les seeds pour la reproductibilité
    seed_everything(args.seed)
    logger.info(f"Seed fixé à {args.seed}")

    # Charger les données
    logger.info("Chargement des données...")
    train_dataloader, val_dataloader = create_data_loader(
        batch_size=args.batch_size, shuffle=True
    )
    logger.info(f"Nombre de batches d'entraînement : {len(train_dataloader)}")
    logger.info(f"Nombre de batches de validation : {len(val_dataloader)}")

    # Charger le modèle
    logger.info(f"Chargement du modèle pré-entraîné : {args.model_name}")
    model = WolofWhisperModel(model_name=args.model_name)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Définir l'optimiseur et le scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=args.epochs * len(train_dataloader)
    )

    # Initialiser le gestionnaire d'entraînement
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir
    )

    # Lancer l'entraînement
    logger.info("Démarrage de l'entraînement...")
    try:
        trainer.train(epochs=args.epochs, max_grad_norm=args.max_grad_norm)
        logger.info("Entraînement terminé avec succès.")
    except KeyboardInterrupt:
        logger.warning("Entraînement interrompu par l'utilisateur.")
        trainer.save_checkpoint()
    except Exception as e:
        logger.error(f"Une erreur inattendue est survenue : {str(e)}")
        trainer.save_checkpoint()

if __name__ == "__main__":
    # Définition des arguments du script
    parser = argparse.ArgumentParser(description="Script d'entraînement du modèle WolofWhisper.")
    parser.add_argument("--model_name", default="openai/whisper-small", type=str,
                        help="Nom du modèle pré-entraîné.")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Taille du batch pour l'entraînement.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Nombre d'époques d'entraînement.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Taux d'apprentissage pour l'optimiseur.")
    parser.add_argument("--seed", default=42, type=int,
                        help="Seed pour la reproductibilité.")
    parser.add_argument("--log_interval", default=10, type=int,
                        help="Intervalle des logs pendant l'entraînement.")
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str,
                        help="Répertoire où sauvegarder les checkpoints.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Norme maximale des gradients pour le clipping.")

    # Parse les arguments et lance le script principal
    args = parser.parse_args()
    main(args)
