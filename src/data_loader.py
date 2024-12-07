from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
import logging
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, Union, Optional
from torch.utils import data

logger = logging.getLogger(__name__)

def collate_fn(batch):
    """
    Fonction de regroupement pour gérer les séquences de longueur variable.
    """
    max_length = max(item['audio'].shape[0] for item in batch)
    audio_tensor = torch.zeros((len(batch), max_length))
    for i, item in enumerate(batch):
        audio_tensor[i, :item['audio'].shape[0]] = item['audio']

    return {
        'audio': audio_tensor,
        'text': [item['text'] for item in batch],
        'duration': torch.tensor([item['duration'] for item in batch])
    }


class WolofDataset(Dataset):
    def __init__(self, split: str = "train", max_samples: Optional[int] = None, use_combined: bool = False):
        """
        Charge le dataset Wolof TTS depuis HuggingFace et divise `train` en `train` et `test` si nécessaire.
        
        Args:
            split (str): Jeu de données à charger ('train' ou 'test').
            max_samples (int, optional): Nombre maximum d'exemples à charger. Si None, charge tout.
            use_combined (bool): Si True, combine `anta_women_tts` et `wolof_tts`.
        """
        try:
            datasets = []
            
            # Charger le premier dataset
            logger.info(f"Chargement du dataset 'anta_women_tts'...")
            dataset_anta = load_dataset(
                "galsenai/anta_women_tts",
                split="train",
                cache_dir="./cache"
            )
            datasets.append(dataset_anta)
            
            # Charger le second dataset si `use_combined` est activé
            if use_combined:
                logger.info(f"Chargement du dataset 'wolof_tts'...")
                dataset_wolof = load_dataset(
                    "galsenai/wolof_tts",
                    split="train",
                    cache_dir="./cache"
                )
                datasets.append(dataset_wolof)
            
            # Fusionner les datasets si nécessaire
            if len(datasets) > 1:
                logger.info("Fusion des datasets...")
                full_dataset = concatenate_datasets(datasets)
            else:
                full_dataset = datasets[0]
            
            # Diviser en `train` et `test` si nécessaire
            if split == "test":
                logger.info("Création d'un split `test` à partir du split `train`...")
                split_data = full_dataset.train_test_split(test_size=0.2, seed=42)
                self.dataset = split_data["test"]
            elif split == "train":
                logger.info("Création d'un split `train` à partir du split `train`...")
                split_data = full_dataset.train_test_split(test_size=0.2, seed=42)
                self.dataset = split_data["train"]
            else:
                raise ValueError(f"Split inconnu : {split}")
            
            # Limiter les échantillons si max_samples est défini
            total_samples = len(self.dataset)
            if max_samples is not None and max_samples > 0:
                self.dataset = self.dataset.select(range(min(max_samples, total_samples)))
                logger.info(f"Dataset limité de {total_samples} à {len(self.dataset)} exemples (max_samples={max_samples})")
            else:
                logger.info(f"Chargement réussi de {total_samples} exemples pour le split {split}")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du dataset : {str(e)}")
            raise
        
        self.sampling_rate = 16000

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Récupère un échantillon audio et texte avec une gestion robuste des erreurs.
        """
        try:
            item = self.dataset[idx]
            audio_data = item['audio']
            sampling_rate = int(audio_data['sampling_rate'])

            # Charger et rééchantillonner l'audio si nécessaire
            audio = torch.from_numpy(audio_data['array']).float()
            if sampling_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate,
                    new_freq=self.sampling_rate
                )
                audio = resampler(audio)

            # Normalisation sécurisée
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            return {
                'audio': audio,
                'text': item['text'],
                'duration': len(audio) / self.sampling_rate
            }
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'élément à l'indice {idx} : {str(e)}")
            return {
                'audio': torch.zeros(self.sampling_rate),
                'text': "",
                'duration': 1.0
            }


def create_data_loader(batch_size: int = 8, max_samples: Optional[int] = None, use_combined: bool = False):
    """
    Crée des DataLoaders pour l'entraînement et la validation en combinant plusieurs datasets si nécessaire.
    
    Args:
        batch_size (int): Taille de batch pour les DataLoaders.
        max_samples (int, optional): Nombre maximum d'échantillons à charger. Si None, charge tout.
        use_combined (bool): Si True, combine `anta_women_tts` et `wolof_tts`.
        
    Returns:
        tuple: (train_loader, val_loader) - DataLoaders pour l'entraînement et la validation.
    """
    train_dataset = WolofDataset(split="train", max_samples=max_samples, use_combined=use_combined)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataset = WolofDataset(split="test", max_samples=max_samples, use_combined=use_combined)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader