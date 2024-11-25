from datasets import load_dataset, Dataset as HFDataset
import logging

logger = logging.getLogger(__name__)
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, Union, cast, Optional
from torch.utils import data

class WolofDataset(Dataset):
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        """Load Wolof TTS dataset from HuggingFace.
        
        Args:
            split (str): Dataset split to load ('train' or 'test')
            max_samples (int, optional): Maximum number of samples to load. If None, load all samples.
        """
        try:
            logger.info(f"Downloading and loading {split} dataset...")
            dataset = load_dataset(
                "galsenai/wolof_tts",
                split=split,
                cache_dir="./cache"
            )
            if isinstance(dataset, dict):
                self.dataset = cast(HFDataset, dataset[split])
            else:
                self.dataset = cast(HFDataset, dataset)
                
            total_samples = len(self.dataset)
            if max_samples is not None and max_samples > 0:
                self.dataset = self.dataset.select(range(min(max_samples, total_samples)))
                logger.info(f"Limited {split} dataset from {total_samples} to {len(self.dataset)} samples (max_samples={max_samples})")
            else:
                logger.info(f"Successfully loaded all {total_samples} examples from {split} split")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
        self.sampling_rate = 16000

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get audio and text pair with robust error handling."""
        try:
            item = self.dataset[idx]
            audio_data = item['audio']
            sampling_rate = int(audio_data['sampling_rate'])

            # Load and resample audio if necessary
            audio = torch.from_numpy(audio_data['array']).float()
            
            # Add validation for audio data
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                logger.warning(f"Invalid audio values detected at index {idx}, replacing with zeros")
                audio = torch.zeros_like(audio)
            
            if sampling_rate != self.sampling_rate:
                try:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sampling_rate,
                        new_freq=self.sampling_rate
                    )
                    audio = resampler(audio)
                except Exception as e:
                    logger.error(f"Resampling failed for index {idx}: {str(e)}")
                    audio = torch.zeros(self.sampling_rate)  # Return 1 second of silence

            # Safe normalization
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            else:
                logger.warning(f"Zero or invalid audio at index {idx}")
                audio = torch.zeros_like(audio)

            return {
                'audio': audio,
                'text': item['text'],
                'duration': len(audio) / self.sampling_rate
            }
        except Exception as e:
            logger.error(f"Error loading item at index {idx}: {str(e)}")
            # Return a safe fallback value
            return {
                'audio': torch.zeros(self.sampling_rate),
                'text': "",
                'duration': 1.0
            }

def create_data_loader(batch_size: int = 8, max_samples: Optional[int] = None):
    """Create train and validation data loaders.
    
    Args:
        batch_size (int): Batch size for the data loaders
        max_samples (int, optional): Maximum number of samples to load. If None, load all samples.
        
    Returns:
        tuple: (train_loader, val_loader) - PyTorch DataLoader objects for training and validation
    """
    # Créer le dataset d'entraînement
    train_dataset = WolofDataset(split="train", max_samples=max_samples)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Créer le dataset de validation
    val_dataset = WolofDataset(split="test", max_samples=max_samples)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """Custom collate function for batching."""
    max_length = max(item['audio'].shape[0] for item in batch)

    # Pad audio sequences
    audio_tensor = torch.zeros((len(batch), max_length))
    for i, item in enumerate(batch):
        audio_tensor[i, :item['audio'].shape[0]] = item['audio']

    return {
        'audio': audio_tensor,
        'text': [item['text'] for item in batch],
        'duration': torch.tensor([item['duration'] for item in batch])
    }