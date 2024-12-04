from datasets import load_dataset, Dataset as HFDataset
import logging
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Union
from torch.utils import data

logger = logging.getLogger(__name__)

class WolofDataset(Dataset):
    def __init__(self, base: str, split: str = "train", max_samples: Optional[int] = None):
        """Dataset loader compatible with both wolof_tts and asr-wolof-dataset.

        Args:
            base (str): Dataset base to use ('wolof_tts' or 'asr_wolof')
            split (str): Dataset split to load ('train')
            max_samples (int, optional): Max samples to load. If None, load all.
        """
        self.sampling_rate = 16000
        self.base = base

        try:
            logger.info(f"Downloading and loading {base} ({split}) dataset...")
            if base == "wolof_tts":
                dataset = load_dataset("galsenai/wolof_tts", split="train", cache_dir="./cache")
            elif base == "asr_wolof":
                dataset = load_dataset("IndabaxSenegal/asr-wolof-dataset", split="train", cache_dir="./cache")
            else:
                raise ValueError(f"Unsupported dataset base: {base}")

            self.dataset = dataset
            total_samples = len(self.dataset)

            if max_samples is not None and max_samples > 0:
                self.dataset = self.dataset.select(range(min(max_samples, total_samples)))
                logger.info(f"Limited {split} dataset from {total_samples} to {len(self.dataset)} samples (max_samples={max_samples})")
            else:
                logger.info(f"Successfully loaded all {total_samples} examples from {split} split")

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load an audio and text sample based on the dataset structure."""
        try:
            item = self.dataset[idx]
            if self.base == "wolof_tts":
                audio_data = item['audio']
                audio = torch.from_numpy(audio_data['array']).float()
                sampling_rate = int(audio_data['sampling_rate'])
            elif self.base == "asr_wolof":
                audio_path = item['audio']['path']
                audio, sampling_rate = torchaudio.load(audio_path)
                audio = audio.squeeze(0).float()
            else:
                raise ValueError(f"Unsupported dataset base: {self.base}")

            # Resample audio if necessary
            if sampling_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=self.sampling_rate)
                audio = resampler(audio)

            # Normalize audio
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            else:
                logger.warning(f"Zero or invalid audio at index {idx}")
                audio = torch.zeros_like(audio)

            # Handle text/transcription differences
            if self.base == "wolof_tts":
                text = item['text']
                duration = len(audio) / self.sampling_rate
            elif self.base == "asr_wolof":
                text = item.get('transcription', "")
                duration = item['duration']

            return {
                'audio': audio,
                'text': text,
                'duration': duration
            }
        except Exception as e:
            logger.error(f"Error loading item at index {idx}: {str(e)}")
            return {
                'audio': torch.zeros(self.sampling_rate),
                'text': "",
                'duration': 1.0
            }

def create_data_loader(base: str, batch_size: int = 8, max_samples: Optional[int] = None):
    """Create train and validation data loaders for the specified dataset.

    Args:
        base (str): Dataset base to use ('wolof_tts' or 'asr_wolof')
        batch_size (int): Batch size for the data loaders
        max_samples (int, optional): Max samples to load. If None, load all.

    Returns:
        tuple: (train_loader, val_loader) - PyTorch DataLoader objects
    """
    # Only use 'train' split for now as 'test' split doesn't exist for this dataset
    train_dataset = WolofDataset(base=base, split="train", max_samples=max_samples)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Optionally create a validation set from a subset of 'train'
    val_dataset = train_dataset  # If you want a validation set, you can split here
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
