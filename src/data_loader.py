from datasets import load_dataset, Dataset as HFDataset
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, Union, cast

class WolofDataset(Dataset):
    def __init__(self, split: str = "train"):
        """Load Wolof TTS dataset from HuggingFace."""
        try:
            print(f"Downloading and loading {split} dataset...")
            dataset = load_dataset(
                "galsenai/wolof_tts",
                split=split,
                cache_dir="./cache"
            )
            if isinstance(dataset, dict):
                self.dataset = cast(HFDataset, dataset[split])
            else:
                self.dataset = cast(HFDataset, dataset)
            print(f"Successfully loaded {len(self.dataset)} examples from {split} split")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise
        self.sampling_rate = 16000

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get audio and text pair."""
        item = self.dataset[idx]
        audio_data = item['audio']
        sampling_rate = int(audio_data['sampling_rate'])

        # Load and resample audio if necessary
        audio = torch.from_numpy(audio_data['array']).float()
        if sampling_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=self.sampling_rate
            )
            audio = resampler(audio)

        # Normalize audio
        audio = audio / torch.max(torch.abs(audio))

        return {
            'audio': audio,
            'text': item['text'],
            'duration': len(audio) / self.sampling_rate
        }

def create_data_loader(split="train", batch_size=8):
    """Create data loader for the specified split."""
    dataset = WolofDataset(split=split)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn
    )

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