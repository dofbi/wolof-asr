"""
Wolof ASR - A speech recognition system for the Wolof language based on OpenAI's Whisper.

This package provides tools and models for training and using an ASR system
specifically optimized for the Wolof language.
"""

from src.model import WolofWhisperModel
from src.audio_processor import AudioPreprocessor
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.data_loader import WolofDataset, create_data_loader

__all__ = [
    'WolofWhisperModel',
    'AudioPreprocessor',
    'Trainer',
    'Evaluator',
    'WolofDataset',
    'create_data_loader',
]
