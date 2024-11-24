import torch
import torchaudio
import numpy as np
import logging
from typing import Dict, List, Union, Any, Optional, cast
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from transformers.models.whisper import WhisperConfig

class AudioPreprocessor:
    def __init__(self, model_name: str = "openai/whisper-small"):
        """Initialize audio preprocessor with Whisper's requirements."""
        try:
            # Initialize components directly to ensure proper typing
            self.feature_extractor: WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained(model_name)
            self.tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(model_name)
            
            # Initialize processor with the components
            self.processor = WhisperProcessor(
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer
            )
            
            # Validate initialization
            if not isinstance(self.processor, WhisperProcessor):
                raise TypeError("WhisperProcessor initialization failed")
            
            # Set audio processing parameters
            self.sampling_rate: int = 16000
            self.max_length: int = 30 * self.sampling_rate  # 30 seconds
            
            logging.info(f"Successfully initialized AudioPreprocessor with model: {model_name}")
            
        except Exception as e:
            logging.error(f"Failed to initialize AudioPreprocessor: {str(e)}")
            raise RuntimeError(f"Error initializing AudioPreprocessor: {str(e)}")

    def preprocess(self, audio_array: torch.Tensor, sampling_rate: int = 16000) -> torch.Tensor:
        """Preprocess audio array for Whisper model."""
        try:
            # Resample if necessary
            if sampling_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate,
                    new_freq=self.sampling_rate
                )
                audio_array = resampler(audio_array)

            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = torch.mean(audio_array, dim=0)

            # Normalize audio
            audio_array = audio_array / torch.max(torch.abs(audio_array))

            # Pad or truncate
            if audio_array.shape[0] < self.max_length:
                padding = self.max_length - audio_array.shape[0]
                audio_array = torch.nn.functional.pad(audio_array, (0, padding))
            else:
                audio_array = audio_array[:self.max_length]

            # Convert to numpy array before feature extraction
            audio_numpy = audio_array.numpy()
            
            # Extract features using processor
            features = self.processor(
                audio_numpy,
                sampling_rate=self.sampling_rate,
                return_tensors="pt"
            )

            return features.input_features
            
        except Exception as e:
            logging.error(f"Error in audio preprocessing: {str(e)}")
            raise RuntimeError(f"Failed to preprocess audio: {str(e)}")

    def prepare_dataset(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare a batch of data for training."""
        try:
            audio_arrays: List[torch.Tensor] = batch['audio']
            input_features: List[torch.Tensor] = []

            # Process each audio sample
            for audio in audio_arrays:
                features = self.preprocess(audio)
                input_features.append(features)

            # Combine features into a batch
            input_features_tensor = torch.cat(input_features, dim=0)

            # Use processor for text tokenization
            labels = self.processor(
                text=batch['text'],
                return_tensors="pt",
                padding=True
            ).input_ids

            return {
                'input_features': input_features_tensor,
                'labels': labels
            }
            
        except Exception as e:
            logging.error(f"Error in dataset preparation: {str(e)}")
            raise RuntimeError(f"Failed to prepare dataset batch: {str(e)}")
