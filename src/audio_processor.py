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

            # Noise reduction
            noise_gate = torchaudio.transforms.Vad(sample_rate=self.sampling_rate)
            audio_array = noise_gate(audio_array)

            # Normalize audio
            if audio_array.numel() == 0 or torch.max(torch.abs(audio_array)) == 0:
                logging.warning("Audio array is empty or its max value is zero. Skipping normalization.")
            else:
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

    def batch_decode(self, token_ids, skip_special_tokens=True):
        """Decode a batch of token IDs into text with gestion robuste des erreurs et validation.

        Args:
            token_ids: Liste, numpy.ndarray ou torch.Tensor contenant les IDs des tokens
            skip_special_tokens: Booléen indiquant si les tokens spéciaux doivent être ignorés

        Returns:
            List[str]: Liste des textes décodés

        Raises:
            ValueError: Si token_ids est invalide ou vide
            RuntimeError: Si une erreur survient pendant le décodage
        """
        if token_ids is None:
            error_msg = "token_ids ne peut pas être None"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.debug(f"Début du décodage batch - Type des token_ids: {type(token_ids)}")
        try:
            if token_ids is None:
                logging.warning("token_ids is None, returning empty list")
                return [""]

            # Configuration et validation du tokenizer
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
                logging.info(f"Using alternative token as pad_token_id: {pad_token_id}")

            # Conversion et validation des tenseurs
            if torch.is_tensor(token_ids):
                if token_ids.dim() == 0:
                    logging.warning("Received 0-dim tensor, expanding to 1-dim")
                    token_ids = token_ids.unsqueeze(0)

                token_ids = token_ids.detach().cpu()
                if token_ids.requires_grad:
                    token_ids = token_ids.detach()

                # Vérification des valeurs non valides dans le tenseur
                if torch.isnan(token_ids).any() or torch.isinf(token_ids).any():
                    logging.warning("Found invalid values in tensor, replacing with pad token")
                    token_ids = torch.where(torch.isnan(token_ids) | torch.isinf(token_ids),
                                         torch.tensor(pad_token_id, device=token_ids.device),
                                         token_ids)
                token_ids = token_ids.numpy()

            # Traitement des tableaux numpy
            if isinstance(token_ids, np.ndarray):
                # Vérification et correction des valeurs non valides
                invalid_mask = np.logical_or.reduce((
                    np.isnan(token_ids),
                    np.isinf(token_ids),
                    token_ids < 0,
                    token_ids >= self.tokenizer.vocab_size
                ))

                if np.any(invalid_mask):
                    logging.warning(f"Correcting {np.sum(invalid_mask)} invalid token values")
                    # Ensure pad_token_id is a numpy array of the same dtype as token_ids
                    pad_value = np.array(pad_token_id, dtype=token_ids.dtype)
                    token_ids = np.where(invalid_mask, pad_value, token_ids)
                    logging.debug(f"Token IDs after correction: shape={token_ids.shape}, dtype={token_ids.dtype}")

                # Ensure valid range and type
                token_ids = np.clip(np.round(token_ids), 0, self.tokenizer.vocab_size - 1)
                token_ids = token_ids.astype(np.int64)
                logging.debug(f"Token IDs after processing: shape={token_ids.shape}, dtype={token_ids.dtype}")
                token_ids = token_ids.tolist()

            # Validation et normalisation de la structure
            if not isinstance(token_ids, list):
                try:
                    token_ids = list(token_ids)
                except Exception as e:
                    logging.error(f"Failed to convert token_ids to list: {str(e)}")
                    return [""]

            # Assurer une structure de liste de listes
            if token_ids and not isinstance(token_ids[0], (list, np.ndarray)):
                token_ids = [token_ids]

            # Traitement séquentiel avec validation
            processed_ids = []
            for seq_idx, seq in enumerate(token_ids):
                if not seq:
                    processed_ids.append([pad_token_id])
                    continue

                processed_seq = []
                for token_idx, token in enumerate(seq):
                    try:
                        token_val = int(round(float(token)))
                        if not 0 <= token_val < self.tokenizer.vocab_size:
                            logging.warning(
                                f"Token value {token_val} at position {token_idx} in sequence {seq_idx} "
                                f"is out of range [0, {self.tokenizer.vocab_size}), using pad token"
                            )
                            token_val = pad_token_id
                        processed_seq.append(token_val)
                    except (ValueError, TypeError, OverflowError) as e:
                        logging.warning(
                            f"Invalid token at position {token_idx} in sequence {seq_idx}: {token}, "
                            f"using pad token. Error: {str(e)}"
                        )
                        processed_seq.append(pad_token_id)

                if not processed_seq:
                    processed_seq = [pad_token_id]
                processed_ids.append(processed_seq)

            # Décodage final avec gestion d'erreurs
            try:
                decoded_texts = self.tokenizer.batch_decode(
                    processed_ids,
                    skip_special_tokens=skip_special_tokens
                )
                # Validation finale des résultats
                if not all(isinstance(text, str) for text in decoded_texts):
                    logging.error("Tokenizer returned non-string values")
                    return [""] * len(processed_ids)

                return decoded_texts

            except Exception as e:
                logging.error(f"Tokenizer batch_decode failed: {str(e)}")
                return [""] * len(processed_ids)

        except Exception as e:
            logging.error(f"Critical error in batch_decode: {str(e)}")
            return [""] * (len(token_ids) if isinstance(token_ids, list) else 1)

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
