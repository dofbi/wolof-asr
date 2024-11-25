import torch
import torch.nn as nn
import logging
from transformers import WhisperForConditionalGeneration, WhisperConfig
from transformers.models.whisper.modeling_whisper import EncoderDecoderCache
from transformers.generation.utils import GenerateOutput

class WolofWhisperModel(nn.Module):
    def __init__(self, model_name="openai/whisper-small"):
        """Initialize Whisper model for Wolof.
        
        Args:
            model_name (str): Nom du modèle pré-entraîné à utiliser (ex: 'openai/whisper-small')
        """
        super().__init__()
        try:
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            logging.info(f"Modèle {model_name} chargé avec succès")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle {model_name}: {str(e)}")
            raise

        # Modify the model for Wolof
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        # Set language and task
        self.model.config.forced_decoder_ids = [
            [1, 50359],  # Set task to transcribe
            [2, 50259],  # Set language to None to allow model to detect
        ]

        # Adjust generation settings for better Wolof handling
        self.model.config.suppress_tokens = []
        self.model.config.max_length = 225
        self.model.config.num_beams = 5
        self.model.config.temperature = 0.0
        self.model.config.length_penalty = 1.0

        # Define the decoder start token id
        self.decoder_start_token_id = self.model.config.decoder_start_token_id

    def parameters(self, recurse: bool = True):
        """Return model parameters for optimization."""
        return self.model.parameters(recurse=recurse)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return the model's state dictionary following PyTorch's nn.Module interface.

        Args:
            destination (dict, optional): Starting state_dict to update with parameters
            prefix (str): Parameter key prefix
            keep_vars (bool): Whether to keep parameter variables instead of tensor copies

        Returns:
            dict: The model's state dictionary containing parameters and buffers
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model is not properly initialized")

        # Get state dict from underlying model
        model_state = self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # If no destination provided, use model_state as destination
        if destination is None:
            destination = model_state
        else:
            # Update destination with model state
            for name, param in model_state.items():
                destination[prefix + name] = param

        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Load a state dictionary with validation and error handling."""
        try:
            if not state_dict:
                raise ValueError("Empty state dictionary provided")
            
            # Validate keys match current model
            model_keys = set(self.model.state_dict().keys())
            input_keys = set(state_dict.keys())
            
            missing_keys = model_keys - input_keys
            unexpected_keys = input_keys - model_keys
            
            if missing_keys:
                logging.warning(f"Missing keys in state dict: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys in state dict: {unexpected_keys}")
            
            # Load state dict with specified strict parameter
            result = self.model.load_state_dict(state_dict, strict=strict)
            
            if result.missing_keys or result.unexpected_keys:
                logging.warning(f"State dict loaded with mismatches - Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in load_state_dict(): {str(e)}")
            raise RuntimeError(f"Failed to load state dictionary: {str(e)}")

    def save(self, path):
        """Save the model to disk."""
        self.model.save_pretrained(path)

    def load(self, path):
        """Load the model from disk."""
        self.model = WhisperForConditionalGeneration.from_pretrained(path)

    def train(self, mode: bool = True):
        """Set model to training mode."""
        self.model.train(mode)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def to(self, device):
        """Move model to specified device."""
        self.model = self.model.to(device)

    def prepare_decoder_input_ids(self, batch_size, device):
        """Prepare decoder input IDs for initialization."""
        # Create decoder_input_ids starting with decoder_start_token_id
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * self.decoder_start_token_id
        return decoder_input_ids

    def __call__(self, input_features, labels=None):
        """Forward pass through the model."""
        batch_size = input_features.size(0)
        device = input_features.device

        # Prepare decoder input IDs if not in training mode
        decoder_input_ids = None if labels is not None else self.prepare_decoder_input_ids(batch_size, device)

        if labels is not None:
            # Training/validation forward pass
            outputs = self.model(
                input_features=input_features,
                labels=labels,
                return_dict=True
            )
        else:
            # Inference forward pass
            outputs = self.model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,  # Add decoder_input_ids here
                return_dict=True
            )

        # Handle past_key_values
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            if not isinstance(outputs.past_key_values, EncoderDecoderCache):
                outputs.past_key_values = EncoderDecoderCache.from_legacy_cache(outputs.past_key_values)

        return outputs

    def generate(self, input_features):
        """Generate transcriptions."""
        try:
            # Configure generation parameters
            outputs = self.model.generate(
                input_features=input_features,
                max_length=225,
                num_beams=5,
                return_dict_in_generate=False,  # Return only sequences
                output_scores=False,  # Disable scoring for faster generation
                output_attentions=False,  # Disable attention outputs
                output_hidden_states=False  # Disable hidden states
            )
            return outputs

        except Exception as e:
            logging.error(f"Error during generation: {str(e)}")
            return None