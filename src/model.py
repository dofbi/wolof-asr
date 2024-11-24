import torch
import logging
from transformers import WhisperForConditionalGeneration, WhisperConfig
from transformers.models.whisper.modeling_whisper import EncoderDecoderCache
from transformers.generation.utils import GenerateOutput

class WolofWhisperModel:
    def __init__(self, pretrained_model="openai/whisper-small"):
        """Initialize Whisper model for Wolof."""
        self.model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)

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

    def parameters(self):
        """Return model parameters for optimization."""
        return list(self.model.parameters())

    def save(self, path):
        """Save the model to disk."""
        self.model.save_pretrained(path)

    def load(self, path):
        """Load the model from disk."""
        self.model = WhisperForConditionalGeneration.from_pretrained(path)

    def train(self):
        """Set model to training mode."""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def to(self, device):
        """Move model to specified device."""
        self.model = self.model.to(device)

    def __call__(self, input_features, labels=None):
        """Forward pass through the model."""
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
