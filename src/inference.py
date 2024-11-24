import torch
import soundfile as sf
from src.audio_processor import AudioPreprocessor

class WolofASRInference:
    def __init__(self, model, processor, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize inference pipeline."""
        self.model = model
        self.processor = processor
        self.device = device
        self.model.to(device)
        self.model.eval()

    def transcribe_file(self, audio_path):
        """Transcribe audio file."""
        # Load audio
        audio, sampling_rate = sf.read(audio_path)
        return self.transcribe_audio(audio, sampling_rate)

    def transcribe_audio(self, audio_array, sampling_rate):
        """Transcribe audio array."""
        with torch.no_grad():
            # Preprocess audio
            input_features = self.processor.preprocess(
                torch.tensor(audio_array),
                sampling_rate=sampling_rate
            ).to(self.device)

            # Generate transcription
            generated_ids = self.model.generate(input_features)
            transcription = self.processor.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            return transcription

    def transcribe_batch(self, audio_paths):
        """Transcribe a batch of audio files."""
        transcriptions = []

        for path in audio_paths:
            transcription = self.transcribe_file(path)
            transcriptions.append(transcription)

        return transcriptions
