import argparse
from src.model import WolofWhisperModel
from src.audio_processor import AudioPreprocessor
from src.inference import WolofASRInference
from src.utils import get_device
import logging

def main(args):
    # Setup
    device = get_device()
    logging.basicConfig(level=logging.INFO)

    # Load model and processor
    model = WolofWhisperModel(pretrained_model=args.model_path)
    processor = AudioPreprocessor()

    # Initialize inference pipeline
    inference = WolofASRInference(model, processor, str(device))

    # Transcribe audio
    if args.batch:
        with open(args.input, 'r') as f:
            audio_paths = f.readlines()
        audio_paths = [path.strip() for path in audio_paths]

        transcriptions = inference.transcribe_batch(audio_paths)

        # Save results
        with open(args.output, 'w') as f:
            for path, transcription in zip(audio_paths, transcriptions):
                f.write(f"File: {path}\n")
                f.write(f"Transcription: {transcription}\n")
                f.write("-" * 50 + "\n")
    else:
        transcription = inference.transcribe_file(args.input)
        logging.info(f"Transcription: {transcription}")

        if args.output:
            with open(args.output, 'w') as f:
                f.write(transcription)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--batch", action="store_true")

    args = parser.parse_args()
    main(args)