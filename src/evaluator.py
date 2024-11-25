import torch
import logging
from tqdm import tqdm
from jiwer import wer
from transformers import WhisperProcessor

class Evaluator:
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = self._validate_processor(processor)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _validate_processor(self, processor):
        """Validate processor has required methods."""
        if not hasattr(processor, 'batch_decode'):
            raise AttributeError("Processor must have batch_decode method")
        return processor

    def compute_metrics(self, dataloader):
        """Compute Word Error Rate (WER) and sample predictions."""
        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                try:
                    processed_batch = self.processor.prepare_dataset(batch)
                    input_features = processed_batch['input_features'].to(self.device)

                    generated_ids = self.model.generate(input_features)
                    if generated_ids is not None:
                        transcriptions = self.processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                        if transcriptions:
                            all_predictions.extend(transcriptions)
                            all_references.extend(batch['text'])
                    else:
                        logging.warning("Generated IDs is None, skipping batch")
                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
                    continue

        if not all_predictions or not all_references:
            logging.warning("No valid predictions or references found")
            return {
                'wer': float('inf'),
                'samples': []
            }

        error_rate = wer(all_references, all_predictions)
        samples = list(zip(all_references, all_predictions))[:5]

        return {
            'wer': error_rate,
            'samples': samples
        }

    def save_predictions(self, dataloader, output_file):
        """Save predictions to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Generating predictions"):
                    processed_batch = self.processor.prepare_dataset(batch)
                    input_features = processed_batch['input_features'].to(self.device)

                    generated_ids = self.model.generate(input_features)
                    transcriptions = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )

                    for ref, pred in zip(batch['text'], transcriptions):
                        f.write(f"Reference: {ref}\n")
                        f.write(f"Prediction: {pred}\n")
                        f.write("-" * 50 + "\n")