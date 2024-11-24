import torch
from tqdm import tqdm
from jiwer import wer

class Evaluator:
    def __init__(self, model, processor, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize evaluator with model and processor."""
        self.model = model
        self.processor = processor
        self.device = device
        self.model.to(device)
        self.model.eval()

    def compute_metrics(self, dataloader):
        """Compute WER and other metrics."""
        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                processed_batch = self.processor.prepare_dataset(batch)
                input_features = processed_batch['input_features'].to(self.device)

                # Generate predictions
                generated_ids = self.model.generate(input_features)
                transcriptions = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                all_predictions.extend(transcriptions)
                all_references.extend(batch['text'])

        # Compute WER
        error_rate = wer(all_references, all_predictions)

        # Sample predictions
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
