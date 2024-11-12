from dotenv import load_dotenv
import argparse
from tqdm import tqdm

import torch
from torch.utils import data
from data_loader import DFGenerator, StoryPointDataset
from models import get_model_and_tokenizer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Model evaluation details')

    # Dataset details
    parser.add_argument('--eval_split', type=float, default=0.2, help='Evaluation split')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum number of tokens in input sequence')

    # Model details
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to evaluate')
    return parser.parse_args()


class Evaluator():

    def __init__(self, args):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.tokenizer = get_model_and_tokenizer(args.checkpoint)
        self.model.to(self.device)

        df_generator = DFGenerator(eval_split=args.eval_split, random_seed=args.random_seed)

        _, val_df = df_generator.create_dataframes()
        if val_df is None:
            raise ValueError('No validation data available')

        val_dataset = StoryPointDataset(val_df, self.tokenizer, max_length=args.max_length)
        self.val_loader = data.DataLoader(val_dataset, batch_size=1)


    def evaluate(self):

        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(self.val_loader, desc=f'Validation'):
            inputs = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)

            _, predicted = torch.max(outputs.logits, 1)
            correct_predictions += (predicted == batch['labels'].to(self.model.device)).sum().item()
            total_predictions += len(batch['labels'])

        accuracy = correct_predictions / total_predictions
        validation_message = f'Validation Accuracy: {accuracy:.4f}'

        print(validation_message)


if __name__ == '__main__':
    
    load_dotenv()
    args = parse_args()

    evaluator = Evaluator(args)
    evaluator.evaluate()