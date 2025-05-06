from dotenv import load_dotenv
import argparse
import pandas as pd

import torch
from torch.utils import data
from data_loader import StoryPointDataset
from models import get_model_and_tokenizer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Model estimation details')

    # Model details
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to estimate with')
    return parser.parse_args()


class Estimator():

    def __init__(self, args):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.tokenizer = get_model_and_tokenizer(args.checkpoint)
        self.model.to(self.device)
        self.model.eval()


    def estimate(self, title, description):
        input_data = {
            'title': [title],
            'description': [description],
            'label': [0]
        }
        df = pd.DataFrame(input_data)
        df['text'] = df['title'] + ' ' + df['description'].fillna('')
        df = df[['text', 'label']]

        dataset = StoryPointDataset(df, self.tokenizer)
        dataloader = data.DataLoader(dataset, batch_size=1)

        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                outputs = self.model(**inputs)
                predicted = torch.argmax(outputs.logits, dim=1).item()

        label_mapping = {0: 1, 1: 2, 2: 3, 3: 5, 4: 8, 5: 13, 6: 14}
        return label_mapping[predicted]



if __name__ == '__main__':
    
    load_dotenv()
    args = parse_args()
    estimator = Estimator(args)

    title = input("Enter the title of the task: ")
    description = input("Enter the description of the task: ")

    output = estimator.estimate(title, description)
    print(f"Estimated story point: {output}")