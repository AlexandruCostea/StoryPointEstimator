import os
from dotenv import load_dotenv
import argparse
import logging
import json
from tqdm import tqdm

import torch
from torch.utils import data
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

from data_loader import DFGenerator, StoryPointDataset
from models import get_model_and_tokenizer
from utils import serialize_metrics


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Model training details')

    # Dataset details
    parser.add_argument('--eval_split', type=float, default=0.1, help='Evaluation split')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum number of tokens in input sequence')

    # Model training details
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to resume training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--eval', type=bool, default=True, help='Evaluate model after each training epoch')

    # Model saving details
    parser.add_argument('--experiment_name', type=str, default='storypoint_estimator', help='Experiment name (for checkpoint naming)')
    return parser.parse_args()


class Trainer():

    def __init__(self, args):

        self.experiment_dir = f'experiments/{args.experiment_name}'
        self.checkpoints_dir = f'{self.experiment_dir}/checkpoints'

        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        log_path = f'{self.experiment_dir}/{args.experiment_name}.log'

        self.logger = logging.getLogger('training_pipeline')
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)

        params_path = f'{self.experiment_dir}/params.json'

        with open(params_path, 'w') as f:
            json.dump(vars(args), f, indent=4)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.tokenizer = get_model_and_tokenizer(args.checkpoint)
        self.model.to(self.device)

        df_generator = DFGenerator(eval_split=args.eval_split, random_seed=args.random_seed)

        train_df, val_df = df_generator.create_dataframes()
        if train_df is None:
            raise ValueError('No training data available')

        train_dataset = StoryPointDataset(train_df, self.tokenizer, max_length=args.max_length)
        val_dataset = StoryPointDataset(val_df, self.tokenizer, max_length=args.max_length) if val_df is not None else None
        
        self.train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = data.DataLoader(val_dataset, batch_size=1) if val_dataset is not None else None


        self.epochs = args.epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-7)


        self.eval = args.eval
        self.experiment_name = args.experiment_name


    def train(self):

        for epoch in range(self.epochs):
            self.model.train()
            correct_predictions = 0
            total_predictions = 0
            total_loss = 0

            all_labels = []
            all_predictions = []

            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}'):
                inputs = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**inputs)

                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)

                correct_predictions += (predicted == batch['labels'].to(self.model.device)).sum().item()
                total_predictions += len(batch['labels'])

                all_labels.extend(batch['labels'].cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)

            epoch_lr = self.optimizer.param_groups[0]['lr']
            self.lr_scheduler.step()

            average_loss = total_loss / len(self.train_loader)
            accuracy = correct_predictions / total_predictions
            macro_f1 = f1_score(all_labels, all_predictions, average='macro')
            weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
            precision = precision_score(all_labels, all_predictions, average='macro')
            recall = recall_score(all_labels, all_predictions, average='macro')
            conf_matrix = confusion_matrix(all_labels, all_predictions)

            metrics = (accuracy, macro_f1, weighted_f1, precision, recall, conf_matrix)


            epoch_message = f"Training - Epoch {epoch+1}/{self.epochs}: Loss - {average_loss:.4f}\t\tLR - {epoch_lr:.8f}\n"

            print(epoch_message)
            self.logger.info(epoch_message)


            if self.eval and self.val_loader:
                self.evaluate(epoch, metrics)
            
            else:
                self.save_checkpoint('train', metrics, epoch)



    def evaluate(self, epoch, train_metrics = None):

        if self.val_loader is None:
            return
        
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        all_labels = []
        all_predictions = []

        for batch in tqdm(self.val_loader, desc=f'Validation - Epoch {epoch+1}/{self.epochs}'):
            inputs = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)

            _, predicted = torch.max(outputs.logits, 1)
            correct_predictions += (predicted == batch['labels'].to(self.model.device)).sum().item()
            total_predictions += len(batch['labels'])

            all_labels.extend(batch['labels'].cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        accuracy = correct_predictions / total_predictions
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        metrics = (accuracy, macro_f1, weighted_f1, precision, recall, conf_matrix)

        if train_metrics is not None:
            all_metrics = list(train_metrics)
            all_metrics.extend(metrics)
            metrics = tuple(all_metrics)

            self.save_checkpoint('both', metrics, epoch)

        else:
            self.save_checkpoint('val', metrics, epoch)

 
    def save_checkpoint(self, mode, metrics, epoch=0):

        json_data = serialize_metrics(mode, metrics)

        checkpoint_dir = f'{self.checkpoints_dir}/epoch_{epoch+1}'
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(self.model.state_dict(), f'{checkpoint_dir}/{self.experiment_name}_{epoch+1}.pth')
        with open(f'{checkpoint_dir}/metrics.json', 'w') as f:
            f.write(json_data)



if __name__ == '__main__':
    
    load_dotenv()
    args = parse_args()

    trainer = Trainer(args)
    trainer.train()