import os
from dotenv import load_dotenv
import argparse
import logging
import json
from tqdm import tqdm

import torch
from torch.utils import data
from data_loader import DFGenerator, StoryPointDataset
from models import get_model_and_tokenizer


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
    parser.add_argument('--save_epoch', type=bool, default=True, help='Save checkpoint after each training epoch (default saves only best model)')
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
        self.save_epoch = args.save_epoch
        self.experiment_name = args.experiment_name
        self.best_train_accuracy = 0
        self.best_val_accuracy = 0


    def train(self):

        for epoch in range(self.epochs):
            self.model.train()
            correct_predictions = 0
            total_predictions = 0
            total_loss = 0

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

            epoch_lr = self.optimizer.param_groups[0]['lr']
            self.lr_scheduler.step()

            average_loss = total_loss / len(self.train_loader)
            accuracy = correct_predictions / total_predictions

            epoch_message = f'Epoch {epoch+1}/{self.epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {epoch_lr:.8f}'
            print(epoch_message)
            self.logger.info(epoch_message)

            self.save_checkpoint('train', accuracy, epoch)

            if self.eval and self.val_loader:
                self.evaluate(epoch)


    def evaluate(self, epoch):

        if self.val_loader is None:
            return
        
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(self.val_loader, desc=f'Validation - Epoch {epoch+1}/{self.epochs}'):
            inputs = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)

            _, predicted = torch.max(outputs.logits, 1)
            correct_predictions += (predicted == batch['labels'].to(self.model.device)).sum().item()
            total_predictions += len(batch['labels'])

        accuracy = correct_predictions / total_predictions
        validation_message = f'Validation - Epoch {epoch+1}/{self.epochs} - Accuracy: {accuracy:.4f}'

        print(validation_message)
        self.logger.info(validation_message)

        self.save_checkpoint('val', accuracy)

 
    def save_checkpoint(self, mode, accuracy, epoch=0):

        if mode == 'train':
            if accuracy > self.best_train_accuracy:
                self.best_train_accuracy = accuracy
                torch.save(self.model.state_dict(), f'{self.checkpoints_dir}/{self.experiment_name}_best_train.pth')
            
            if self.save_epoch:
                torch.save(self.model.state_dict(), f'{self.checkpoints_dir}/{self.experiment_name}_{epoch+1}.pth')

        elif mode == 'val':
            if accuracy > self.best_val_accuracy:
                self.best_val_accuracy = accuracy
                torch.save(self.model.state_dict(), f'{self.checkpoints_dir}/{self.experiment_name}_best_val.pth')



if __name__ == '__main__':
    
    load_dotenv()
    args = parse_args()

    trainer = Trainer(args)
    trainer.train()