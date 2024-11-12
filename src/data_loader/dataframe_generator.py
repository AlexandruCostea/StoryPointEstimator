import os
import glob
from dotenv import load_dotenv

import pandas as pd


class DFGenerator:

    '''
    Class to generate dataframes from csv files in the data_path directory.
    '''

    def __init__(self, eval_split = 0.1, random_seed = 42):

        load_dotenv()
        self.data_path = os.getenv('DATA_PATH')
        self.eval_split = eval_split
        self.random_seed = random_seed

    
    def create_dataframes(self):

        '''
        Reads all csv files in the data_path directory and creates a dataframe out of them.
        The dataframe is then filtered to only include storypoints that are fibonacci numbers <= 13.

        Dataframe structure: (text: str, label: int)
            - text: made by concatenating the title and description columns
            - label: position in fibonacci sequence of the storypoint value

        Returns:
            train_df, eval_df: dataframes containing the training and evaluation data
            if eval_split is 0, eval_df will be None
        '''

        
        data_files = glob.glob(self.data_path + '/*.csv')
        data = []

        for filename in data_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            data.append(df)
            storypoint_df = pd.concat(data, axis=0, ignore_index=True)

        fibonacci_numbers = [1, 2, 3, 5, 8, 13]
        storypoint_df = storypoint_df[storypoint_df['storypoint'].isin(fibonacci_numbers)]

        storypoint_df['text'] = storypoint_df['title'] + ' ' + storypoint_df['description'].fillna('')
        storypoint_df = storypoint_df[['text', 'storypoint']]

        label_mapping = {1: 0, 2: 1, 3: 2, 5: 3, 8: 4, 13: 5}
        storypoint_df['label'] = storypoint_df['storypoint'].map(label_mapping)
        storypoint_df = storypoint_df[['text', 'label']]

        if self.eval_split == 0:
            return storypoint_df, None
        
        elif self.eval_split >=1:
            return None, storypoint_df
        
        validation_df = pd.DataFrame()
        label_counts_total = self._label_counts(storypoint_df, 'label')

        for label in label_counts_total.index:
            label_data = storypoint_df[storypoint_df['label'] == label]
            validation_size = int(len(label_data) * self.eval_split)
            validation_sample = label_data.sample(n=validation_size, random_state=self.random_seed)
            validation_df = pd.concat([validation_df, validation_sample], ignore_index=True)

        train_df = storypoint_df[~storypoint_df.index.isin(validation_df.index)].copy()

        return train_df, validation_df

    
    def _label_counts(self, df, column_name):
        
        label_counts = df[column_name].value_counts()
        return label_counts