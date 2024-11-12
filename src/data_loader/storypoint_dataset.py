import torch
from torch.utils.data import Dataset


class StoryPointDataset(Dataset):
    
    '''
    Dataset class for the storypoint data.
    Requires a dataframe with columns 'text' and 'label' and a tokenizer.
    '''

    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):

        return len(self.dataframe)
    

    def __getitem__(self, idx):
        
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['label']
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(label)
        return item