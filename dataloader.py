from transformers import AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd 
import torch

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self,data_file, state, text_columns, target_columns=None, delete_columns=None, max_length=512, model_name='klue/roberta-small'):
        self.state = state
        self.data = pd.read_csv(data_file)
        self.text_columns = text_columns
        self.delete_columns = delete_columns if delete_columns is not None else []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.state == 'test':
            self.inputs = self.preprocessing(self.data)
        else : 
            self.target_columns = target_columns if target_columns is not None else []
            self.inputs, self.targets = self.preprocessing(self.data)

    def __getitem__(self, idx):
        if self.state == 'test':
            return {'input_ids': torch.tensor(self.inputs[idx])}
        else : 
            if len(self.targets) == 0:
                return torch.tensor(self.inputs[idx])
            else:
                return {'input_ids': torch.tensor(self.inputs[idx]), 'labels': torch.tensor(self.targets[idx])}
        
    def __len__(self):
        return len(self.inputs)

    def tokenizing(self, dataframe):
        data=[]
        for _, item in tqdm(dataframe.iterrows(), desc='Tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        inputs = self.tokenizing(data)
        if self.state == 'test':
            return inputs
        else : 
            return inputs, targets

