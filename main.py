import os

import pandas as pd
from tqdm.auto import tqdm
import transformers
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForPreTraining
from transformers import ElectraModel, ElectraTokenizer
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
from scipy.stats import pearsonr
import random
from datetime import datetime

#import nltk
# from nltk.corpus import stopwords
def compute_pearson_correlation(pred):
    preds = pred.predictions.flatten()
    labels = pred.label_ids.flatten()
    return {"pearson_correlation": pearsonr(preds, labels)[0]}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



class Train_val_TextDataset(torch.utils.data.Dataset):
    def __init__(self,state,data_file, text_columns, target_columns=None, delete_columns=None, max_length=512, model_name='klue/roberta-small'):
        if state == 'train':
            self.data = pd.read_csv(data_file)
            #self.add_data = pd.read_csv('./data/preprocessed_data_sin_v2_filter.csv')
            #self.data = pd.concat([self.data,self.add_data])
        else:
            self.data = pd.read_csv(data_file)
        self.text_columns = text_columns
        self.target_columns = target_columns if target_columns is not None else []
        self.delete_columns = delete_columns if delete_columns is not None else []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.inputs, self.targets = self.preprocessing(self.data)

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return {"input_ids": torch.tensor(self.inputs[idx]), "labels": torch.tensor(self.targets[idx])}

    def __len__(self):
        return len(self.inputs)

    def remove_stopwords(self, text):
        words = text.split()
        words = [word for word in words if word not in stopwords]
        return ' '.join(words)

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='Tokenizing', total=len(dataframe)):

            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            ##불용어 제거
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True,
                                     max_length=self.max_length)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        inputs = self.tokenizing(data)
        return inputs, targets



if __name__ == '__main__':

    seed_everything(42)
    #model = AutoModelForSequenceClassification.from_pretrained("lighthouse/mdeberta-v3-base-kor-further",num_labels=1,ignore_mismatched_sizes=True)
    
    model_name = "monologg/koelectra-base-v3-discriminator"
    train_data_name = './data/train.csv'

    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=1,ignore_mismatched_sizes=True)
    # model = AutoModelForSequenceClassification.from_pretrained('E:/nlp/checkpoint/TEST-6/checkpoint-4081', num_labels=1, ignore_mismatched_sizes=True)

    max_length = 256  # 원래 512



    Train_textDataset = Train_val_TextDataset('train', train_data_name ,['sentence_1', 'sentence_2'],'label','binary-label',max_length=max_length,model_name=model_name)
    Val_textDataset = Train_val_TextDataset('val','./data/dev.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=max_length,model_name=model_name)

    args = TrainingArguments(
        f'checkpoint/{model_name}/{train_data_name}/{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}',
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2.860270719188072e-05, #0.000005
        # group_by_length=True,
        # auto_find_batch_size=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=7,
        weight_decay=0.5,
        load_best_model_at_end=True,
        dataloader_num_workers = 4,
        logging_steps=200,
        seed = 42
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=Train_textDataset,
        eval_dataset=Val_textDataset,
        #tokenizer=tokenizer,
        compute_metrics=compute_pearson_correlation
    )

    trainer.train()