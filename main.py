# Code Refactoring

import os

import pandas as pd
from soynlp.normalizer import repeat_normalize
from soynlp.tokenizer import RegexTokenizer
from tqdm.auto import tqdm
import transformers
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForPreTraining, \
    AutoModel
from transformers import ElectraModel, ElectraTokenizer
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
from scipy.stats import pearsonr
import random
import torch.nn.functional as F
from torch import nn
import wandb
from lion_pytorch import Lion


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#stopwords = pd.read_csv('./data/stopwords.csv',encoding='cp949')
#stopwords = list(stopwords['stop_words'])

#Regextokenizer = RegexTokenizer()
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
    def __init__(self,data_file, state,text_columns, target_columns=None, delete_columns=None, max_length=512, model_name='klue/roberta-small'):
        self.state = state
        if self.state == 'train':
            self.data = pd.read_csv(data_file)
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
            # if self.state == 'train':
            #     target_val = self.targets[idx]
            #     random1 = random.random()
            #     if random1 <= 0.5:
            #         add_score = random.uniform(0.0, 0.15)
            #         if random.random() >= 0.5:
            #             target_val += add_score
            #         else:
            #             target_val -= add_score
            #
            #     target_val = max(min(target_val, 5.0), 0.0)
            #     return {"input_ids": torch.tensor(self.inputs[idx]), "labels": torch.tensor(target_val)}

            return {"input_ids": torch.tensor(self.inputs[idx]), "labels": torch.tensor(self.targets[idx])}

    def __len__(self):
        return len(self.inputs)

    def tokenizing(self, dataframe):
        data=[]
        for idx, item in tqdm(dataframe.iterrows(), desc='Tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)
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

    seed_everything(43)
    model_name = "kykim/electra-kor-base"
    #model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=1,ignore_mismatched_sizes=True)
    #model = AutoModelForSequenceClassification.from_pretrained("E:/nlp/checkpoint/elector/best/checkpoint-8955",num_labels=1,ignore_mismatched_sizes=True)
    # Train_textDataset = Train_val_TextDataset('./data/best_data_v1.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=256,model_name=model_name)
    # Val_textDataset = Train_val_TextDataset('./data/dev.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=256,model_name=model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,
                                                               ignore_mismatched_sizes=True)



    Train_textDataset = Train_val_TextDataset('./data/best_data_v1.csv', 'train', ['sentence_1', 'sentence_2'], 'label',
                                              'binary-label', max_length=256,
                                              model_name=model_name)
    Val_textDataset = Train_val_TextDataset('./data/dev.csv', 'val', ['sentence_1', 'sentence_2'], 'label',
                                            'binary-label', max_length=256,
                                            model_name=model_name)

    #opt = Lion(model.parameters(), lr=0.00001860270719188072, weight_decay=0.5)
    args = TrainingArguments(
        "E:/nlp/checkpoint/elector/electra-kor-base_jehyun__RE_RE",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=0.00002071889728509824, #0.000005
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.5,
        load_best_model_at_end=True,
        dataloader_num_workers = 4,
        logging_steps=100,
        seed = 43,
        group_by_length=True,
        #lr_scheduler_type = 'cosine'
    )
#10 7 3
#8 6 2
#10 8 2
#최댓값 3


    trainer = Trainer(
        model = model,
        args = args,
        train_dataset=Train_textDataset,
        eval_dataset=Val_textDataset,
        #tokenizer=tokenizer,
        compute_metrics=compute_pearson_correlation,
    )

    trainer.train()