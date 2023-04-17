import os
import random

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
import wandb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def compute_pearson_correlation(pred):
    preds = pred.predictions.flatten()
    labels = pred.label_ids.flatten()
    return {"pearson_correlation": pearsonr(preds, labels)[0]}

class Train_val_TextDataset(torch.utils.data.Dataset):
    def __init__(self,data_file, state,text_columns, target_columns=None, delete_columns=None, max_length=512, model_name='klue/roberta-small'):
        self.state = state
        if self.state == 'train':
            self.data = pd.read_csv(data_file)
            self.add_data = pd.read_csv('./data/train_arg_hanspell_shuffle_RE.csv')
            self.data = pd.concat([self.data,self.add_data])
        else:
            self.data = pd.read_csv(data_file)
    def __init__(self,data_file, state,text_columns, target_columns=None, delete_columns=None, max_length=512, model_name='klue/roberta-small'):
        self.state = state
        if self.state == 'train':
            self.data = pd.read_csv(data_file)
            self.add_data = pd.read_csv('./data/train_arg_hanspell_shuffle_RE.csv')
            self.data = pd.concat([self.data,self.add_data])
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
            if self.state == 'train':
                target_val = self.targets[idx]
                random1 = random.random()
                if random1 <= 0.5:
                    add_score = random.uniform(0.0, 0.15)
                    if random.random() >= 0.5:
                        target_val += add_score
                    else:
                        target_val -= add_score

                target_val = max(min(target_val, 5.0), 0.0)
                return {"input_ids": torch.tensor(self.inputs[idx]), "labels": torch.tensor(target_val)}
            else:
                return {"input_ids": torch.tensor(self.inputs[idx]), "labels": torch.tensor(self.targets[idx])}
        else:
            if self.state == 'train':
                target_val = self.targets[idx]
                random1 = random.random()
                if random1 <= 0.5:
                    add_score = random.uniform(0.0, 0.15)
                    if random.random() >= 0.5:
                        target_val += add_score
                    else:
                        target_val -= add_score

                target_val = max(min(target_val, 5.0), 0.0)
                return {"input_ids": torch.tensor(self.inputs[idx]), "labels": torch.tensor(target_val)}
            else:
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
    seed_everything(42)
    wandb.login(key='38e2b6604d2670c05fd7f22edb2a711faf495709')

    sweep_config = {
        'method': 'random'
    }

    # hyperparameters
    parameters_dict = {
        'epochs': {
            'value': 10
            'value': 10
        },
        'batch_size': {
            'values': [4,8,16]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 6e-6,
            'max': 3e-5
            'min': 6e-6,
            'max': 3e-5
        },
        'weight_decay': {
            'values': [0.5]
            'values': [0.5]
        },
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project='electra-kor-base')
    seed_everything(43)
    model = AutoModelForSequenceClassification.from_pretrained('kykim/electra-kor-base',num_labels=1,ignore_mismatched_sizes=True)

    sweep_id = wandb.sweep(sweep_config, project='electra-kor-base')
    seed_everything(43)
    model = AutoModelForSequenceClassification.from_pretrained('kykim/electra-kor-base',num_labels=1,ignore_mismatched_sizes=True)

    #model = transformers.AutoModelForSequenceClassification.from_pretrained(
    #   'C:/Users/tm011/PycharmProjects/NLP_COMP/checkpoint/checkpoint-6993')
    Train_textDataset = Train_val_TextDataset('./data/train.csv','train',['sentence_1', 'sentence_2'],'label','binary-label',max_length=256,model_name='kykim/electra-kor-base')
    Val_textDataset = Train_val_TextDataset('./data/dev.csv','val',['sentence_1', 'sentence_2'],'label','binary-label',max_length=256,model_name='kykim/electra-kor-base')
    Train_textDataset = Train_val_TextDataset('./data/train.csv','train',['sentence_1', 'sentence_2'],'label','binary-label',max_length=256,model_name='kykim/electra-kor-base')
    Val_textDataset = Train_val_TextDataset('./data/dev.csv','val',['sentence_1', 'sentence_2'],'label','binary-label',max_length=256,model_name='kykim/electra-kor-base')


    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained('kykim/electra-kor-base',num_labels=1,ignore_mismatched_sizes=True)
        model = AutoModelForSequenceClassification.from_pretrained('kykim/electra-kor-base',num_labels=1,ignore_mismatched_sizes=True)
        return model


    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            t = config.learning_rate
            args = TrainingArguments(
                f"E:/nlp/funnel/PLZ{t}",
                f"E:/nlp/funnel/PLZ{t}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                report_to='wandb',
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                num_train_epochs=config.epochs,
                weight_decay=config.weight_decay,
                load_best_model_at_end=True,
                dataloader_num_workers=4,
                logging_steps=200,
                group_by_length = True,
                seed = 43
                group_by_length = True,
                seed = 43
            )
            trainer = Trainer(
                model_init = model_init,
                args=args,
                train_dataset=Train_textDataset,
                eval_dataset=Val_textDataset,
                # tokenizer=tokenizer,
                compute_metrics=compute_pearson_correlation


            )

            trainer.train()


    wandb.agent(sweep_id, train, count=20)
    wandb.agent(sweep_id, train, count=20)