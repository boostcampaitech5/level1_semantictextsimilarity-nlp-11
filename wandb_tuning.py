import pandas as pd
from tqdm.auto import tqdm
import transformers
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForPreTraining
from transformers import ElectraModel, ElectraTokenizer
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from scipy.stats import pearsonr
import random
import os
from datetime import datetime
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
    def __init__(self, data_file, text_columns, target_columns=None, delete_columns=None, max_length=512, model_name='klue/roberta-small'):
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
    f = open("/opt/ml/data/wandb_key.txt", 'r')
    key_wandb = f.readline()
    f.close()
    wandb.login(key=key_wandb)

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'goal': 'maximize', 
            'name': 'val_pearson'
        },
    }

    # hyperparameters
    parameters_dict = {
        'epochs': {
            'values': [8]  
        },
        'batch_size': {
            'values': [4]  
        },
        # 'learning_rate': {
        #     'distribution': 'log_uniform_values',
        #     'min': 0.000018234535374473915, # 0.00002
        #     'max': 0.000018234535374473915  # 0.00003
        #                    # 4~4.5
        # },
        'learning_rate': {
            'values': [0.000018234535374473915]
        },
        # 'warmup_steps': {
        #     'values': [0, 400, 800]
        # },
        'weight_decay': {
            'values': [0.5]
            # 'values': ['linear', 'cosine']
        },
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="snunlp_KR-ELECTRA-discriminator")

    model_name = "snunlp/KR-ELECTRA-discriminator"
    # model_name = "snunlp/KR-ELECTRA-discriminator"
    # model_name = "monologg/koelectra-base-v3-discriminator"
    # model_name = "lighthouse/mdeberta-v3-base-kor-further"
    # model_name = "jhn9803/Contract-new-tokenizer-mDeBERTa-v3-kor-further"

    train_data_name = 'best_data_v1.csv'
    max_length = 256  # 512
    
    # model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=1,ignore_mismatched_sizes=True)


    #model = transformers.AutoModelForSequenceClassification.from_pretrained(
    #   'C:/Users/tm011/PycharmProjects/NLP_COMP/checkpoint/checkpoint-6993')
    Train_textDataset = Train_val_TextDataset(f'./data/{train_data_name}',['sentence_1', 'sentence_2'],'label','binary-label',max_length=max_length,model_name=model_name)
    Val_textDataset = Train_val_TextDataset('./data/dev.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=max_length,model_name=model_name)


    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=1,ignore_mismatched_sizes=True)
        return model


    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            t = config.learning_rate
            args = TrainingArguments(
                f'param_sweep/checkpoint/snunlp/KR-ELECTRA-discriminator',
                evaluation_strategy="epoch",
                save_strategy= 'epoch', 
                # save_strategy="no",
                report_to='wandb',
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                num_train_epochs=config.epochs,
                weight_decay=config.weight_decay,
                # load_best_model_at_end=True,
                dataloader_num_workers=4,
                logging_steps=200,
                seed=42
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


    wandb.agent(sweep_id, train, count=15)