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
    random.seed(seed) # Python의 random 모듈의 시드를 설정
    os.environ['PYTHONHASHSEED'] = str(seed) # Python 해시 함수의 시드를 설정
    np.random.seed(seed) # NumPy의 무작위 함수인 random 모듈의 시드를 설정
    torch.manual_seed(seed) # PyTorch의 무작위 함수를 위한 시드를 설정
    torch.cuda.manual_seed(seed) # PyTorch의 CUDA(Compute Unified Device Architecture) 연산을 위한 시드를 설정
    torch.backends.cudnn.deterministic = True # PyTorch의 cuDNN(CUDA Deep Neural Network) 라이브러리의 동작을 재현 가능하게 만들기 위한 옵션을 설정
    # PyTorch의 cuDNN 라이브러리가 입력 데이터의 크기에 따라 최적의 컨볼루션 연산 알고리즘을 선택하는 옵션을 설정
    # True로 설정하면 동일한 입력 데이터에 대해서도 항상 같은 알고리즘이 선택되어, 동일한 결과를 재현할 수 있다.
    torch.backends.cudnn.benchmark = True

def compute_pearson_correlation(pred):
    preds = pred.predictions.flatten()
    labels = pred.label_ids.flatten()
    return {"pearson_correlation": pearsonr(preds, labels)[0]}

class Train_val_TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, text_columns, target_columns=None, delete_columns=None, max_length=512, model_name='klue/roberta-small'):
        """
        클래스의 생성자 메서드로, 데이터셋을 초기화한다.
        Args:
            data_file (string): 데이터 파일의 경로
            text_columns (list): 텍스트 데이터를 갖는 column들의 이름
            target_columns (string or list, optional): 레이블 데이터를 갖는 column의 이름
            delete_columns (string or list, optional): 제거할 column의 이름
            max_length (int, optional): 최대 텍스트 길이
            model_name ()
        """
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
    wandb.login(key='c0d96b72557660bd63642b07a905e93575f72573')

    sweep_config = {
        'method': 'random'
    }

    # hyperparameters
    parameters_dict = {
        'epochs': {
            'value': 8
        },
        'batch_size': {
            'values': [4,8,16]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 5e-5
        },
        'weight_decay': {
            'values': [0.4,0.5]
        },
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="mdeberta-v3-base-kor-further_weight_decay")

    model = AutoModelForSequenceClassification.from_pretrained("lighthouse/mdeberta-v3-base-kor-further",num_labels=1,ignore_mismatched_sizes=True)

    #model = transformers.AutoModelForSequenceClassification.from_pretrained(
    #   'C:/Users/tm011/PycharmProjects/NLP_COMP/checkpoint/checkpoint-6993')
    Train_textDataset = Train_val_TextDataset('./data/train.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=512,model_name="lighthouse/mdeberta-v3-base-kor-further")
    Val_textDataset = Train_val_TextDataset('./data/dev.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=512,model_name="lighthouse/mdeberta-v3-base-kor-further")


    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained("lighthouse/mdeberta-v3-base-kor-further",num_labels=1,ignore_mismatched_sizes=True)
        return model


    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            t = config.learning_rate
            args = TrainingArguments(
                f"E:/nlp/checkpoint/baseline_Test_fine_{t}",
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