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
            model_name (string, optional): 사용할 토크나이저의 모델 이름
        """
        self.data = pd.read_csv(data_file)
        self.text_columns = text_columns
        self.target_columns = target_columns if target_columns is not None else []
        self.delete_columns = delete_columns if delete_columns is not None else []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.inputs, self.targets = self.preprocessing(self.data)

    def __getitem__(self, idx):
        """
        데이터셋의 특정 인덱스(idx)에 해당하는 데이터를 반환하는 메서드
        """
        if len(self.targets) == 0: # 레이블이 없는 경우
            return torch.tensor(self.inputs[idx]) # 텍스트 데이터만 반환
        else: # 레이블이 있는 경우
            # 딕셔너리 형태로 텍스트 데이터와 레이블을 반환
            return {"input_ids": torch.tensor(self.inputs[idx]), "labels": torch.tensor(self.targets[idx])}

    def __len__(self):
        """
        데이터셋의 총 데이터 개수를 반환하는 메서드
        """
        return len(self.inputs)

    def tokenizing(self, dataframe):
        """
        데이터프레임(dataframe)을 입력으로 받아, 텍스트 데이터를 토크나이징하여 모델의 입력 형식에 맞게 변환하는 메서드
        """
        data=[]
        for idx, item in tqdm(dataframe.iterrows(), desc='Tokenizing', total=len(dataframe)):
            # text_columns에 지정된 열들의 값을 [SEP]으로 구분하여 이어붙인다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns]) # self.text_columns 예) ['sentence_1', 'sentence_2']
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)
            data.append(outputs['input_ids']) # 토크나이징된 input_ids를 반환
        return data

    def preprocessing(self, data):
        """
        데이터프레임('data')를 입력 받아, 전처리를 수행하는 메서드
        """
        data = data.drop(columns=self.delete_columns) # 삭제할 열을 제거
        try:
            targets = data[self.target_columns].values.tolist() # 레이블 열의 값을 리스트로 변환
        except:
            targets = []
        inputs = self.tokenizing(data) # 텍스트 데이터를 토크나이징
        return inputs, targets

if __name__ == '__main__':

    model_name = "snunlp/KR-ELECTRA-discriminator"

    seed_everything(42)
    wandb.login(key='c0d96b72557660bd63642b07a905e93575f72573')

    # 튜닝할 하이퍼 파라미터의 범위를 지정
    # parameters_dict = {
    #     'epochs': {
    #         'value': 8
    #     },
    #     'batch_size': {
    #         'values': [4,8,16]
    #     },
    #     'learning_rate': {
    #         'distribution': 'log_uniform_values',
    #         'min': 1e-5,
    #         'max': 5e-5
    #     },
    #     'weight_decay': {
    #         'values': [0.3,0.4,0.5]
    #     },
    # }

    parameters_dict = {
        'epochs': {
            'value': [8]  
        },
        'batch_size': {
            'value': [4]  
        },
        'learning_rate': {
            'value': [0.000018234535374473915]
        },
        'weight_decay': {
            'value': [0.5]
        },
    }

    # 하이퍼 파라미터 sweep config
    sweep_config = {
        'method': 'bayes',
        'parameters': parameters_dict,
        'metric':{
            'name': 'val_pearson',
            'goal': 'maximize'
        }
    }

    # wandb를 사용하여 sweep를 생성하고, sweep_id를 반환받는다.
    sweep_id = wandb.sweep(sweep_config, project=f"{model_name.replace('/', '_')}_new_dataset")

    # model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-finetuned-nsmc",num_labels=1,ignore_mismatched_sizes=True)
    
    Train_textDataset = Train_val_TextDataset('./data/best_data_v1.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=200,model_name=model_name)
    Val_textDataset = Train_val_TextDataset('./data/dev.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=200,model_name=model_name)


    def model_init():
        """
        선학습된 모델 로드 후 분류를 위한 마지막 레이어 추가(num_labels=1)
        """
        model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=1,ignore_mismatched_sizes=True)
        return model


    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            t = config.learning_rate
            args = TrainingArguments(
                output_dir=f"./checkpoint/baseline_Test_fine_{t}",
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


    wandb.agent(sweep_id, train, count=10)