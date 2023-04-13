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
import nltk
from nltk.corpus import stopwords
import re
from soynlp.normalizer import repeat_normalize
from soynlp.tokenizer import RegexTokenizer
# from konlpy.tag import Hannanum TODO

def compute_pearson_correlation(pred):
    preds = pred.predictions.flatten()
    labels = pred.label_ids.flatten()
    return {"pearson_correlation": pearsonr(preds, labels)[0]}


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



class Train_val_TextDataset(torch.utils.data.Dataset):
    def __init__(self, state, data_file, text_columns, target_columns=None, delete_columns=None, max_length=512, model_name='klue/roberta-small'):
        """
        클래스의 생성자 메서드로, 데이터셋을 초기화한다.
        Args:
            state (string): 데이터 셋의 상태
            data_file (string): 데이터 파일의 경로
            text_columns (list): 텍스트 데이터를 갖는 column들의 이름
            target_columns (string or list, optional): 레이블 데이터를 갖는 column의 이름
            delete_columns (string or list, optional): 제거할 column의 이름
        """
        if state == 'train':
            self.data = pd.read_csv(data_file)
            #self.add_data = pd.read_csv('./data/preprocessed_data_sin_v2_filter.csv')
            #self.data = pd.concat([self.data,self.add_data])
        else: # state == val
            self.data = pd.read_csv(data_file)
        self.text_columns = text_columns
        self.target_columns = target_columns if target_columns is not None else []
        self.delete_columns = delete_columns if delete_columns is not None else []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.inputs, self.targets = self.preprocessing(self.data)

        # create Korean stemmer and lemmatizer
        # self.stemmer = Hannanum() TODO

        # 한글 불용어 파일을 다운로드
        # wget -O korean_stopwords.txt https://www.ranks.nl/stopwords/korean

        # 파일의 내용을 읽어 불용어 리스트 생성
        # with open('korean_stopwords.txt', 'r', encoding='utf-8') as f:
        #     stopwords = f.read().splitlines()

        # self.stopwords = stopwords

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

    def preprocess_text_wrapper(self, text_list):
        text1, text2 = text_list
        return self.preprocess_text(text1), self.preprocess_text(text2)

    def preprocess_text(self, text):
        # create Korean tokenizer using soynlp library
        # tokenizer = RegexTokenizer()

        # 2회 이상 반복된 문자를 정규화
        text = repeat_normalize(text, num_repeats=2)
        # 불용어 제거
        # text = ' '.join([token for token in text.split() if not token in stopwords])
        # 대문자를 소문자로 변경
        text = text.lower()
        # "<PERSON>"을 "사람"으로 변경
        text = re.sub('<PERSON>', '사람', text)
        # 한글 문자, 영어 문자, 공백 문자를 제외한 모든 문자 제거
        text = re.sub('[^가-힣a-z\\s]', '', text)
        # 텍스트를 토큰으로 분리  예) "안녕하세요" -> "안녕", "하", "세요"
        # tokens = tokenizer.tokenize(text)
        # 어간 추출
        # tokens = [self.stemmer.morphs(token)[0] for token in text.split()]
        # join tokens back into sentence
        # text = ' '.join(tokens)
        return text

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
            targets = data[self.target_columns].apply(lambda x: min(5, round(max(0, x + 0.3),2)) if x >= 2.5 else min(5, round(max(0, x - 0.3),2))).values.tolist()
        except:
            targets = []
        inputs = self.tokenizing(data)
        return inputs, targets



if __name__ == '__main__':

    seed_everything(42)
    model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-finetuned-nsmc",num_labels=1,ignore_mismatched_sizes=True)
    #model = AutoModelForSequenceClassification.from_pretrained("E:/nlp/checkpoint/best_acc/checkpoint-16317",num_labels=1,ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    Train_textDataset = Train_val_TextDataset('train','./data/train.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=512,model_name="monologg/koelectra-base-finetuned-nsmc")
    Val_textDataset = Train_val_TextDataset('val','./data/dev.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=512,model_name="monologg/koelectra-base-finetuned-nsmc")


    args = TrainingArguments(
        "./checkpoint/baseline_Test_fine_3.073982620831417e-05",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=0.00003073982620831417,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.2,
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