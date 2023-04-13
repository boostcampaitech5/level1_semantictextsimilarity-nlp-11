import os

import pandas as pd
from soynlp.normalizer import repeat_normalize
from soynlp.tokenizer import RegexTokenizer
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

stopwords = pd.read_csv('./data/stopwords.csv',encoding='cp949')
Regextokenizer = RegexTokenizer()
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
        self.state = state
        if self.state == 'train':
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
        self.stopwords = pd.read_csv('./data/stopwords.csv', encoding='cp949')

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            if self.state=='train':
                target_val = self.targets[idx]
                if self.delete_columns is not None and self.data.iloc[idx][self.delete_columns] == 1:
                    if random.random() <= 0.2:
                        target_val += random.uniform(0.0, 0.1)
                else:
                    if random.random() <= 0.2:
                        target_val -= random.uniform(0.0, 0.1)

                target_val = max(min(target_val, 5.0), 0.0)
                return {"input_ids": torch.tensor(self.inputs[idx]), "labels": torch.tensor(target_val)}
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
            text = '[SEP]'.join([self.preprocess_text(item[text_column]) for text_column in self.text_columns])

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

    def preprocess_text(self,text):
        # normalize repeated characters using soynlp library
        text = repeat_normalize(text, num_repeats=2)
        # remove stopwords
        #text = ' '.join([token for token in text.split() if not token in stopwords])
        # remove special characters and numbers
        # text = re.sub('[^가-힣 ]', '', text)
        # text = re.sub('[^a-zA-Zㄱ-ㅎ가-힣]', '', text)
        # tokenize text using soynlp tokenizer
        tokens = Regextokenizer.tokenize(text)
        # lowercase all tokens
        tokens = [token.lower() for token in tokens]
        # join tokens back into sentence
        text = ' '.join(tokens)
        # kospacing_sent = spacing(text)
        return text

if __name__ == '__main__':

    seed_everything(43)
    model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator",num_labels=1,ignore_mismatched_sizes=True)
    #model = AutoModelForSequenceClassification.from_pretrained("E:/nlp/checkpoint/best_acc/checkpoint-16317",num_labels=1,ignore_mismatched_sizes=True)


    Train_textDataset = Train_val_TextDataset('train','./data/train.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=512,model_name="monologg/koelectra-base-v3-discriminator")
    Val_textDataset = Train_val_TextDataset('val','./data/dev.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=512,model_name="monologg/koelectra-base-v3-discriminator")

    args = TrainingArguments(
        "E:/nlp/checkpoint/best_acc_/koelectra-labelsoomthing_0.4_0.1_0.2",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=0.00002860270719188072, #0.000005
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.5,
        load_best_model_at_end=True,
        dataloader_num_workers = 4,
        logging_steps=200,
        seed = 43
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