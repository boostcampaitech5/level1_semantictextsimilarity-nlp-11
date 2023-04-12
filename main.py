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

    model = AutoModelForSequenceClassification.from_pretrained("lighthouse/mdeberta-v3-base-kor-further",num_labels=1,ignore_mismatched_sizes=True)

    Train_textDataset = Train_val_TextDataset('./train.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=512,model_name="lighthouse/mdeberta-v3-base-kor-further")
    Val_textDataset = Train_val_TextDataset('./dev.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=512,model_name="lighthouse/mdeberta-v3-base-kor-further")


    args = TrainingArguments(
        "E:/nlp/checkpoint/baseline_Test_",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=5e-6, #0.000005
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        dataloader_num_workers = 4,
        logging_steps=200,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=Val_textDataset,
        eval_dataset=Val_textDataset,
        #tokenizer=tokenizer,
        compute_metrics=compute_pearson_correlation
    )

    trainer.train()