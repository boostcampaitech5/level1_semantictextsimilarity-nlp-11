from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from tqdm.auto import tqdm
from utils import compute_pearson_correlation, seed_everything
from dataloader import Dataset
import pandas as pd
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class Train_val_TextDataset(torch.utils.data.Dataset):
#     def __init__(self,data_file, state,text_columns, target_columns=None, delete_columns=None, max_length=512, model_name='klue/roberta-small'):
#         self.state = state
#         if self.state == 'train':
#             self.data = pd.read_csv(data_file)
#         else:
#             self.data = pd.read_csv(data_file)
#         self.text_columns = text_columns
#         self.target_columns = target_columns if target_columns is not None else []
#         self.delete_columns = delete_columns if delete_columns is not None else []
#         self.max_length = max_length
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.inputs, self.targets = self.preprocessing(self.data)

#     def __getitem__(self, idx):
#         if len(self.targets) == 0:
#             return torch.tensor(self.inputs[idx])
#         else:
#             return {"input_ids": torch.tensor(self.inputs[idx]), "labels": torch.tensor(self.targets[idx])}

#     def __len__(self):
#         return len(self.inputs)

#     def tokenizing(self, dataframe):
#         data=[]
#         for idx, item in tqdm(dataframe.iterrows(), desc='Tokenizing', total=len(dataframe)):
#             text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
#             outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)
#             data.append(outputs['input_ids'])
#         return data

#     def preprocessing(self, data):
#         data = data.drop(columns=self.delete_columns)
#         try:
#             targets = data[self.target_columns].values.tolist()
#         except:
#             targets = []
#         inputs = self.tokenizing(data)
#         return inputs, targets



if __name__ == '__main__':

    seed_everything(43)
    model_name = "kykim/electra-kor-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,
                                                               ignore_mismatched_sizes=True)



    Train_textDataset = Dataset('./data/best_data_v1.csv', 'train', ['sentence_1', 'sentence_2'], 'label',
                                              'binary-label', max_length=256,
                                              model_name=model_name)
    Val_textDataset = Dataset('./data/dev.csv', 'val', ['sentence_1', 'sentence_2'], 'label',
                                            'binary-label', max_length=256,
                                            model_name=model_name)

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
    )



    trainer = Trainer(
        model = model,
        args = args,
        train_dataset=Train_textDataset,
        eval_dataset=Val_textDataset,
        compute_metrics=compute_pearson_correlation,
    )

    trainer.train()