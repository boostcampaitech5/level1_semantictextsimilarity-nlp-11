import torch
import pandas as pd
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class Infer_TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, text_columns, target_columns=None, delete_columns=None, max_length=512, model_name='klue/roberta-small'):
        self.data = pd.read_csv(data_file)
        self.text_columns = text_columns
        self.delete_columns = delete_columns if delete_columns is not None else []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.inputs = self.preprocessing(self.data)

    def __getitem__(self, idx):
            return {"input_ids": torch.tensor(self.inputs[idx])}

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
        inputs = self.tokenizing(data)
        return inputs

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = '/opt/ml/level1_semantictextsimilarity-nlp-11/param_sweep/checkpoint/monologg/koelectra-base-v3-discriminator/train.csv/2023-04-16 08:50:52.114287/checkpoint-3498'
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, ignore_mismatched_sizes=True)
    model.to(device)
    test_textDataset = Infer_TextDataset('./data/test.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=512,model_name="monologg/koelectra-base-v3-discriminator")
    test_dataloader = DataLoader(dataset=test_textDataset,
                                 batch_size=4,
                                 num_workers=0,
                                 shuffle=False,
                                 drop_last=False)
    score = []
    model.eval()
    with torch.no_grad():
        for batch_id, x in enumerate(tqdm(test_dataloader)):
            y_pred = model(x['input_ids'].to(device))
            logits = y_pred.logits
            y_pred = logits.detach().cpu().numpy()
            score.extend(y_pred)
    score = list(float(i) for i in score)
    #predictions = list(round(float(i), 1) for i in torch.cat(output))
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = score
    output.to_csv('output.csv', index=False)