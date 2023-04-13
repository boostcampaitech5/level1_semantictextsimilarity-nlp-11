import torch
import pandas as pd
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from soynlp.normalizer import repeat_normalize
from soynlp.tokenizer import RegexTokenizer

tokenizer = RegexTokenizer()
stopwords = pd.read_csv('./data/stopwords.csv',encoding='cp949')


def preprocess_text(text):
    # normalize repeated characters using soynlp library
    text = repeat_normalize(text, num_repeats=2)
    # remove stopwords
    text = ' '.join([token for token in text.split() if not token in stopwords])
    # remove special characters and numbers
    #text = re.sub('[^가-힣 ]', '', text)
    #text = re.sub('[^a-zA-Zㄱ-ㅎ가-힣]', '', text)
    # tokenize text using soynlp tokenizer
    tokens = tokenizer.tokenize(text)
    # lowercase all tokens
    tokens = [token.lower() for token in tokens]
    # join tokens back into sentence
    text = ' '.join(tokens)
    #kospacing_sent = spacing(text)
    return text
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
            text = '[SEP]'.join([preprocess_text(item[text_column]) for text_column in self.text_columns])
            print(text)
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        inputs = self.tokenizing(data)
        return inputs

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = transformers.AutoModelForSequenceClassification.from_pretrained('E:/nlp/checkpoint/best_acc_mdeberta_preproceess_include_en/checkpoint-8162')
    model.to(device)
    test_textDataset = Infer_TextDataset('./data/test.csv',['sentence_1', 'sentence_2'],'label','binary-label',max_length=512,model_name="lighthouse/mdeberta-v3-base-kor-further")
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
    output.to_csv('pretest.csv', index=False)