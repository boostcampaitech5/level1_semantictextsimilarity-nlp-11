from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataloader import CustomDataset
from utils import load_yaml
import pandas as pd
import os
import torch
import transformers

prj_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    config_path = os.path.join(prj_dir, 'config_yaml', 'test.yaml')
    config = load_yaml(config_path)

    #name_list = [xlm_robberta_large,snunlp,kykim]
    model_name = 'xlm_roberta_large'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = transformers.AutoModelForSequenceClassification.from_pretrained(os.path.join(prj_dir, 'save_folder', config['checkpoint'][model_name]))

    test_textDataset = CustomDataset(config['data_folder']['Test_data'],'test',['sentence_1', 'sentence_2'],None,None,max_length=256,model_name=config['name'][model_name])
    test_dataloader = DataLoader(dataset=test_textDataset,
                                 batch_size=4,
                                 num_workers=0,
                                 shuffle=False,
                                 drop_last=False)
    score = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_id, x in enumerate(tqdm(test_dataloader)):
            y_pred = model(x['input_ids'].to(device))
            logits = y_pred.logits
            y_pred = logits.detach().cpu().numpy()
            score.extend(y_pred)
    score = list(float(i) for i in score)
    output = pd.read_csv(config['data_folder']['Test_data'])
    output['target'] = score
    output.to_csv(f'./output/{model_name}.csv', index=False)
