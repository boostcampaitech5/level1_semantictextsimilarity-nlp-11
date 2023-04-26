import os

import pandas as pd
import torch
import transformers
from dataloader import CustomDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import load_yaml

prj_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    config_path = os.path.join(prj_dir, "config_yaml", "test.yaml")
    config = load_yaml(config_path)

    model_list = ["xlm_roberta_large", "snunlp", "kykim"]
    model_name = model_list[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        os.path.join(prj_dir, "save_folder", config["checkpoint"][model_name])
    )

    test_text_dataset = CustomDataset(
        data_file=config["data_folder"]["test_data"],
        state="test",
        text_columns=["sentence_1", "sentence_2"],
        target_columns=None,
        delete_columns=None,
        max_length=256,
        model_name=config["name"][model_name],
    )
    test_dataloader = DataLoader(
        dataset=test_text_dataset,
        batch_size=4,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    score = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_id, x in enumerate(tqdm(test_dataloader)):
            y_pred = model(x["input_ids"].to(device))
            logits = y_pred.logits
            y_pred = logits.detach().cpu().numpy()
            score.extend(y_pred)
    score = list(float(i) for i in score)
    output = pd.read_csv(config["data_folder"]["submission"])
    output["target"] = score
    output.to_csv(f"./output/{model_name}.csv", index=False)
