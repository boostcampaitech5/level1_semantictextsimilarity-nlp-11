import os

import torch
from augmentation import augment
from dataloader import CustomDataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import compute_pearson_correlation, load_yaml, seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prj_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    model_list = ["xlm_roberta_large", "snunlp", "kykim"]
    model_name = model_list[0]
    config_path = os.path.join(prj_dir, "config_yaml", f"{model_name}.yaml")
    config = load_yaml(config_path)
    seed_everything(config["seed"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config["architecture"], num_labels=1, ignore_mismatched_sizes=True
    )

    train_text_dataset = CustomDataset(
        data_file=config["aug_data_folder"]["train_data"],
        state="train",
        text_columns=["sentence_1", "sentence_2"],
        target_columns="label",
        delete_columns="binary-label",
        max_length=256,
        model_name=config["architecture"],
    )
    val_text_dataset = CustomDataset(
        data_file=config["data_folder"]["val_data"],
        state="val",
        text_columns=["sentence_1", "sentence_2"],
        target_columns="label",
        delete_columns="binary-label",
        max_length=256,
        model_name=config["architecture"],
    )

    args = TrainingArguments(
        output_dir=os.path.join(prj_dir, "save_folder", config["name"]),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["n_epochs"],
        weight_decay=config["weight_decay"],
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        logging_steps=100,
        seed=config["seed"],
        group_by_length=True,
        lr_scheduler_type=config["scheduler"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_text_dataset,
        eval_dataset=val_text_dataset,
        compute_metrics=compute_pearson_correlation,
    )

    trainer.train()
