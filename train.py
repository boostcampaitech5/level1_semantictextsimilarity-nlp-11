from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import compute_pearson_correlation, load_yaml,seed_everything
from augmentation import augment
from dataloader import CustomDataset
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prj_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    
    # name_list = [xlm_roberta_large,snunlp,kykim]
    model_name = 'xlm_roberta_large'
    config_path = os.path.join(prj_dir, 'config_yaml', 'kykim.yaml')
    config = load_yaml(config_path)
    seed_everything(config['seed'])

    if (os.path.isfile(config['aug_data_folder']['Train_data'])) == False:
        augment( config['data_folder']['Train_data'], config['aug_data_folder']['Train_data'])

    model = AutoModelForSequenceClassification.from_pretrained(config['architecture'], num_labels=1,ignore_mismatched_sizes=True)

    Train_textDataset = CustomDataset(config['aug_data_folder']['Train_data'], 'train', ['sentence_1', 'sentence_2'], 'label','binary-label', max_length=256,model_name=config['architecture'])
    Val_textDataset = CustomDataset(config['data_folder']['Val_data'], 'val', ['sentence_1', 'sentence_2'], 'label','binary-label', max_length=256,model_name=config['architecture'])
    
    args = TrainingArguments(
        os.path.join(prj_dir, 'save_folder', config['name']),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config['lr'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['n_epochs'],
        weight_decay=config['weight_decay'],
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        logging_steps=100,
        seed=config['seed'],
        group_by_length=True,
        lr_scheduler_type = config['scheduler']
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Train_textDataset,
        eval_dataset=Val_textDataset,
        compute_metrics=compute_pearson_correlation,
    )

    trainer.train()

