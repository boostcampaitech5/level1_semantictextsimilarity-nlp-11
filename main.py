from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import compute_pearson_correlation, seed_everything
from dataloader import Dataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    seed_everything(43)
    model_name = 'kykim/electra-kor-base'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,
                                                               ignore_mismatched_sizes=True)



    Train_textDataset = Dataset('./data/best_data_v1.csv', ['sentence_1', 'sentence_2'], 'label',
                                              'binary-label', max_length=256,
                                              model_name=model_name)
    Val_textDataset = Dataset('./data/dev.csv', ['sentence_1', 'sentence_2'], 'label',
                                            'binary-label', max_length=256,
                                            model_name=model_name)

    args = TrainingArguments(
        'E:/nlp/checkpoint/elector/electra-kor-base_jehyun__RE_RE',
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        learning_rate=0.00002071889728509824,
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