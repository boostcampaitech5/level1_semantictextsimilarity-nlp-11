from transformers import AutoTokenizer, AutoModelForPreTraining

tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-ELECTRA-discriminator")

model = AutoModelForPreTraining.from_pretrained("snunlp/KR-ELECTRA-discriminator")

print(model)