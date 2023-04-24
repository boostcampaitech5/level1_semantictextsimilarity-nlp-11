from transformers import AutoTokenizer, AutoModelForPreTraining

tokenizer = AutoTokenizer.from_pretrained("kykim/electra-kor-base")

model = AutoModelForPreTraining.from_pretrained("kykim/electra-kor-base")
print(model)