import pandas as pd
import numpy as np
import googletrans
from tqdm import tqdm
translator = googletrans.Translator()
translator.raise_Exception = True
# str1 = "나는 한국인 입니다."
# str2 = "I like burger."
# print(str1)
# result1 = translator.translate(str1, dest='en')
# result2 = translator.translate(str2, dest='ko')

data = pd.read_csv('./data/train.csv')
data = data[['sentence_1','sentence_2','label']]
data = np.array(data)
result1 = []
result2 = []
labels = []
for i in tqdm(range(len(data))):
    sentence_1 = translator.translate(data[i][0],dest='en')
    sentence_1 = translator.translate(sentence_1.text,dest='ko')

    sentence_2 = translator.translate(data[i][1], dest='en')
    sentence_2 = translator.translate(sentence_2.text, dest='ko')

    result1.append(sentence_1.text)
    result2.append(sentence_2.text)
    labels.append(data[i][2])

    if i % 1000 == 0 and i != 0:
        new_train = pd.DataFrame({ 'id' : list(range(0,len(result1))),
                          'source':['add_data' for i in range(len(result1))],
                              'sentence_1':result1,
                             'sentence_2':result2,
                             'label':labels,
                          'binary-label': list(range(0,len(result1)))})

        new_train.to_csv(f'./data/add_NMT_en_{i}.csv')

new_train = pd.DataFrame({ 'id' : list(range(0,len(result1))),
                          'source':['add_data' for i in range(len(result1))],
                              'sentence_1':result1,
                             'sentence_2':result2,
                             'label':labels,
                          'binary-label': list(range(0,len(result1)))})
new_train.to_csv('./data/add_NMT_en.csv', index = False)