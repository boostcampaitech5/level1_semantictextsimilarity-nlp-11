import pandas as pd
import re
import requests
from soynlp.normalizer import repeat_normalize
from soynlp.tokenizer import RegexTokenizer
from konlpy.tag import Hannanum
from pykospacing import Spacing
# download Korean stopwords file from provided link



# create Korean tokenizer using soynlp library
tokenizer = RegexTokenizer()
spacing = Spacing()
# create Korean stemmer

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

# load csv data
data = pd.read_csv('./data/train.csv')

# drop rows with NaN values in sentence_1 column

# preprocess sentence_1 and sentence_2 columns
data['sentence_1'] = data['sentence_1'].apply(lambda x: preprocess_text(x))
data['sentence_2'] = data['sentence_2'].apply(lambda x: preprocess_text(x))

data = data.dropna(subset=['sentence_1'])
data = data.dropna(subset=['sentence_2'])

# save preprocessed data to csv
data.to_csv('./data/preprocessed_train_data_sin_v2.csv', index=False)

data = pd.read_csv('./data/preprocessed_train_data_sin_v2.csv')
data = data.dropna(subset=['sentence_1'])
data = data.dropna(subset=['sentence_2'])
data.to_csv('./data/preprocessed_train_data_sin_v2_filter_.csv', index=False)
