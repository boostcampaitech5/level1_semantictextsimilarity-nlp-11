import pandas as pd
import re
import urllib.request
from soynlp.normalizer import repeat_normalize
from soynlp.tokenizer import RegexTokenizer

# download Korean stopwords file from provided link
stopword_url = 'https://www.ranks.nl/stopwords/korean'
with urllib.request.urlopen(stopword_url) as response:
    stopwords = response.read().decode().splitlines()

# create Korean tokenizer using soynlp library
tokenizer = RegexTokenizer()

def preprocess_text(text):
    # normalize repeated characters using soynlp library
    text = repeat_normalize(text, num_repeats=2)
    # remove special characters and numbers
    text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]', '', text)
    # tokenize text using soynlp tokenizer
    tokens = tokenizer.tokenize(text)
    # remove stopwords
    tokens = [token for token in tokens if not token in stopwords]
    # join tokens back into sentence
    text = ' '.join(tokens)
    return text

# load csv data
data = pd.read_csv('./data/train.csv')

# remove null values
data = data.dropna()

# preprocess sentence_1 and sentence_2 columns
data['sentence_1'] = data['sentence_1'].apply(lambda x: preprocess_text(x))
data['sentence_2'] = data['sentence_2'].apply(lambda x: preprocess_text(x))

# save preprocessed data to csv
data.to_csv('preprocessed_data.csv', index=False)