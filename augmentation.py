import pandas as pd
from pykospacing import Spacing
from konlpy.tag import *
import pickle
from hanspell import spell_checker
from soynlp.normalizer import repeat_normalize
import pandas as pd
spacing = Spacing()
okt = Okt()

def under_sampling(data_path: str)->pd.DataFrame:
    """
    label 값이 0인 데이터를 under sampling하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
        Returns:
            df_new (DataFrame): under sampling된 데이터
    """
    df = pd.read_csv(data_path)
    df_0 = df[df['label']==0][1000:2000].copy()
    df_new = df[df['label']!=0].copy()
    df_new = pd.concat([df_new, df_0])
    return df_new

def swap_sentence(data_path: str)->pd.DataFrame:
    """
    sentence 1과 sentence 2의 위치를 바꾸어 증강하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
        Returns:
            df_swapped (DataFrame): 증강된 데이터
    """
    df = pd.read_csv(data_path)
    df_swapped = df.copy()
    df_swapped["sentence_1"] = df["sentence_2"]
    df_swapped["sentence_2"] = df["sentence_1"]
    df_swapped = df_swapped[df_swapped['label'] != 0]
    return df_swapped

def copy_sentence(data_path: str)->pd.DataFrame:
    """
    sentence 1과 sentence 2에 같은 문장을 배치해 5점짜리 데이터를 생성하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
        Returns:
            df_copied (DataFrame): 증강된 데이터
    """
    df = pd.read_csv(data_path)
    df_copied = df[df['label']==0][250:750].copy()
    df_copied['sentence_1'] = df_copied['sentence_2']
    df_copied['label'] = 5.0
    df_copied.reset_index(inplace=True)
    return df_copied

def concat_data(data_path: str, *dataframes: pd.DataFrame):
    """
    데이터프레임을 합쳐서 csv 파일로 저장하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
            dataframes (DataFrame): 합치려고 하는 데이터프레임
    """
    result = pd.concat(dataframes)
    result.to_csv(data_path, index = False)

def augment(source_data_path, dest_data_path):
    under_sampled = under_sampling(source_data_path)
    swapped_sentence = swap_sentence(source_data_path)
    copied_sentence = copy_sentence(source_data_path)
    concat_data(dest_data_path, under_sampled, swapped_sentence, copied_sentence)


# SR 
def han_spell(text):
    text = repeat_normalize(text, num_repeats=2)
    text= text.lower()
    text = re.sub('[^a-z가-힣0-9 ]', '', text)
    text = text.strip()
    return spell_checker.check(text).as_dict()['checked']


def apply_hanspell(data):
    data['sentence_1'] = data['sentence_1'].apply(lambda x: han_spell(x))
    data['sentence_2'] = data['sentence_2'].apply(lambda x: han_spell(x))
    data = data.dropna(subset=['sentence_1'])
    data = data.dropna(subset=['sentence_2'])
    return data

# 받침이 있는지 확인
def check_end(noun):
    # 한글의 유니코드가 28로 나누어 떨어지면 받침 없다
    if (ord(noun[-1]) - ord('가')) % 28 == 0:
        return False
    # 한글의 유니코드가 28로 나누어 떨어지지 않으면 받침 있다
    else : 
        return True

# 종성에 따라 조사 바꾸기
def change_josa(none, josa):
    if josa == '이' or josa == '가':
        return '이' if check_end(none) else '가' 
    elif josa == '은' or josa == '는':
        return '은' if check_end(none) else '는'
    elif josa == '을' or josa == '를':
        return '을' if check_end(none) else '를'
    elif josa == '과' or josa == '와':
        return '과' if check_end(none) else '와'
    else:
        return josa


def make_sentence(sentence, c, sym):
    replace_sentence = []
    check = set(['이', '가', '을', '를', '과', '와'])
    for j in range(len(sentence)):
        # 문장에서 동의어를 추가한다.
        if sentence[j][0] == c:
            replace_sentence.append(sym)
            # 뒷말이 조사면 조사를 확인하고 바꾼다.
            if j+1 < len(sentence) and sentence[j+1][1] == 'Josa' and sentence[j+1][0] in check:
                # 바뀐 명사 마지막 받침 확인 후 조사 변경
                sentence[j+1]  = (change_josa(replace_sentence[-1][0] , sentence[j+1][0]), 'Josa')
        else: 
            replace_sentence.append(sentence[j][0])

    replace_sentence = ''.join(replace_sentence)
    replace_sentence = spacing(replace_sentence)                
    return replace_sentence


def sr_noun_replace(dt):
    dt = apply_hanspell(dt)
    n1, n2 = dt['sentence_1'],  dt['sentence_2'] 
    new_sentence = []
    for i in range(len(n1)):
        now_sentence1 = n1[i]
        now_sentence2 = n2[i]
        noun1 = okt.nouns(now_sentence1)
        noun2 = okt.nouns(now_sentence2)
        # 두 문장에서 공통된 명사를 추출
        compare = set(noun1) & set(noun2)
        for com in compare:
            # 길이가 2이상인지(잘못 고를 수 있음), wordnet에 있는지 확인
            if len(com) > 1 and com in wordnet and len(wordnet[com]) >= 2:
                sym_list = wordnet[com][1:]
                for sym in sym_list:
                    s1 = okt.pos(now_sentence1)
                    s2 = okt.pos(now_sentence2)
                    new_sentence.append([dt['id'][i], dt['source'][i], make_sentence(s1, com, sym) ,make_sentence(s2, com, sym), dt['label'][i], dt['binary-label'][i]])
                    print([dt['id'][i], dt['source'][i], make_sentence(s1, com, sym) ,make_sentence(s2, com, sym), dt['label'][i], dt['binary-label'][i]])
    return new_sentence


if __name__ == "__main__":
    augment("./data/train.csv", "./data/augment.csv")