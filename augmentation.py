import pickle
import re

import pandas as pd
from hanspell import spell_checker
from konlpy.tag import Okt
from pykospacing import Spacing
from soynlp.normalizer import repeat_normalize
from tqdm.auto import tqdm


def under_sampling(data_path: str) -> pd.DataFrame:
    """
    label 값이 0인 데이터를 under sampling하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
        Returns:
            df_new (DataFrame): under sampling된 데이터
    """
    df = pd.read_csv(data_path)
    df_0 = df[df["label"] == 0][1000:2000].copy()
    df_new = df[df["label"] != 0].copy()
    df_new = pd.concat([df_new, df_0])
    return df_new


def swap_sentence(data_path: str) -> pd.DataFrame:
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
    df_swapped = df_swapped[df_swapped["label"] != 0]
    return df_swapped


def copy_sentence(data_path: str, index_min=250, index_max=750) -> pd.DataFrame:
    """
    sentence 1과 sentence 2에 같은 문장을 배치해 5점짜리 데이터를 생성하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
            index_min (int): 증강할 데이터에서 슬라이싱 시작  defalt = 250
            index_max (int): 증강할 데이터에서 슬라이싱 끝    defalt = 750
        Returns:
            df_copied (DataFrame): 증강된 데이터
    """
    df = pd.read_csv(data_path)
    df_copied = df[df["label"] == 0][index_min:index_max].copy()
    df_copied["sentence_1"] = df_copied["sentence_2"]
    df_copied["label"] = 5.0
    return df_copied


def concat_data(data_path: str, *dataframes: pd.DataFrame):
    """
    데이터프레임을 합쳐서 csv 파일로 저장하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
            dataframes (DataFrame): 합치려고 하는 데이터프레임
    """
    result = pd.concat(dataframes)
    result.to_csv(data_path, index=False)


def augment(source_data_path, dest_data_path):
    df_under_sampled = under_sampling(source_data_path)
    df_swapped_sentence = swap_sentence(source_data_path)
    df_copied_sentence = copy_sentence(source_data_path)
    concat_data(
        dest_data_path, df_under_sampled, df_swapped_sentence, df_copied_sentence
    )


def han_spell(text: str) -> str:
    """
    특수문자 제거 및 hanspell 맞춤법 검사
        Args :
            text (str): 교정할 sentence
        Returns :
            correct_text (str): 교정한 sentence
    """
    text = repeat_normalize(text, num_repeats=2)
    text = text.lower()
    text = re.sub("[^a-z가-힣0-9 ]", "", text)
    text = text.strip()
    correct_text = spell_checker.check(text).as_dict()["checked"]
    return correct_text


def apply_hanspell(df: pd.DataFrame) -> pd.DataFrame:
    """
    han_spell()을 데이터에 적용
        Args :
            data (DataFrame): 맞춤법을 교정할 데이터
        Returns :
            data (DataFrame): 맞춤법을 교정한 데이터
    """
    tqdm.pandas()
    df["sentence_1"] = df["sentence_1"].progress_map(han_spell)
    df["sentence_2"] = df["sentence_2"].progress_map(han_spell)
    df = df.dropna(subset=["sentence_1"])
    df = df.dropna(subset=["sentence_2"])
    return df


def check_end(noun: str) -> bool:
    """
    한글의 유니코드가 28로 나누어 떨어지면 받침이 없음을 판단
        Args :
            noun (str): 받침 유무를 판단할 명사
        Returns :
            False (bool) : 받침이 없음
            True  (bool) :  받침이 있음
    """
    if (ord(noun[-1]) - ord("가")) % 28 == 0:
        return False
    else:
        return True


def change_josa(noun: str, josa: str) -> str:
    """
    명사의 끝음절 받침 여부에 따라서 조사 교체
        Args :
            none (str): 끝음절의 받침 확인할 명사
            josa (str): 교정할 조사
        Returns :
            josa (str): 교정한 조사
    """
    if josa == "이" or josa == "가":
        return "이" if check_end(noun) else "가"
    elif josa == "은" or josa == "는":
        return "은" if check_end(noun) else "는"
    elif josa == "을" or josa == "를":
        return "을" if check_end(noun) else "를"
    elif josa == "과" or josa == "와":
        return "과" if check_end(noun) else "와"
    else:
        return josa


def make_sentence(sentence: list, compare: str, sym: str) -> str:
    """
    sentence_1, sentence_2에 모두 등장하는 명사를 교체하고 조사를 교정
        Args :
            sentence (list): 형태소 분석한 문장
            compare  (str): 문장에서 바꿀 명사
            sym      (str): 문장 삽입되는 동의어
        Returns :
            replace_sentence (str): 동의어로 교체한 문장
    """
    spacing = Spacing()
    replace_sentence = []
    check = set(["이", "가", "을", "를", "과", "와"])
    for j in range(len(sentence)):
        # 문장에서 동의어를 추가한다.
        if sentence[j][0] == compare:
            replace_sentence.append(sym)
            # 뒷말이 조사면 조사를 확인하고 바꾼다.
            if (
                j + 1 < len(sentence)
                and sentence[j + 1][1] == "Josa"
                and sentence[j + 1][0] in check
            ):
                # 바뀐 명사 마지막 받침 확인 후 조사 변경
                sentence[j + 1] = (
                    change_josa(replace_sentence[-1][0], sentence[j + 1][0]),
                    "Josa",
                )
        else:
            replace_sentence.append(sentence[j][0])

    replace_sentence = "".join(replace_sentence)
    replace_sentence = spacing(replace_sentence)
    return replace_sentence


def sr_noun_replace(data_path: str, wordnet_path: str) -> pd.DataFrame:
    """
    데이터를 맞춤법 교정 후 명사와 조사를 교체 증강
        Args :
            data_path    (str): 증강하고자 하는 데이터의 경로
            wordnet_path (str): 동의어 사전 경로
        Returns :
            sr_sentence (DataFrame): 증강된 데이터
    """
    with open(wordnet_path, "rb") as f:
        wordnet = pickle.load(f)

    data = pd.read_csv(data_path)
    okt = Okt()
    data = apply_hanspell(data)
    n1, n2 = data["sentence_1"], data["sentence_2"]
    sr_sentence = []

    for i in tqdm(range(len(n1)), desc="SR Sentece"):
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
                    sr_sentence.append(
                        [
                            data["id"][i],
                            data["source"][i],
                            make_sentence(s1, com, sym),
                            make_sentence(s2, com, sym),
                            data["label"][i],
                            data["binary-label"][i],
                        ]
                    )
    sr_sentence = pd.DataFrame(
        sr_sentence,
        columns=["id", "source", "sentence_1", "sentence_2", "label", "binary-label"],
    )
    return sr_sentence


def sr_swap_sentence(df: pd.DataFrame) -> pd.DataFrame:
    """
    sentence 1과 sentence 2의(1<= label <3) 위치를 바꾸어 증강하는 함수
        Args:
            data_path (str): 증강하고자 하는 데이터의 경로
            sr (bool): swap 범위
        Returns:
            df_swapped (DataFrame): 증강된 데이터
    """
    df_swapped = df.copy()
    df_swapped["sentence_1"] = df["sentence_2"]
    df_swapped["sentence_2"] = df["sentence_1"]
    df_swapped = df_swapped[(df_swapped["label"] >= 1) & (df_swapped["label"] < 3)]

    return df_swapped


def sr_augment(source_data_path, dest_data_path, wordnet_path):
    df_source = pd.read_csv(source_data_path)
    df_noun_replaced = sr_noun_replace(source_data_path, wordnet_path)
    df_noun_replaced = df_noun_replaced[df_noun_replaced["label"] >= 1]
    df_source_noun = pd.concat([df_source, df_noun_replaced])
    df_swapped_sentence = sr_swap_sentence(df_source_noun)
    df_copied_sentence = copy_sentence(source_data_path, index_min=250, index_max=1250)
    concat_data(dest_data_path, df_source_noun, df_swapped_sentence, df_copied_sentence)


if __name__ == "__main__":
    augment("./data/train.csv", "./data/aug_train.csv")
    sr_augment("./data/train.csv", "./data/sr_augment.csv", "./wordnet/wordnet.pickle")
