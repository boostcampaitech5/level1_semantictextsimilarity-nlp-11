import pandas as pd

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

if __name__ == "__main__":
    augment("./data/train.csv", "./data/augment.csv")