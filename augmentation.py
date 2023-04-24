import pandas as pd

def under_sampling(data_path):
    df = pd.read_csv(data_path)
    df_0 = df[df['label']==0][1000:2000].copy()
    df_new = df[df['label']!=0].copy()
    df_new = pd.concat([df_new, df_0])
    return df_new

def swap_sentence(data_path):
    df = pd.read_csv(data_path)
    df_switched = df.copy()
    df_switched["sentence_1"] = df["sentence_2"]
    df_switched["sentence_2"] = df["sentence_1"]
    df_switched = df_switched[df_switched['label'] != 0]
    return df_switched

def copy_sentence(data_path):
    df = pd.read_csv(data_path)
    copied_df = df[df['label']==0][250:750].copy()
    copied_df['sentence_1'] = copied_df['sentence_2']
    copied_df['label'] = 5.0
    copied_df.reset_index(inplace=True)
    return copied_df

def concat_data(data_path, *dataframes):
    result = pd.concat(dataframes)
    result.to_csv(data_path, index = False)

def augment(source_data_path, dest_data_path):
    under_sampled = under_sampling(source_data_path)
    swapped_sentence = swap_sentence(source_data_path)
    copied_sentence = copy_sentence(source_data_path)
    concat_data(dest_data_path, under_sampled, swapped_sentence, copied_sentence)

if __name__ == "__main__":
    augment("./data/train.csv", "./data/augment.csv")