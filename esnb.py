import numpy as np
import pandas as pd


def ensemble(*data_path_list: str):
    output_list = []

    for data_path in data_path_list:
        df = pd.read_csv(data_path)
        output_list.append(np.array(df))

    esnb_result = []
    for i in range(len(output_list[0])):
        average = sum([output[i][1] / len(output_list) for output in output_list])
        esnb_result.append(average)

    esnb_dataframe = pd.DataFrame({"id": output_list[0][:, 0], "target": esnb_result})
    esnb_dataframe.to_csv("./esnb/esnb.csv", index=False)
