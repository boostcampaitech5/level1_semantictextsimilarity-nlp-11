import numpy as np
import pandas as pd


def ensemble(*data_path_list: str) -> None:
    """
    여러 모델의 예측 결과를 모아 앙상블을 수행하는 함수
    앙상블 기법으로는 예측 결과의 평균값을 계산하는 방식을 사용
        Args:
            *data_path_list (str): 앙상블을 수행할 각 모델의 경로 (2개 이상)
        Returns:
            None
    """
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
