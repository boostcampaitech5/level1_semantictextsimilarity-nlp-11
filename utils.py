import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import transformers
import yaml
from scipy.stats import pearsonr


def compute_pearson_correlation(
    pred: transformers.trainer_utils.EvalPrediction,
) -> dict:
    """
    피어슨 상관 계수를 계산해주는 함수
        Args:
            pred (torch.Tensor): 모델의 예측값과 레이블을 포함한 데이터
        Returns:
            perason_correlation (dict): 입력값을 통해 계산한 피어슨 상관 계수
    """
    preds = pred.predictions.flatten()
    labels = pred.label_ids.flatten()
    perason_correlation = {"pearson_correlation": pearsonr(preds, labels)[0]}
    return perason_correlation


def seed_everything(seed: int) -> None:
    """
    모델에서 사용하는 모든 랜덤 시드를 고정해주는 함수
        Args:
            seed (int): 시드 고정에 사용할 정수값
        Returns:
            None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def load_yaml(path: str) -> dict:
    """
    모델 훈련, 예측에 사용할 yaml 파일을 불러오는 함수
        Args:
            path (str): 불러올 yaml 파일의 경로
        Returns:
            loaded_yaml (dict): 지정한 경로에서 불러온 yaml 파일 내용

    """
    with open(path, "r") as f:
        loaded_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return loaded_yaml
