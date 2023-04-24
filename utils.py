from scipy.stats import pearsonr
import os
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np

def compute_pearson_correlation(pred):
    preds = pred.predictions.flatten()
    labels = pred.label_ids.flatten()
    return {"pearson_correlation": pearsonr(preds, labels)[0]}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True