import pandas as pd
import numpy as np

df = pd.read_csv('./data/train.csv')

d = np.array(df['label'])
count= 0
for i in d:
    if i == 0:
        count += 1
print(count)

#(짤라보고)

#(길이 분포 보고)(별로면)

#(문장의 길이를 고려 + 랜덤 초이스 + 시드고정 가능)(주원이형)

