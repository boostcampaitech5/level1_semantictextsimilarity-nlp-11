import pandas as pd
import numpy as np

df1 = pd.read_csv('./2023-04-16_model_Dis.csv')
df2 = pd.read_csv('./2023-04-16_model_mdeberta.csv')
df3 = pd.read_csv('./danin.csv')
a = np.array(df1)
b = np.array(df2)
c = np.array(df3)
total = []

for i in range(len(a)):
    p = (a[i][1] + b[i][1]+c[i][1])/3
    total.append(p)

print(len(total))
print(a[:,0])
df3 = pd.DataFrame({'id':a[:,0],
                             'target':total})

df3.to_csv('./data/기도메타.csv')


