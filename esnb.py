import pandas as pd
import numpy as np

df1 = pd.read_csv('./data/output_sota.csv')
df2 = pd.read_csv('./data/output_2d.csv')

a = np.array(df1)
b = np.array(df2)
total = []

for i in range(len(a)):
    p = round(a[i][1] + b[i][1]/2, 2)
    total.append(p)

print(len(total))
print(a[:,0])
df3 = pd.DataFrame({'id':a[:,0],
                             'target':total})

df3.to_csv('./data/esnb.csv')


