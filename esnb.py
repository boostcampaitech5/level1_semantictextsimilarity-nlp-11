import os
import pandas as pd
import numpy as np

def ensemble(path: str):
    file_path = os.listdir(path)
    output_files = [file for file in file_path if file.endswith('.csv')]
    output_list = []

    for output in output_files:
        df = pd.read_csv(os.path.join(path, output))
        output_list.append(np.array(df))
    
    ensb_result = []
    for i in range(len(output_list[0])):
        average = sum([output[i][1] / len(output_list) for output in output_list])
        ensb_result.append(average)
    
    e
    return ensb_result



# df1 = pd.read_csv('./esnb_indegrient/xlm-roberta-large_consine_9e-6.csv')
# df2 = pd.read_csv('./esnb_indegrient/kykim.csv')
# df3 = pd.read_csv('./esnb_indegrient/sin_sonlpy.csv')

# a = np.array(df1)
# b = np.array(df2)
# c = np.array(df3)

# total = []

# for i in range(len(b)):
#     p = (a[i][1]+b[i][1]+c[i][1])/
#     total.append(p)

# print(len(total))
# print(b[:,0])
# df3 = pd.DataFrame({'id':b[:,0],
#                              'target':total})

# df3.to_csv('./ESNB_output/xlm-kykim_sonlpy_sin__04_20_final.csv')


