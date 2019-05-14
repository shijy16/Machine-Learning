import pandas as pd
import numpy as np
import random
import csv
import math
np.set_printoptions(suppress=True)

df1 = pd.read_csv('./result.csv', sep = ',')
df2 = pd.read_csv('./result _compare.csv', sep = ',')
df1 = df1['predicted']
df2 = df2['predicted']
c = 0
f =0

for i in range(0,len(df2)):
    if 0 <= df1[i] < 0.5:
        if 0 <= df2[i] < 0.5:
            c+=1
        else:
            f+=1
    elif 0.5 < df1[i] <= 1:
        if 0.5 < df2[i] <=1:
            c+=1
        else:
            f+=1
print(c,f,float(c)/float(c+f))