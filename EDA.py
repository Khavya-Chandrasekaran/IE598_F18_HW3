#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
df=pd.read_csv('K:\Fall 2019\MLF\Assignments\HW3\HY_Universe_corporate bond.csv')
data= df.iloc[0:,20:-1]

#2.1 Number of rows and columns
row=len(df.index)
col=len(df.columns)
print("The number of rows are "+str(row))
print("The number of columns are "+str(col))

#2.2 Col#, Number, strings, Other
type = [0]*3
colCounts = []
for i in range(col):
    for j in range(row):
        try:
            a = float(i)
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(j[i]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0]*3
print("Col#" + '\t' + "Number" + '\t' + "Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    print(str(iCol) + '\t' + str(types[0]) + '\t' + str(types[1]) + '\t' + str(types[2]) + "\n")
    iCol += 1

#2.3 Percentiles
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(data, i*(100)/ntiles))
print("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
print(" \n")
#run again with 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(data, i*(100)/ntiles))
print("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
print(" \n")

#2.4 Probability plot
from pylab import *
col_random=df["weekly_median_ntrades"]
stats.probplot(col_random, dist='norm', plot=plt)
plt.show()

#2.5 Statistics Summary
stat=df.describe()
print(stat)

#2.6 Plot
df=pd.read_csv('K:\Fall 2019\MLF\Assignments\HW3\HY_Universe_corporate bond.csv',header=None)
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
for i in range(300):
    if df.iat[i,21]==1:
        pcolor='red'
    else:
        pcolor='blue'
    data_row=df.iloc[i,0:21]
plt.pcolor(data_row)
plt.xlabel("Attribute Index")
plt.ylabel("Attribute Values")
plt.show()

#2.7 Cross-plotting of attributes
data_col15 = df.iloc[:,9]
data_col1 = df.iloc[:,36]
plt.scatter(data_col15, data_col1)
plt.xlabel("Coupon")
plt.ylabel("Weekly_median_ntrades")
plt.show()

#2.8 Correlation between Classification Target and Real Attributes
from random import uniform
target=df.iloc[0:2720,36]
data_col=df.iloc[0:2720,15]
plt.scatter(data_col, target)
plt.xlabel("Attribute- Liquidity Score")
plt.ylabel("Target Value- Weekly_median_ntrades")
plt.show()

#2.9 Pearson’s Correlation Calculation
from math import sqrt
import pandas as pd
from pandas import DataFrame
data_col2 = df.iloc[1:2720,1]
data_col3 = df.iloc[1:2720,2]
data_col21 = df.iloc[1:2720,20]
mean2 = 0.0; mean3 = 0.0; mean21 = 0.0
numElt = len(data_col2)
for i in range(numElt):
    mean2 += data_col2[i]/numElt
    mean3 += data_col3[i]/numElt
    mean21 += data_col21[i]/numElt
    var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (data_col2[i] - mean2) * (data_col2[i] - mean2)/numElt
    var3 += (data_col3[i] - mean3) * (data_col3[i] - mean3)/numElt
    var21 += (data_col21[i] - mean21) * (data_col21[i] - mean21)/numElt
    corr23 = 0.0; corr221 = 0.0
for i in range(numElt):
    corr23 += (data_col2[i] - mean2) * (data_col3[i] - mean3) / (sqrt(var2*var3) * numElt)
    corr221 += (data_col2[i] - mean2) * (data_col21[i] - mean21) / (sqrt(var2*var21) * numElt)
    print("Correlation between attribute 2 and 3 \n")
print(corr23)
print(" \n")
print("Correlation between attribute 2 and 21 \n")
print(corr221)
print(" \n")

#2.10 Presenting Attribute Correlations Visually

from pandas import DataFrame
corMat = DataFrame(df.corr())
#visualize correlations using heatmap
plt.pcolor(corMat)
plt.show()

print("My name is Khavya Chandrasekaran")
print("My NetID is: khavyac2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[288]:





# In[ ]:




