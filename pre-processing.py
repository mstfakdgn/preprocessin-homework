import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

columns = ['Perimeter2', 'Area2', 'Smoothness2', 'Compactness2', 'Class']
dataframe = pd.read_csv('./wdbc.csv', usecols=columns)
df = dataframe.copy()

y = df['Class']
df = df.drop('Class', axis = 1)

for i in df:
    print('=================================',i,'==========================================')
    print(i,' mean:', df[i].mean())
    print(i,' mode:', df[i].mode())

    perimeter_Q1 = df[i].quantile(0.25)
    perimeter_Q3 = df[i].quantile(0.75)
    perimeter_IQR = perimeter_Q3 - perimeter_Q1
    print(i,' IQR:', perimeter_IQR)
    print(i,' Q1:', perimeter_Q1)
    print(i,' Q3:', perimeter_Q3)
    print(i,' max:', df[i].max())
    print(i,' min:', df[i].min())
    print(i,' variance:', df[i].var())
    print(i,' standar deviation:', df[i].std())
    sns.boxplot(x = df[i]).set_title(i+' Boxplot')
    plt.show()    

#Outlier değerleri baskılama yöntemi ile üst ve alt sınırlara yaslayarak etkilerini azaltacağız. (Pressure)
for i in df:
    Q1= df[i].quantile(0.25) 
    Q3= df[i].quantile(0.75)
    IQR = Q3 - Q1
    bottom_line = Q1 - 1.5*IQR
    upper_line = Q3 + 1.5*IQR

    isOutlier =  ((df[i] < bottom_line) | (df[i] > upper_line))
    outliers = df[i][isOutlier]
    outlierIndexes = df[i][isOutlier].index
    print('===========',i,'ayrık değerler===============' )
    print(outliers)
    df[i][df[i] < bottom_line] = bottom_line
    df[i][df[i] > upper_line] = upper_line


#0-1 min-max normalization
numpy_array = df.values
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(numpy_array)
df_01_normalized = pd.DataFrame(scaled, columns=df.columns)
print('=========== Normalized min/max==================')
print(df_01_normalized)

#z-score normalizasyon
std_scaler = StandardScaler()
df_zscore_std = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
print('=========== Normalized zscore==================')
print(df_zscore_std)


#n equal-width discretization
def equiwidth(arr1, m, frame): 
    a = len(arr1) 
    w = (max(arr1) - min(arr1)) / m
    min1 = min(arr1) 
    arr = [] 
    for i in range(0, m + 1): 
        arr = arr + [min1 + w * i] 
    arri=[] 
      
    for i in range(0, m): 
        temp = [] 
        for j in arr1: 
            if j >= arr[i] and j <= arr[i+1]: 
                temp += [j] 
        arri += [temp] 
    print(frame,':ayrılması\n',arri) 
    plt.hist(arri, alpha=0.5)
    plt.xlabel(frame)
    plt.title(frame +" frequency histogram") 
    plt.show()

for i in df:
    equiwidth(df[i], 3, i)


#Bilgi Kazancı hesaplaması
