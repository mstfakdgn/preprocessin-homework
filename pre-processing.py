import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

columns = ['Perimeter2', 'Area2', 'Smoothness2', 'Compactness2', 'Fractal_dimension3']
dataframe = pd.read_csv('./wdbc.csv', usecols=columns)
df = dataframe.copy()

y = df['Perimeter2'] 
df = df.drop('Fractal_dimension3', axis = 1)

# print('Perimeter2 mean:', df['Perimeter2'].mean())
# print('Perimeter2 mode:', df['Perimeter2'].mode())

# perimeter_Q1 = df['Perimeter2'].quantile(0.25)
# perimeter_Q3 = df['Perimeter2'].quantile(0.75)
# perimeter_IQR = perimeter_Q3 - perimeter_Q1
# print('Perimeter2 IQR:', perimeter_IQR)
# print('Perimeter2 max:', df['Perimeter2'].max())
# print('Perimeter2 min:', df['Perimeter2'].min())
# print('Perimeter2 variance:', df['Perimeter2'].var())
# print('Perimeter2 standar deviation:', df['Perimeter2'].std())
# sns.boxplot(x = df['Perimeter2'])
# plt.show()


# print('============================================================================')


# print('Area2 mean:', df['Area2'].mean())
# print('Area2 mode:', df['Area2'].mode())

# area_Q1 = df['Area2'].quantile(0.25)
# area_Q3 = df['Area2'].quantile(0.75)
# area_IQR = area_Q3 - area_Q1
# print('Area2 IQR:', area_IQR)
# print('Area2 max:', df['Area2'].max())
# print('Area2 min:', df['Area2'].min())
# print('Area2 variance:', df['Area2'].var())
# print('Area2 standar deviation:', df['Area2'].std())
# sns.boxplot(x = df['Area2'])
# plt.show()


# print('============================================================================')


# print('Smoothness2 mean:', df['Smoothness2'].mean())
# print('Smoothness2 mode:', df['Smoothness2'].mode())

# smoothness_Q1 = df['Smoothness2'].quantile(0.25)
# smoothness_Q3 = df['Smoothness2'].quantile(0.75)
# smoothness_IQR = smoothness_Q3 - smoothness_Q1
# print('Smoothness2 IQR:', smoothness_IQR)
# print('Smoothness2 max:', df['Smoothness2'].max())
# print('Smoothness2 min:', df['Smoothness2'].min())
# print('Smoothness2 variance:', df['Smoothness2'].var())
# print('Smoothness2 standar deviation:', df['Smoothness2'].std())
# sns.boxplot(x = df['Smoothness2'])
# plt.show()


# print('===========================================================================')


# print('Compactness2 mean:', df['Compactness2'].mean())
# print('Compactness2 mode:', df['Compactness2'].mode())

# compactness_Q1 = df['Compactness2'].quantile(0.25)
# compactness_Q3 = df['Compactness2'].quantile(0.75)
# compactness_IQR = compactness_Q3 - compactness_Q1
# print('Compactness2 IQR:', compactness_IQR)
# print('Compactness2 max:', df['Compactness2'].max())
# print('Compactness2 min:', df['Compactness2'].min())
# print('Compactness2 variance:', df['Compactness2'].var())
# print('Compactness2 standar deviation:', df['Compactness2'].std())


# sns.boxplot(x = df['Compactness2'])
# plt.show()

# #Outlier değerleri baskılama yöntemi ile alt ve üst sınırlara pressure yönetmini kullanarak yaslama
# # Burada bütün aykırı değerleri üst ve alt sınıra yasladığım için boxplotlarda ayrık değer olmıcak
# for i in df:
#     Q1= df[i].quantile(0.25) 
#     Q3= df[i].quantile(0.75)
#     IQR = Q3 - Q1
#     bottom_line = Q1 - 1.5*IQR
#     upper_line = Q3 + 1.5*IQR

#     isOutlier =  ((df[i] < bottom_line) | (df[i] > upper_line))
#     outliers = df[i][isOutlier]
#     outlierIndexes = df[i][isOutlier].index
#     print('===========',i,'ayrık değerler===============' )
#     print(outliers)


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
