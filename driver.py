# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# INPUT
data = pd.read_csv("cardiovascular.csv")


# NULL CHECK (none)
# sum_null = 0
# for i in range (len(data.columns)):
#     sum_null += (data.iloc[:,i].isna().sum())
#     print(len(data.iloc[:,i].unique()))
# print("Total number of null values is")
# print(sum_null)


# REMOVE DUPLICATES (1 entry removed)
data = data.drop_duplicates(ignore_index=True)


# STRUCTURE: 14 fields, 302 records


# STATISTICAL SUMMARY
# for i in range(len(data.columns)):
#     print(data.iloc[:, i].describe())


# CATEGORIES: sex(0,1), cp(0-3), fbs(0,1), restecg(0-2), exang(0,1), slope(0-2), ca(0-3), thal(1-3)
# for i in range(len(data.columns)):
#     print(data.iloc[:, i].unique())


# COUNT PLOT
# for i in range(len(data.columns)):
#     if len(data.iloc[:, i].unique()) < 6:
#         seaborn.countplot(data=data, x=data.iloc[:, i].name)
#         plt.show()


# CATEGORICAL VS TARGET
# for col in data.columns:
#     length = len(data[col].unique())
#     if length < 6:
#         x_axis = data[col].unique()
#         no_cvd = []
#         cvd = []

#         for i in range(length):
#             no_cvd.append(len(data[(data[col]==x_axis[i]) & (data['target']==0)]))
#             cvd.append(len(data[(data[col]==x_axis[i]) & (data['target']==1)]))
        
#         fig, ax = plt.subplots()
#         no_cvd = np.array(no_cvd)
#         cvd = np.array(cvd)
#         ax.bar(x_axis, cvd, label='CVD')
#         ax.bar(x_axis, no_cvd, label='No CVD', bottom=cvd)
#         ax.set_ylabel(col)
#         plt.show()


# AGE VS TARGET
# age_sorted_data = data.sort_values(by=['age'], ascending=True, ignore_index=True)

# low = []
# high = []
# curr_age = 0
# age_low_count = 0
# age_high_count = 0

# for i in range(len(age_sorted_data)):
#     if age_sorted_data['age'][i] != curr_age:
#         low.append(age_low_count)
#         high.append(age_high_count)
#         curr_age = age_sorted_data['age'][i]
#     if age_sorted_data['target'][i] == 1:
#         age_low_count += 1
#     age_high_count += 1
# low.append(age_low_count)
# high.append(age_high_count)

# low = np.array(low)
# high = np.array(high)

# plt.plot(age_sorted_data['age'].unique(), low[1:], label='Patients with CVD')
# plt.plot(age_sorted_data['age'].unique(), high[1:], label='Total patients')
# plt.show()

# for i in range (25, 70, 5):
#     low_bound = i
#     high_bound = i + 5
#     print(data[(low_bound <= data['age']) & (data['age'] < high_bound)].describe())


# BPS VS TARGET

# data0 = data[(data['target'] == 0)]
# data1 = data[(data['target'] == 1)]
# print (data0['trestbps'].describe())
# print (data1['trestbps'].describe())
# print (data['trestbps'].describe())

# dataLOWBPS = data[(data['trestbps'] < 105)]
# dataHIGHBPS = data[(data['trestbps'] > 155)]

# print(dataLOWBPS.describe())
# print(dataHIGHBPS.describe())
# print(data.describe())


# CHOLESTROL VS TARGET
# for i in range (200, 300, 10):
#     low_bound = i
#     high_bound = i + 10
#     print(data[(low_bound <= data['chol']) & (data['chol'] < high_bound)].describe())


# OLDPEAK VS TARGET
# for i in range (200, 300, 10):
#     low_bound = i
#     high_bound = i + 10
#     print(data[(low_bound <= data['chol']) & (data['chol'] < high_bound)].describe())


# MODEL
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', 1), data['target'], test_size = .2, random_state=0)
model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)
pred = model.predict(X_test)
confusion_matrix = confusion_matrix(y_test, pred)
print(confusion_matrix)