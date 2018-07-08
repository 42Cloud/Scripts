import pandas as pd
import dask.dataframe as dd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve,auc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

'''
when missing_data>25% all delete
when missing_data<25% dropna
'''

'''
    # data info
'''
df_train=pd.read_csv('../base_data/train.csv')
print('train read')
df_test=pd.read_csv('../base_data/test.csv')
print('test read')
df_val=pd.read_csv('../base_data/validation.csv')
print('validation read')
df_train['T'] = 0
df_test['T'] = 1
df_val['T'] = 2
df_train = df_train.append(df_test)
df_train = df_train.append(df_val)
print('appended')
# ''' 
    # missing value
# '''
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# '''
    # delete missing too much
# '''
print((missing_data[missing_data['Percent']>0.25]).head())
# print((missing_data[missing_data['Percent']<0.001]).index)
df_train.drop((missing_data[missing_data['Percent']>0.25]).index, 1, inplace=True)
# drop label = -1  
df_train.drop(df_train.loc[df_train['label']==-1].index, inplace=True)
print('drop label -1')
print('drop 0.25')
print(df_train.shape)
print(df_train.info())
'''
    dropna
'''
df_train.dropna(inplace=True)
print('dropna')
print(df_train.isnull().count())
print(df_train.info())

df_train.to_csv('../baseline/train.csv', index=False)
# df_train = pd.read_csv('../baseline/train.csv')
'''
    baseline:  drop id
'''
df_train.drop('id', 1, inplace=True)
print('drop id')

'''
    normalization
'''
dog=['T', 'id', 'date', 'label', 'f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']
for col in df_train.columns:
    if col in dog:
        pass
    else:
        ma = df_train[col].max()
        mi = df_train[col].min()
        df_train[col] = (df_train[col] - mi) / (ma - mi)
print('normalization')
print(df_train.head())
df_train.to_csv('../baseline/normalization.csv', index=False, float_format='%.6f')
# df_train = pd.read_csv('../baseline/normalization.csv')
# '''
    # sparse features:date 
# '''
print('categorical')
cat=['date', 'f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']  # categorical feature
for col in df_train.columns:
    if col in cat:
        df_train[col] = pd.Categorical(df_train[col])
print(df_train.head())
print(df_train.info())

print('dummies')
df_train = pd.get_dummies(df_train)
print(df_train.head())
df_train.to_csv('../baseline/categorical.csv', index=False)


# df_train = pd.read_csv('../baseline/categorical.csv')
'''
    split train and test
'''
print('split train and test')
X_train = df_train[df_train['T'] == 0][df_train.loc[:, df_train.columns != 'label'].columns]
print(X_train.head())
print(X_train.info())
y_train = df_train[df_train['T'] == 0]['label']
X_test = df_train[df_train['T'] == 1][df_train.loc[:, df_train.columns != 'label'].columns]
y_test = df_train[df_train['T'] == 1]['label']
X_val = df_train[df_train['T'] == 2][df_train.loc[:, df_train.columns != 'label'].columns]
y_val = df_train[df_train['T'] == 2]['label']
X_train.drop('T', inplace=True, axis=1)
print(X_train.head())
print(X_train.info())
X_test.drop('T', inplace=True, axis=1)
print(X_test.head())
print(X_test.info())
X_val.drop('T', inplace=True, axis=1)

'''
    logistic regression model
    cv is not applicable because of time series
'''
print('training')
loggre = LogisticRegression(verbose=1)
loggre.fit(X_train, y_train)
# print(loggre.score(X_test, y_test))
'''
    ROC
'''
print('ROC')
predictions=loggre.predict_proba(X_test)#每一类的概率
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:
, 1])
roc_auc=auc(false_positive_rate,recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()
'''
    ROC
'''
print('ROC')
predictions=loggre.predict_proba(X_val)#每一类的概率
false_positive_rate, recall, thresholds = roc_curve(y_val, predictions[:
, 1])
roc_auc=auc(false_positive_rate,recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()