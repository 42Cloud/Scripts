import pandas as pd
import dask.dataframe as dd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve,auc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

df_test=pd.read_csv('../data/test_a.csv')
print('df_test read')
df_test.info(verbose=True, null_counts=True)
df_test.describe().to_csv('../analysis/test_describe.csv')
total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.to_csv('../analysis/test_missing_data.csv')
# df_test.drop((missing_data[missing_data['Percent']>0.25]).index, 1, inplace=True)
# print((missing_data[missing_data['Percent']>0.25]).index)
# print('drop 0.25')
# print(df_test.shape)
# l=[]
# l = (missing_data[missing_data['Percent']<0.001]).index
# print(l)
# for i in l:
    # df_test.drop((df_test.loc[(df_test.loc[:, i]).isnull()]).index, inplace=True)
# print('drop 0.001')
# #df_test.to_csv('../analysis/test_df_test.csv')
# print(df_test.shape)

# # df_test=pd.read_csv('../analysis/test_df_test.csv')
# df_test.drop(df_test.loc[df_test['label']==-1].index, inplace=True)
# print(df_test.shape)
# df_test.to_csv('../analysis/test_df_test.csv', index=False)

# df_test=pd.read_csv('../analysis/test_df_test.csv')
# df_test.drop('id', 1, inplace=True)
# print(df_test.info(verbose=True, null_counts=True))
# cat=['f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']
# for col in df_test.columns:
    # if col in cat:
        # df_test[col].fillna(round(df_test[col].mean()), inplace=True)
    # else:
        # df_test[col].fillna(df_test[col].mean(), inplace=True)
# print('fill nan')
# print(df_test.head())
# for col in df_test.columns:
    # df_test[col] = pd.Categorical(df_test[col])
# print('categorical')
# print(df_test.head())
# df_test.to_csv('../analysis/test_categorical.csv', index=False)
# df_test = pd.read_csv('../analysis/test_categorical.csv')
# print('dummies')
# df_test = pd.get_dummies(df_test)
# print(df_test.head())
# print('sort_values')
# df_test.sort_values(by=['date'], inplace=True)
# print(df_test.head())
# df_test.to_csv('../analysis/test_date.csv', index=False)

# df_test = pd.read_csv('../analysis/test_date.csv')
# dog=['date', 'label', 'f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']
# for col in df_test.columns:
    # if col in dog:
        # pass
    # else:
        # ma = df_test[col].max()
        # mi = df_test[col].min()
        # df_test[col] = (df_test[col] - mi) / (ma - mi)
# df_test.to_csv('../analysis/test_normalization.csv', index=False, float_format='%.6f')