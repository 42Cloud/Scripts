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

snake = ['f296',  'f297',  'f288',  'f284',  'f285',  'f286',  'f287',  'f282',  'f289',  'f290',  'f291',  'f292',  'f293',  'f294',  'f295',  'f281',  'f280',  'f279',  'f278',  'f283',  'f22',  'f23',  'f21',  'f20']
'''
    data info
'''
#df_train=pd.read_csv('../data/train.csv')
#print('train read')
# #train.info(verbose=True, null_counts=True)
# # train.groupby('label').count().to_csv('../data/count.csv')
# # train.head(5).to_csv('../data/head.csv')
# #train.describe().to_csv('../data/describe.csv')
# #train.corr().to_csv('../analysis/correlation.csv')
'''
    data visualization
'''

#df_train = pd.read_csv('../analysis/date.csv')
# plt.figure(figsize=(25, 10))
# plt.scatter(range(df_train.shape[0]), df_train.label)
# plt.xlabel('timeline', fontsize=12)
# plt.ylabel('label', fontsize=12)
# plt.show()
# cat=['f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']
# for col in df_train.columns:
    # if col in cat:
        # print(col)
        # sns.barplot(df_train[col].dropna() ,df_train['label'])
        # plt.savefig(col+'.png')
        # plt.close()
        # # print(col)
        # # sns.distplot(df_train[col].dropna())
        # # plt.savefig(col+'.png')
        # # plt.close()
    # elif col in ['id', 'date', 'label']:
        # print(col)
        # pass
    # else:   
        # print(col)
        # plt.figure(figsize=(25,10))
        # sns.barplot(df_train[col].dropna() ,df_train['label'])
        # plt.xticks(rotation=90);
        # plt.savefig(col+'.png')
        # plt.close()
        # # print(col)
        # # sns.distplot(df_train[col].dropna())
        # # # ax1 = sns.kdeplot(df_train[col][df_train['label']==0], color='r')
        # # # ax2 = sns.kdeplot(df_train[col][df_train['label']==1], color='y')
        # # # ax3 = sns.kdeplot(df_train[col][df_train['label']==-1], color='b')
        # # plt.savefig(col+'.png')
        # # plt.close()