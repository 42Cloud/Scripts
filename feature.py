import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve,auc
from sklearn.feature_extraction.text import CountVectorizer

from scipy import sparse
import os, sys


from sklearn.datasets import make_classification
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.pipeline import make_pipeline



np.random.seed(10)

def cal_score(fpr, tpr):
    #score=0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]] 
    score=0.4*tpr[np.where(fpr<=0.001)[0][-1]]+0.3*tpr[np.where(fpr<=0.005)[0][-1]]+0.3*tpr[np.where(fpr<=0.01)[0][-1]] 
    return score

online = True

'''
split data for train and test
'''


'''
data read
'''
if online:
    df_train = pd.read_csv('../data/train.csv')
    print('train read')
    df_test = pd.read_csv('../data/test_a.csv')
    print('test read')
    df_test['label'] = 2
else:
    df_train = pd.read_csv('../base_data/train.csv')
    print('train read')
    df_test = pd.read_csv('../base_data/test.csv')
    print('test read')

print(df_train.info())
print(df_test.info())
df_train['T'] = 0
df_test['T'] = 1


df_train = df_train.append(df_test)
print('append')
print(df_train.info())

if online:
    pass
else:    
    df_val = pd.read_csv('../base_data/validation.csv')
    print('val read')
    df_val['T'] = 2
    df_train = df_train.append(df_val)
    print('append val')
    print(df_train.info())
    
'''
Polynomial Features
'''

'''
data preprocessing
'''
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print('missing data')
print(missing_data.head())

df_train.drop((missing_data[missing_data['Percent']>0.25]).index, 1, inplace=True)
df_train.drop(df_train.loc[df_train['label'] == -1].index, inplace=True)
# df_train.dropna(inplace=True)
print('data preprocessing')
print(df_train.info())
print(df_train.head())
'''
categorical data
'''
dog = ['date', 'f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15',
 'f16', 'f17', 'f18', 'f19']

'''
normalization : continuous data
categorical : discrete data
'''
for col in df_train.columns:
    if col in dog:
        df_train[col] = pd.Categorical(df_train[col])
    elif col in ['label', 'T', 'id']:
        pass
    else:
        null_index = df_train[col].isnull()
        df_train.loc[~null_index, [col]] = MinMaxScaler().fit_transform(df_train.loc[~null_index, [col]])
print('cat dog')
print(df_train.info())
print(df_train.head())
'''
missing data
'''
cat = ['f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15',
 'f16', 'f17', 'f18', 'f19']

for col in df_train.columns:
    if col in cat:
        df_train[col].fillna(df_train[col].value_counts().index[0], inplace=True)
    elif col in ['date', 'label', 'T', 'id']:
        pass
    else:
        df_train[col].fillna(df_train[col].mean(), inplace=True)
        # df_train[col].fillna(df_train[col].median(), inplace=True)
print('missing data')
print(df_train.head())
print(df_train.info(verbose=True))

#float_cols = [c for c in df_train if df_train[c].dtype == 'float64']
#float32_cols = {c: np.float32 for c in float_cols}
#for col in float32_cols:
    #df_train[col] = df_train[col].astype('float32')
#print(df_train.head())
#print(df_train.info(verbose=True))
'''
dummies or one-hot
'''
for col in df_train.columns:
    if col in dog:
        dummies = pd.get_dummies(df_train[col], prefix=col)
        df_train[dummies.columns] = dummies
        df_train.drop(col, axis=1, inplace=True)
print('dummies')
print(df_train.head())
print(df_train.info(verbose=True))
'''
split train and test
'''
 
X_train = df_train.loc[df_train['T'] == 0]
X_test = df_train.loc[df_train['T'] == 1]

y_train = X_train.pop('label')
y_test = X_test.pop('label')

X_train.drop(['id'], 1, inplace=True)
id_test = X_test.pop('id')

X_train.drop(['T'], 1, inplace=True)
X_test.drop(['T'], 1, inplace=True)

if online:
    pass 
else:
    X_val = df_train.loc[df_train['T'] == 2]
    y_val = X_val.pop('label')
    X_val.drop(['id'], 1, inplace=True)
    X_val.drop(['T'], 1, inplace=True)
    
'''
check for data format
'''
print(X_train.head())
print(X_test.head())
print(X_train.info(verbose=True))
print(X_test.info(verbose=True))
'''
models
'''

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict_proba(X_test)[:, 1]
if online:
    res = pd.DataFrame({'id': id_test, 'score': y_pred_lr})
    print(res.head())
    res.to_csv('score.csv', index=False, float_format='%.6f', encoding='utf-8')
else:
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
    roc_auc = auc(fpr_lr, tpr_lr)
    print('lr')
    print(cal_score(fpr_lr, tpr_lr))

#     y_pred_lr_val = lr.predict_proba(X_val)[:, 1]
#     fpr_lr_val, tpr_lr_val, _ = roc_curve(y_val, y_pred_lr_val)
#     roc_auc_val = auc(fpr_lr_val, tpr_lr_val)
#     print([cal_score(fpr_lr_val, tpr_lr_val), roc_auc_val])
# 
#     plt.figure()
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr_lr, tpr_lr, label='LR')
#     
#     plt.plot(fpr_lr_val, tpr_lr_val, label='LR_val')
#     
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.title('ROC curve')
#     plt.legend(loc='best')
#     plt.savefig('figure1.png')
#     # plt.close()
#     plt.show()

    n_estimator = 100
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

    # grd = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100)
    # grd_enc = OneHotEncoder()
    # grd_lm = LogisticRegression()
    # grd.fit(X_train, y_train)
    # grd_enc.fit(grd.apply(X_train)[:, :, 0])
    # grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
    # features = X_train.columns
    # print(sorted(zip(map(lambda x: round(x, 4), grd.feature_importances_), features), reverse=True))     
     
    # Unsupervised transformation based on totally random trees
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
        random_state=0)
     
    rt_lm = LogisticRegression()
    pipeline = make_pipeline(rt, rt_lm)
    pipeline.fit(X_train, y_train)
    y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)
    print('rt_lm')
    print(cal_score(fpr_rt_lm, tpr_rt_lm))
     
    # Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
    rf_enc = OneHotEncoder()
    rf_lm = LogisticRegression()
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
     
    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
    print('rf_lm')
    print(cal_score(fpr_rf_lm, tpr_rf_lm))
     
    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression()
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
     
    y_pred_grd_lm = grd_lm.predict_proba(
        grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
    fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)
    print('grd_lm')
    print(cal_score(fpr_grd_lm, tpr_grd_lm))
     
     
    # The gradient boosted model by itself
    y_pred_grd = grd.predict_proba(X_test)[:, 1]
    fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
    print('grd')
    print(cal_score(fpr_grd, tpr_grd))
     
     
    # The random forest model by itself
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    print('rf')
    print(cal_score(fpr_rf, tpr_rf))
    
    # combine grd grd+lm rf+lm lm
     
    # plt.figure(figsize=(38, 28), num=1)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.savefig('figure1.png')
    plt.show()
     
    # plt.figure(figsize=(38, 28), num=2)
    plt.figure(2)
    plt.xlim(0, 0.02)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    # plt.savefig('figure2.png')
    plt.show()
