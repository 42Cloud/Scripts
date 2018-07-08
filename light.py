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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import make_scorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import make_classification
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,GradientBoostingClassifier)
from sklearn.pipeline import make_pipeline

from scipy import sparse
import lightgbm as lgb
import os, sys
from bayes_opt import BayesianOptimization





np.random.seed(10)

def cal_score(fpr, tpr):
    score=0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]] 
    #score=0.4*tpr[np.where(fpr<=0.001)[0][-1]]+0.3*tpr[np.where(fpr<=0.005)[0][-1]]+0.3*tpr[np.where(fpr<=0.01)[0][-1]] 
    return score
    
def create_train_test(split_date):
    df_data = pd.read_csv('../data/train.csv')
    
    print(df_data.info())
    for col in df_data.columns:
        if df_data[col].dtype == 'float64':
            df_data[col] = df_data[col].astype('float32')
        elif df_data[col].dtype == 'int64':
            df_data[col] = df_data[col].astype('int32')
        else:
            pass
    print(df_data.info())
    
    df_train = df_data.loc[df_data['date'] < split_date]
    df_test = df_data.loc[df_data['date'] >= split_date]
    print(df_data.info())
    print(df_train.info())
    print(df_test.info())
    return df_train, df_test

online = False 
gbdt = False
optimization = True

test = False
categorical_flag = False
fillna_flag = False


'''
categorical data
'''
dog = ['date', 'f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']
cat = ['f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']
     
'''
split data for train and test
'''
if test:
    df_data = pd.read_csv('../data/train.csv')
    
    print(df_data.info())
    for col in df_data.columns:
        if df_data[col].dtype == 'float64':
            df_data[col] = df_data[col].astype('float32')
        elif df_data[col].dtype == 'int64':
            df_data[col] = df_data[col].astype('int32')
        else:
            pass
    print(df_data.info())
    
    split_date = 20171023
    df_train = df_data.loc[df_data['date'] < split_date]
    df_test = df_data.loc[df_data['date'] >= split_date]
    print(df_data.info())
    print(df_train.info())
    print(df_test.info())
    df_train.to_csv('../data/split_train'+str(split_date)+'.csv', index=False)
    df_test.to_csv('../data/split_test_a'+str(split_date)+'.csv', index=False)
    sys.exit(0)

'''
data read
'''
if online:
    df_train = pd.read_csv('../data/train.csv')
    print('train read')
    df_test = pd.read_csv('../data/test_a.csv')
    print('test read')
    df_test['label'] = 2
    
    df_train = df_train.append(df_test)
    print('append')
else:
    df_train = pd.read_csv('../data/train.csv')
    print('train read')
    df_train.sort_values(by=['date'], inplace=True)
    print('sort')

for col in df_train.columns:
    if df_train[col].dtype == 'float64':
        df_train[col] = df_train[col].astype('float32')
    elif df_train[col].dtype == 'int64':
        df_train[col] = df_train[col].astype('int32')
    else:
        pass
print(df_train.info())

    
'''
Polynomial Features
'''
if gbdt:
    df_train.drop(df_train.loc[df_train['label'] == -1].index, inplace=True)
else:
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
    normalization : continuous data
    categorical : discrete data
    '''
    for col in df_train.columns:
        if col in dog:
            df_train[col] = pd.Categorical(df_train[col])
        elif col in ['label', 'id']:
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
    

    for col in df_train.columns:
        if col in cat:
            df_train[col].fillna(df_train[col].value_counts().index[0], inplace=True)
        elif col in ['date', 'label', 'id']:
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
    
# df_train.to_csv('train_offline.csv', index=False)
# df_train = pd.read_csv('train_offline.csv')
# print(df_train.info(verbose=True))
# for col in df_train.columns:
    # if df_train[col].dtype == 'float64':
        # df_train[col] = df_train[col].astype('float32')
    # elif df_train[col].dtype == 'int64':
        # df_train[col] = df_train[col].astype('int32')
    # else:
        # pass
# print(df_train.info())
'''
average score
'''
if online:
    pass
else:
    y_train = df_train.pop('label')
    df_train.drop(['id'], 1, inplace=True)
def ali_score(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred[:,1])
    # score=0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]] 
    score=0.4*tpr[np.where(fpr<=0.001)[0][-1]]+0.3*tpr[np.where(fpr<=0.005)[0][-1]]+0.3*tpr[np.where(fpr<=0.01)[0][-1]] 
    return score
ali_scorer = make_scorer(ali_score, greater_is_better=True, needs_proba=True)
tscv = TimeSeriesSplit(n_splits=3)

lr = LogisticRegression(n_jobs=-1)
score = cross_val_score(lr, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
print('lr')
print(score.mean())

n_estimator = 500

# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator, random_state=0, n_jobs=-1)
rt_lm = LogisticRegression(n_jobs=-1)
pipeline = make_pipeline(rt, rt_lm)
score = cross_val_score(pipeline, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
print('rt_lm')
print(score.mean())

rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
score = cross_val_score(rf, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
print('rf')
print(score.mean())

grd = GradientBoostingClassifier(n_estimators=n_estimator)
score = cross_val_score(grd, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
print('grd')
print(score.mean())

gbm = lgb.LGBMClassifier()
score = cross_val_score(gbm, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
print('gbm')
print(score.mean())

if optimization:
    print('optimization')
    def lr_cv(max_iter):
        lr = LogisticRegression(solver='sag', n_jobs=-1, max_iter=max_iter)
        score = cross_val_score(lr, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        return score.mean()
    aliBO = BayesianOptimization(lr_cv, {'max_iter': (100, 2000)})
    init_points = 5
    num_iter = 20
    aliBO.maximize(init_points=init_points, n_iter=num_iter)
    print(aliBO.res['max'])
    print(aliBO.res['all'])

sys.exit(0)

score_lr = 0
auc_lr = 0
score_rt_lm = 0
auc_rt_lm = 0
score_rf_lm = 0
auc_rf_lm = 0
score_grd_lm = 0
auc_grd_lm = 0
score_grd = 0
auc_grd = 0
score_rf = 0
auc_rf = 0
score_gbm_lm = 0
auc_gbm_lm = 0
score_gbm = 0
auc_gbm = 0
'''
split train and test
'''
for i_split in [0.79, 0.80, 0.81]:
    if online:
        X_train = df_train.loc[df_train['label'] != 2]
        X_test = df_train.loc[df_train['label'] == 2]
    else:
        split_point = round(df_train.shape[0]*i_split)
        X_train = df_train.loc[:split_point, :]
        X_test = df_train.loc[split_point:, :]
        
    y_train = X_train.pop('label')
    y_test = X_test.pop('label')

    X_train.drop(['id'], 1, inplace=True)
    id_test = X_test.pop('id')

    fea = X_train.columns
        
    '''
    check for data format
    '''
    # print(X_train.head())
    # print(X_test.head())
    # print(X_train.info(verbose=True))
    # print(X_test.info(verbose=True))
    
    '''
    models
    '''
    if gbdt:
        lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=dog)
        lgb_test = lgb.Dataset(X_test, reference=lgb_train, label=y_test, categorical_feature=dog)
        params = {
            'boosting': 'gbdt',
            'metric': 'auc',
            'application': 'binary',
        }
        gbm = lgb.train(params, lgb_train, valid_sets=lgb_test)
        y_pred_gbm = gbm.predict_proba(X_test, num_iteration=gbm.best_iteration) 
        fpr_gbm, tpr_gbm, _ = roc_curve(y_test, y_pred_gbm)
        roc_auc = auc(fpr_gbm, tpr_gbm)
        print('gbm')
        print(cal_score(fpr_gbm, tpr_gbm))
        # df = pd.DataFrame(fea, columns=['feature'])
        # df['importance']=list(gbm.feature_importance())
        # df = df.sort_values(by='importance',ascending=False)
        # df.to_csv("feature_score.csv",index=False,encoding='utf-8')
        
    else:
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
        # # gbdt + lm
        # gbm = lgb.LGBMClassifier()
        # gbm_enc = OneHotEncoder()
        # gbm_lm = LogisticRegression()
        # gbm.fit(X_train, y_train)
        # gbm_enc.fit(gbm.apply(X_train)[:, :])
        # gbm_lm.fit(gbm_enc.transform(gbm.apply(X_train_lr)[:, :]), y_train_lr)
             
        # y_pred_gbm_lm = gbm_lm.predict_proba(gbm_enc.transform(gbm.apply(X_test)[:, :]))[:, 1]

        # res = pd.DataFrame({'id': id_test, 'score': y_pred_gbm_lm})
        # print(res.head())
        # print(res.info(verbose=True))
        # res.to_csv('score.csv', index=False, float_format='%.6f', encoding='utf-8')
        # sys.exit(0)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict_proba(X_test)[:, 1]
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
        
        print('lr')
        score_lr += cal_score(fpr_lr, tpr_lr)
        auc_lr += auc(fpr_lr, tpr_lr)
        print([cal_score(fpr_lr, tpr_lr), auc(fpr_lr, tpr_lr)])

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
        score_rt_lm += cal_score(fpr_rt_lm, tpr_rt_lm)
        auc_rt_lm += auc(fpr_rt_lm, tpr_rt_lm)
        print([cal_score(fpr_rt_lm, tpr_rt_lm), auc(fpr_rt_lm, tpr_rt_lm)])

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
        auc_rf_lm += auc(fpr_rf_lm, tpr_rf_lm)
        score_rf_lm += cal_score(fpr_rf_lm, tpr_rf_lm)
        print([cal_score(fpr_rf_lm, tpr_rf_lm), auc(fpr_rf_lm, tpr_rf_lm)])

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
        score_grd_lm += cal_score(fpr_grd_lm, tpr_grd_lm)
        auc_grd_lm += auc(fpr_grd_lm, tpr_grd_lm)
        print([cal_score(fpr_grd_lm, tpr_grd_lm), auc(fpr_grd_lm, tpr_grd_lm)])


        # The gradient boosted model by itself
        y_pred_grd = grd.predict_proba(X_test)[:, 1]
        fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
        
        print('grd')
        auc_grd += auc(fpr_grd, tpr_grd)
        score_grd += cal_score(fpr_grd, tpr_grd)
        print([cal_score(fpr_grd, tpr_grd), auc(fpr_grd, tpr_grd)])


        # The random forest model by itself
        y_pred_rf = rf.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
        
        print('rf')
        auc_rf += auc(fpr_rf, tpr_rf)
        score_rf += cal_score(fpr_rf, tpr_rf)
        print([cal_score(fpr_rf, tpr_rf), auc(fpr_rf, tpr_rf)])
        
        # gbdt + lm
        gbm = lgb.LGBMClassifier()
        gbm_enc = OneHotEncoder()
        gbm_lm = LogisticRegression()
        gbm.fit(X_train, y_train)
        gbm_enc.fit(gbm.apply(X_train)[:, :])
        gbm_lm.fit(gbm_enc.transform(gbm.apply(X_train_lr)[:, :]), y_train_lr)
         
        y_pred_gbm_lm = gbm_lm.predict_proba(
            gbm_enc.transform(gbm.apply(X_test)[:, :]))[:, 1]
        fpr_gbm_lm, tpr_gbm_lm, _ = roc_curve(y_test, y_pred_gbm_lm)
        
        print('gbm_lm')
        auc_gbm_lm += auc(fpr_gbm_lm, tpr_gbm_lm)
        score_gbm_lm += cal_score(fpr_gbm_lm, tpr_gbm_lm)
        print([cal_score(fpr_gbm_lm, tpr_gbm_lm), auc(fpr_gbm_lm, tpr_gbm_lm)])
        
        # gbdt
        y_pred_gbm = gbm.predict_proba(X_test)[:, 1]
        fpr_gbm, tpr_gbm, _ = roc_curve(y_test, y_pred_gbm)
        
        print('gbm')
        auc_gbm += auc(fpr_gbm, tpr_gbm)
        score_gbm += cal_score(fpr_gbm, tpr_gbm)
        print([cal_score(fpr_gbm, tpr_gbm), auc(fpr_gbm, tpr_gbm)])
        
        continue
       
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
        plt.plot(fpr_gbm, tpr_gbm, label='GBDT')
        plt.plot(fpr_gbm_lm, tpr_gbm_lm, label='GBDT + LR')
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
        plt.plot(fpr_gbm, tpr_gbm, label='GBDT')
        plt.plot(fpr_gbm_lm, tpr_gbm_lm, label='GBDT + LR')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (zoomed in at top left)')
        plt.legend(loc='best')
        # plt.savefig('figure2.png')
        plt.show()

'''
average score
'''
n_average = 3
print('lr')
print('%f-%f' % (score_lr/n_average, auc_lr/n_average))
print('rt_lm')
print('%f-%f' % (score_rt_lm/n_average, auc_rt_lm/n_average))
print('rf_lm')
print('%f-%f' % (score_rf_lm/n_average, auc_rf_lm/n_average))
print('grd_lm')
print('%f-%f' % (score_grd_lm/n_average, auc_grd_lm/n_average))
print('grd')
print('%f-%f' % (score_grd/n_average, auc_grd/n_average))
print('rf')
print('%f-%f' % (score_rf/n_average, auc_rf/n_average))
print('gbm_lm')
print('%f-%f' % (score_gbm_lm/n_average, auc_gbm_lm/n_average))
print('gbm')
print('%f-%f' % (score_gbm/n_average, auc_gbm/n_average))
    