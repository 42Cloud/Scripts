import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from scipy import sparse
import lightgbm as lgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import PolynomialFeatures


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,GradientBoostingClassifier)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit


from sklearn.metrics import roc_curve,auc
from sklearn.metrics import make_scorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import make_classification

from sklearn.pipeline import make_pipeline


import os, sys
import gc
import datetime


np.random.seed(10)

def cal_score(fpr, tpr):
    #score=0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]] 
    score=0.4*tpr[np.where(fpr<=0.001)[0][-1]]+0.3*tpr[np.where(fpr<=0.005)[0][-1]]+0.3*tpr[np.where(fpr<=0.01)[0][-1]] 
    return score


online = False 
gbdt = False
nn = False
optimization = False 
cross_val = False
save_mem = False
poly = False

'''
discrete data(dog and cat) and continuous data
'''
dog = ['date', 'f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']
cat = ['f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']

start = datetime.datetime.now()
'''
data read
'''
if online:
    df_train = pd.read_csv('../data/train.csv')
    print('train read')
    df_test = pd.read_csv('../data/test_a.csv')
    print('test read')
    df_test['label'] = 2
    print(df_test.info())
    print(df_train.info())
    df_train = df_train.append(df_test, ignore_index=True)
    print('append')
    print(df_train.info())
else:
    df_train = pd.read_csv('../data/train.csv')
    print('train read')
    df_train.sort_values(by=['date'], inplace=True)
    print('sort')

end = datetime.datetime.now()
print((end-start).seconds)

if save_mem:
    start = datetime.datetime.now()
    for col in df_train.columns:
        if df_train[col].dtype == 'float64':
            df_train[col] = df_train[col].astype('float32')
        elif df_train[col].dtype == 'int64':
            df_train[col] = df_train[col].astype('int32')
        else:
            pass
    print(df_train.info())
    end = datetime.datetime.now()
    print((end-start).seconds)

gc.collect()

data_preprocessing = True
data_normalization = True
data_categorical = True
data_fillna = True
discretization = True
combination = False

if gbdt:
    df_train.drop(df_train.loc[df_train['label'] == -1].index, inplace=True)
    data_preprocessing = False
    data_normalization = False
    data_categorical = False
    data_fillna = False
    discretization = False
    combination = False
if nn:
    data_preprocessing = True
    data_normalization = False
    data_categorical = False
    data_fillna = True
    discretization = False
    combination = False
# offline used for accuracy
if not online:
    df_train.drop(df_train.loc[df_train['label'] == -1].index, inplace=True)

'''
data preprocessing
'''
if data_preprocessing:
    print('data preprocessing')
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print('missing data')
    print(missing_data.head())

    df_train.drop((missing_data[missing_data['Percent']>0.25]).index, 1, inplace=True)
    
    # df_train.dropna(inplace=True)
    print(df_train.info())
    print(df_train.head())
    
    print(df_train.groupby('label').count())
    
    
'''
missing data
'''
if data_fillna:
    print('fillna')
    for col in df_train.columns:
        if col in cat:
            df_train[col].fillna(df_train[col].value_counts().index[0], inplace=True)
        elif col in ['date', 'label', 'id']:
            pass
        # elif col in zebra or col in rhinoceros:
        #     df_train[col].fillna(df_train[col].value_counts().index[0], inplace=True)
        else:
            df_train[col].fillna(df_train[col].median(), inplace=True)
            # df_train[col].fillna(df_train[col].median(), inplace=True)
    print(df_train.head())
    print(df_train.info())


'''
discretization
'''
# 'f166', 'f167', 'f168', 'f169', 'f170', 'f171', 'f172', 'f173', 'f174', 'f175', 'f176', 'f177', 'f178', 'f179', 'f180', 'f181', 'f182', 'f183', 'f184', 'f185', 'f186', 'f187', 'f188', 'f189', 'f190', 'f191', 'f192', 'f193', 'f194', 'f195', 'f196', 'f197', 'f198', 'f199', 'f200', 'f201', 'f202', 'f203' 
# 'f254', 'f255', 'f256', 'f257', 'f258' 
# 'f267', 'f268', 'f269'
# 'f272', 'f273', 'f274', 'f275', 'f276', 'f277'
if discretization:
    print('discretization')
    
    # zebra = ['f28', 'f29', 'f30', 'f31', 'f166', 'f167', 'f168', 'f169', 'f170', 'f171', 'f172', 'f173', 'f174', 'f175', 'f176', 'f177', 'f178', 'f179', 'f180', 'f181', 'f182', 'f183', 'f184', 'f185', 'f186', 'f187', 'f188', 'f189', 'f190', 'f191', 'f192', 'f193', 'f194', 'f195', 'f196', 'f197', 'f198', 'f199', 'f200', 'f201', 'f202', 'f203' 
    # , 'f254', 'f255', 'f256', 'f257', 'f258' 
    # , 'f267', 'f268', 'f269'
    # , 'f272', 'f273', 'f274', 'f275', 'f276', 'f277']
    # df_train.loc[:, ['f5']] = df_train.loc[:, ['f5']].apply(lambda x: np.round(x / 10000))
    # print(df_train['f5'].head())
    # df_train.loc[:, zebra] = df_train.loc[:, zebra].apply(lambda x: np.round(x))
    # print(df_train[zebra].head())
    # zebra.append('f5')
    
    zebra = []
    rhinoceros = []  

    for col in df_train.columns:
        if col not in zebra and col not in dog and col not in ['label', 'id']:
            # df_train[col] = pd.cut(df_train[col], 10, labels=range(10))
            df_train[col] = pd.qcut(df_train[col], 10, labels=False, duplicates='drop')
            df_train[col] = df_train[col].astype('uint8')
            rhinoceros.append(col)
    print(df_train.info())
    print(df_train.head())
else:
    zebra = []
    rhinoceros = []


'''
Features combination
'''
if combination:
    print('combination')
    if poly:
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        output_array = poly.fit_transform(df_train.loc[:, 'f28':])
        df_output = pd.DataFrame(output_array, columns = poly.get_feature_names(df_train.columns['f28':]))
        print(df_output.info())
        print(df_output.head())
        sys.exit(0)

    donkey = []
    cat_zebra = False
    cat_rhinoceros = False
    date_cat = False
    f5_cat = False
    catD = True

    if cat_zebra:
        for col1 in cat:
            for col2 in zebra:
                df_train[col1+col2] = df_train[col1] + df_train[col2]* 10
                df_train[col1+col2] = df_train[col1+col2].astype('uint16')
                donkey.append(col1+col2)

    if cat_rhinoceros:
        for col1 in cat:
            for col2 in rhinoceros:
                df_train[col1+col2] = df_train[col1] + df_train[col2]* 10
                df_train[col1+col2] = df_train[col1+col2].astype('uint16')
                donkey.append(col1+col2)

    if date_cat:
        for col in cat:
            df_train['date'+col] = df_train['date'] * 10 + df_train[col]
            donkey.append('date'+col)

    if f5_cat:
        for col in cat:
            df_train['f5'+col] = df_train['f5'] * 10 + df_train[col]
            donkey.append('f5'+col)

    if catD:
        for col in cat:
            if col != 'f1':
                df_train['f1'+col] = df_train[col] * 10 + df_train['f1']
                donkey.append('f1'+col) 

    print(df_train.info())
    print(df_train.head())

    gc.collect()
else:
    donkey = []
'''
normalization : continuous data
categorical : discrete data
'''
if data_normalization and data_categorical:
    print('normalization and categorical')
    for col in df_train.columns:
        if col in dog:
            df_train[col] = pd.Categorical(df_train[col])
        elif col in ['label', 'id']:
            pass
        elif col in zebra or col in rhinoceros or col in donkey:
            df_train[col] = pd.Categorical(df_train[col])
        else:
            null_index = df_train[col].isnull()
            df_train.loc[~null_index, [col]] = MinMaxScaler().fit_transform(df_train.loc[~null_index, [col]])
    print(df_train.info())
    print(df_train.head())


'''
dummies or one-hot
'''
if data_categorical:
    print('dummies')
    for col in df_train.columns:
        if col in dog:
            dummies = pd.get_dummies(df_train[col], prefix=col, dummy_na=False)
            df_train[dummies.columns] = dummies
            df_train.drop(col, axis=1, inplace=True)
        elif col in zebra or col in rhinoceros or col in donkey:
            dummies = pd.get_dummies(df_train[col], prefix=col, dummy_na=False)
            df_train[dummies.columns] = dummies
            df_train.drop(col, axis=1, inplace=True)
        else:
            pass
    print(df_train.head())
    print(df_train.info())

# online used for score
if online:
    df_train.drop(df_train.loc[df_train['label'] == -1].index, inplace=True)
gc.collect()

'''
average score
'''
    
def ali_score(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred[:,1])
    roc_auc = auc(fpr, tpr)
    # score=0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]] 
    # score=0.4*tpr[np.where(fpr<=0.001)[0][-1]]+0.3*tpr[np.where(fpr<=0.005)[0][-1]]+0.3*tpr[np.where(fpr<=0.01)[0][-1]] 
    return roc_auc

if cross_val or optimization:
    y_train = df_train.pop('label')
    df_train.drop(['id'], 1, inplace=True)
    ali_scorer = make_scorer(ali_score, greater_is_better=True, needs_proba=True)
    tscv = TimeSeriesSplit(n_splits=3)

'''
params
'''    
model_lr = True
model_rt_lm = False   
model_rf = False
model_grd = False
model_gbm = False
model_bayes = False
model_knn = False
model_dt = False
model_svm = False
model_nn = False
 
if cross_val:
    print('cross validation')
    if model_lr:
        print('lr')
        lr = LogisticRegression(penalty='l1', C=1.0, n_jobs=1)
        score = cross_val_score(lr, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=1)
        print(score)
        print(score.mean())

    n_estimator = 100

    if model_rt_lm:
        print('rt_lm')
        # Unsupervised transformation based on totally random trees
        rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator, random_state=0, n_jobs=1)
        rt_lm = LogisticRegression(n_jobs=1)
        pipeline = make_pipeline(rt, rt_lm)
        score = cross_val_score(pipeline, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        print(score)
        print(score.mean())

    if model_rf:
        print('rf')
        rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
        score = cross_val_score(rf, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        print(score)
        print(score.mean())

    if model_grd:
        print('grd')
        grd = GradientBoostingClassifier(n_estimators=n_estimator)
        score = cross_val_score(grd, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        print(score)
        print(score.mean())

    if model_gbm:
        print('gbm')
        gbm = lgb.LGBMClassifier()
        score = cross_val_score(gbm, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        print(score)
        print(score.mean())
    
    if model_bayes:
        print('mnb')
        mnb = MultinomialNB()
        score = cross_val_score(mnb, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        print(score)
        print(score.mean())
    
    if model_knn:
        print('knn')
        knn = KNeighborsClassifier()
        score = cross_val_score(knn, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        print(score)
        print(score.mean())
        
    if model_dt:
        print('dt')
        dt = DecisionTreeClassifier()
        score = cross_val_score(dt, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        print(score)
        print(score.mean())
        
    if model_svm:
        print('svm')
        svm = SVC()
        score = cross_val_score(svm, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        print(score)
        print(score.mean())
        
    if model_nn:
        print('nn')
        clf = MLPClassifier()
        score = cross_val_score(clf, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        print(score)
        print(score.mean())
        
    sys.exit(0)

if optimization:
    print('optimization')
    
    def lr_cv(max_iter):
        lr = LogisticRegression(solver='sag', n_jobs=1, max_iter=max_iter)
        score = cross_val_score(lr, df_train, y_train, scoring=ali_scorer, cv=tscv, verbose=1, n_jobs=-1)
        return score.mean()
    aliBO = BayesianOptimization(lr_cv, {'max_iter': (100, 2000)})
    init_points = 5
    num_iter = 20
    aliBO.maximize(init_points=init_points, n_iter=num_iter)
    print(aliBO.res['max'])
    print(aliBO.res['all'])

    sys.exit(0)

if online:
    print('online')
    X_train = df_train.loc[df_train['label'] != 2]
    X_test = df_train.loc[df_train['label'] == 2]
    
    y_train = X_train.pop('label')
    y_test = X_test.pop('label')

    X_train.drop(['id'], 1, inplace=True)
    id_test = X_test.pop('id')
    
    '''
    check for data format
    '''
    print(X_train.head())
    print(X_test.head())
    print(X_train.info())
    print(X_test.info())
    
    # gbm = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators= 100, objective= 'binary', silent= False)
    # gbm.fit(X_train, y_train, categorical_feature=dog)
    # y_pred = gbm.predict_proba(X_test)[:,1]
    
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)[:, 1]
    
    res = pd.DataFrame({'id': id_test, 'score': y_pred})
    print(res.head())
    print(res.info(verbose=True))
    res.to_csv('score.csv', index=False, float_format='%.6f', encoding='utf-8')
    sys.exit(0)
else:
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

    model_rf_lm = False
    plot_fig = False
    '''
    split train and test
    '''
    # for i_split in [0.79, 0.80, 0.81]:
    i_split = 0.80
    j_split = 0.90
    i_split_point = int(round(df_train.shape[0]*i_split))
    j_split_point = int(round(df_train.shape[0]*j_split))
    X_train = df_train.iloc[:i_split_point, :]
    X_test = df_train.iloc[j_split_point:, :]
        
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
        params = {
            'boosting': 'gbdt',
            'metric': 'auc',
            'application': 'binary',
        }
        gbm = lgb.LGBMClassifier()
        gbm.fit(X_train, y_train)
        y_pred_gbm = gbm.predict_proba(X_test, num_iteration=gbm.best_iteration)[:,1]
        fpr_gbm, tpr_gbm, _ = roc_curve(y_test, y_pred_gbm)
        roc_auc = auc(fpr_gbm, tpr_gbm)
        print('gbm')
        print(cal_score(fpr_gbm, tpr_gbm))
        # df = pd.DataFrame(fea, columns=['feature'])
        # df['importance']=list(gbm.feature_importance())
        # df = df.sort_values(by='importance',ascending=False)
        # df.to_csv("feature_score.csv",index=False,encoding='utf-8')
        

    # res = pd.DataFrame({'id': id_test, 'score': y_pred_gbm_lm})
    # print(res.head())
    # print(res.info(verbose=True))
    # res.to_csv('score.csv', index=False, float_format='%.6f', encoding='utf-8')
    # sys.exit(0)
    if model_lr:
        lr = LogisticRegression(penalty='l1', C=1.0, n_jobs=1)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict_proba(X_test)[:, 1]
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
        
        print('lr')
        score_lr += cal_score(fpr_lr, tpr_lr)
        auc_lr += auc(fpr_lr, tpr_lr)
        print([cal_score(fpr_lr, tpr_lr), auc(fpr_lr, tpr_lr)])

    n_estimator = 100
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

    if model_rt_lm:
    
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

    if model_rf_lm:
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
        
        
        # The random forest model by itself
        y_pred_rf = rf.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
        
        print('rf')
        auc_rf += auc(fpr_rf, tpr_rf)
        score_rf += cal_score(fpr_rf, tpr_rf)
        print([cal_score(fpr_rf, tpr_rf), auc(fpr_rf, tpr_rf)])

    if model_grd_lm:
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


    if model_gbm:
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
    
    if plot_fig:
   
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
