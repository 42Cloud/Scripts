import lightgbm as lgb
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
#from model.score import eval_metric
#highest offline 0.3154168252759802,oneline 0.3791, 7_2


DELETE_INDEX = ['f160', 'f297', 'f288', 'f289', 'f115', 'f284', 'f285', 'f286', 'f287',
                'f280', 'f281', 'f282', 'f283', 'f41', 'f40', 'f43', 'f42', 'f45', 'f44',
                'f47', 'f46', 'f49', 'f48', 'f292', 'f291', 'f290', 'f118', 'f119', 'f114',
                'f296', 'f295', 'f117', 'f293', 'f111', 'f112', 'f113', 'f56', 'f57', 'f54',
                'f55', 'f52', 'f53', 'f50', 'f51', 'f58', 'f59', 'f116', 'f109', 'f108', 'f107',
                'f106', 'f105', 'f104', 'f103', 'f102', 'f101', 'f100', 'f294', 'f23', 'f22',
                'f21', 'f20', 'f27', 'f26', 'f25', 'f24', 'f29', 'f28', 'f110', 'f138',
                'f139', 'f132', 'f133', 'f130', 'f131', 'f136', 'f137', 'f134', 'f135',
                'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39',
                'f125', 'f124', 'f127', 'f126', 'f121', 'f120', 'f123', 'f122', 'f129',
                'f128', 'f5', 'f89', 'f88', 'f85', 'f84', 'f87', 'f86', 'f81', 'f80', 'f83',
                'f82', 'f150', 'f151', 'f152', 'f153', 'f154', 'f155', 'f156', 'f157', 'f158',
                'f159', 'f92', 'f93', 'f90', 'f91', 'f96', 'f97', 'f94', 'f95', 'f98', 'f99', 'f143',
                'f142', 'f141', 'f140', 'f147', 'f146', 'f145', 'f144', 'f149', 'f148', 'f69', 'f68',
                'f67', 'f66', 'f65', 'f64', 'f63', 'f62', 'f61', 'f60', 'f279', 'f278', 'f78',
                'f79', 'f74', 'f75', 'f76', 'f77', 'f70', 'f71', 'f72', 'f73']

def CN(x):
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x == -1:
        if np.random.randint(10) > 1:
            return 1
        else:
            return 0
                
def dateProcess(data,set_type = 'train'):
    data.drop(DELETE_INDEX, axis=1, inplace=True)
    if set_type == 'train':
        data.drop(data[data['f161'].isnull()].index, axis=0, inplace=True)
    data.sort_values(by=['date'], axis=0, ascending=True, inplace=True)
    data['date'] = pd.to_datetime(data['date'], format="%Y%m%d")
    data['day'] = data['date'].dt.day
    data['weekday'] = data['date'].dt.weekday
    data['date'] = data['day'] + 100 * data['date'].dt.month
    data.drop(['date'], axis=1, inplace=True)
    if set_type == 'train':
        data['label'] = data['label'].apply(CN)
        #data.drop(data[data['label'] == -1].index, axis=0, inplace=True)
    return data


trainset='../data/train.csv'
df_train = pd.read_csv(trainset)
df_train = dateProcess(df_train,'train')
length = int(len(df_train) * 0.5)
x_train,y_train,x_val,y_val,x_test,y_test = df_train[:length].drop(['label','id'],axis=1),\
                                                    df_train[:length]['label'],\
                                                    df_train[length:].drop(['label','id'],axis=1),\
                                                    df_train[length:]['label'],\
                                                    df_train[length:].drop(['label','id'],axis=1),\
                                                    df_train[length:]['label']
del df_train
gc.collect()
lgb_train = lgb.Dataset(x_train,y_train,free_raw_data=False)
lgb_eval = lgb.Dataset(x_val,y_val,reference=lgb_train,free_raw_data=False)
params = {
    'boosting_type':'dart',
    'objective':'binary',
    'metric':'auc',
    'is_unbalance':False,
    'learning_rate':0.01,
    'min_data_in_leaf':50,
    'min_sum_hessian_in_leaf':50,
    'bagging_fraction':0.7,
    'bagging_freq':1,
    'feature_fraction':0.7,
    'lambda_l1': 0.0,
    'lambda_l2': 1,
    'min_gain_to_split': 0,
    'max_bin':255,
    'num_leaves':31,
    #'metric':'binary_logloss',
}
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=10000,
    fobj=None,
    valid_sets=lgb_eval,
    verbose_eval=True,
    early_stopping_rounds=200
)
preds_offline_raw = gbm.predict(x_test,num_iteration=gbm.best_iteration)
fpr, tpr, _ = roc_curve(y_test, preds_offline_raw)

print(auc(fpr, tpr))

#score = eval_metric(preds_offline_raw,y_test)
#print(score)
testset = '../data/test_b.csv'
df_test = pd.read_csv(testset)
index_x = df_test[(df_test['f234'].isnull()) & (df_test['f20'].isnull())].id.tolist()
df_test = dateProcess(df_test,'test')
online_preds = gbm.predict(df_test.drop('id',axis=1),num_iteration=gbm.best_iteration)

submission = pd.DataFrame({'id':df_test['id'],
                           'score':online_preds}
                          )

submission[submission.id.isin(index_x)]['score']=0
print(submission)
submission.to_csv("submission_7_7.csv",index=False)

