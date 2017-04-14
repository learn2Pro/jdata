# -*- coding: utf-8 -*-
# filename: xgb_train.py
__author__ = "learn2pro"

import numpy as np
import matplotlib
import pandas as pd
import os
import datetime
import glob
import matplotlib.pylab as plt

from xgb_feature import report
from xgb_feature import make_test_set
from xgb_feature import make_train_set
from sklearn.model_selection import train_test_split
from sklearn import metrics

matplotlib.use('Agg')
import xgboost as xgb


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['label'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['label'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

def xgboost_make_submission():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2,
                                                        random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
    param['nthread'] = 4
    # param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)
    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date)
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    pred = sub_user_index[sub_user_index['label'] >= 0.03]
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('./sub/submission.csv', index=False, index_label=False)


def xgboost_test_make_submission():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    train = pd.read_csv(os.path.join(os.getcwd(), "data", "feature_train.csv"))

    test = pd.read_csv(os.path.join(os.getcwd(), "data", "feature_train_test.csv"))

    # train = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action.csv"))
    #
    # test = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action_test.csv"))

    predictors = ["a1", "a2", "a3", "age", "sex", "user_lv_cd", "reg_time", "user_buy_follow_ratio",
                  "user_buy_cart_ratio",
                  "user_buy_cancel_ratio",
                  "user_buy_click_ratio", "user_buy_view_ratio", "user_len_buy", "user_len_cate",
                  "user_len_sku", "user_len_brand", "sku_buy_follow_ratio", "sku_buy_cart_ratio",
                  "sku_buy_cancel_ratio", "sku_buy_click_ratio", "sku_buy_view_ratio", "sku_len_buy",
                  "sku_len_cate", "sku_len_user", "sku_len_brand",
                  "cate", "brand", "view_value", "cart_value", "cancel_cart_value", "order_value",
                  "follow_value", "click_value", 'last_order', 'last_cart', 'last_cancel_cart', 'model_std',
                  "date", "num", "has_bad", "percent", "comment_rate"]

    # predictors = ["cate", "brand", "view_value", "cart_value", "cancel_cart_value", "order_value",
    #               "follow_value", "click_value", 'last_order', 'last_cart', 'last_cancel_cart', 'model_std']

    # user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    # X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2,
    #                                                     random_state=0)
    print len(train)
    train.fillna(-1)
    test.fillna(-1)
    X_train, X_test, y_train, y_test = train_test_split(train[predictors].values, train['label'].values, test_size=0.2,
                                                        random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
    param['nthread'] = 4
    # param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)
    sub_trainning_data = xgb.DMatrix(test[predictors].values)
    y = bst.predict(sub_trainning_data)
    test['label'] = y
    pred = test[test['label'] >= 0.1]
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    # pred.drop_duplicates('user_id', inplace=True)
    print pred.tail(5)
    print len(pred)
    pred.to_csv('./submit/xgb_result_1.csv', index=False, index_label=False)


def xgboost_cv():
    train_start_date = '2016-03-05'
    train_end_date = '2016-04-06'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-02-05'
    sub_end_date = '2016-03-05'
    sub_test_start_date = '2016-03-05'
    sub_test_end_date = '2016-03-10'

    train = pd.read_csv(os.path.join(os.getcwd(), "data", "feature_train.csv"))

    test = pd.read_csv(os.path.join(os.getcwd(), "data", "feature_train_test.csv"))

    # train = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action.csv"))
    #
    # test = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action_test.csv"))

    # predictors = ["a1", "a2", "a3", "age", "sex", "user_lv_cd", "reg_time", "user_buy_follow_ratio",
    #               "user_buy_cart_ratio",
    #               "user_buy_cancel_ratio",
    #               "user_buy_click_ratio", "user_buy_view_ratio", "user_len_buy", "user_len_cate",
    #               "user_len_sku", "user_len_brand", "sku_buy_follow_ratio", "sku_buy_cart_ratio",
    #               "sku_buy_cancel_ratio", "sku_buy_click_ratio", "sku_buy_view_ratio", "sku_len_buy",
    #               "sku_len_cate", "sku_len_user", "sku_len_brand",
    #               "cate", "brand", "view_value", "cart_value", "cancel_cart_value", "order_value",
    #               "follow_value", "click_value", 'last_order', 'last_cart', 'last_cancel_cart', 'model_std',
    #               "date", "num", "has_bad", "percent", "comment_rate"]

    predictors = ["cate", "brand", "view_value", "cart_value", "cancel_cart_value", "order_value",
                  "follow_value", "click_value", 'last_order', 'last_cart', 'last_cancel_cart', 'model_std']

    print len(train)
    dtrain = xgb.DMatrix(train[predictors][:800000].values, label=train['label'][:800000], missing=-1.0)
    dtest = xgb.DMatrix(train[predictors][800001:].values, label=train['label'][800001:], missing=-1.0)

    param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 4000
    param['nthread'] = 4
    param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.cv(plst, dtrain, num_round)

    # sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
    #                                                                sub_test_start_date, sub_test_end_date)
    # test = xgb.DMatrix(sub_trainning_date)
    y = bst.predict(dtest)

    pred = dtest.copy()
    y_true = dtest.copy()
    pred['label'] = y
    # y_true['label'] = label
    report(pred, y_true)


if __name__ == '__main__':
    # xgboost_make_submission()
    # xgboost_test_make_submission()
    xgboost_cv()
