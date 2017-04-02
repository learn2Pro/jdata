# -*- coding: utf-8 -*-
# filename: train.py
__author__ = "learn2pro"

import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import datetime
import glob

matplotlib.use('Agg')

from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

train = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action.csv"))

test = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action_test.csv"))

# train["date"] = train["date"].fillna('0')
# train["num"] = train["num"].fillna('0')
# train["has_bad"] = train["has_bad"].fillna('0')
# train["rate"] = train["rate"].fillna('0')
# train["comment_rate"] = train["comment_rate"].fillna('0')
# train["attr1"] = train["attr1"].fillna('0')
# train["attr2"] = train["attr2"].fillna('0')
# train["attr3"] = train["attr3"].fillna('0')
#
# test["date"] = test["date"].fillna('0')
# test["num"] = test["num"].fillna('0')
# test["has_bad"] = test["has_bad"].fillna('0')
# test["rate"] = test["rate"].fillna('0')
# test["comment_rate"] = test["comment_rate"].fillna('0')
# test["attr1"] = test["attr1"].fillna('0')
# test["attr2"] = test["attr2"].fillna('0')
# test["attr3"] = test["attr3"].fillna('0')

# predictors = ["user_id", "sku_id", "cate", "brand", "model_std", "view_value", "cart_value", "order_value",
#               "follow_value",
#               "click_value", "last_order", "date", "num", "has_bad", "rate", "comment_rate", "age", "sex", "user_lv_cd",
#               "reg_time",
#               "attr1", "attr2", "attr3"]
predictors = ["user_id", "sku_id", "cate", "brand", "view_value", "cart_value", "order_value",
              "follow_value", "click_value", "last_order"]
results = []
groud_truth = train["label"][1000001:]
for leaf_size in range(1, 500, 10):
    for estimators_size in range(1, 1000, 20):
        algorithm = RandomForestClassifier(min_samples_leaf=leaf_size,
                                           n_estimators=estimators_size, random_state=50)
        algorithm.fit(train[predictors][:1000000], train["label"][:1000000])
        predict = algorithm.predict(train[predictors][1000001:])
        # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
        ratio = (groud_truth == predict).mean()
        results.append((leaf_size, estimators_size, ratio))
        print (leaf_size, estimators_size, ratio)
        # 真实结果和预测结果进行比较，计算准确率

# 打印精度最大的那一个三元组
result = max(results, key=lambda x: x[2])
print result
# alg = RandomForestClassifier(min_samples_leaf=151,
#                              n_estimators=121, random_state=50)
# alg.fit(train[predictors][:], train["label"][:])
# predict = alg.predict(test[predictors][:])
#
# print len(test)
# test = test.loc[:, ['user_id', 'sku_id']]
# test['label'] = predict
# predict = test[test['label'] == 1].drop(['label'], axis=1)
# print len(predict)
# print predict.tail(5)
# submission = pd.DataFrame({
#     "user_id": predict["user_id"],
#     "sku_id": predict["sku_id"]
# })
# submission.to_csv(os.path.join(os.getcwd(), 'data', 'result.csv'), index=False, index_label=False)
