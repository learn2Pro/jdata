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

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


# F11=6*Recall*Precise/(5*Recall+Precise)
def F11(right, train_len, test_len):
    Recall = right / train_len
    Precise = right / test_len
    return 6 * Recall * Precise / (5 * Recall + Precise + 1)


# train = pd.read_csv(os.path.join(os.getcwd(), "data", "feature_train.csv"))
#
# test = pd.read_csv(os.path.join(os.getcwd(), "data", "feature_train_test.csv"))


# train = pd.read_csv(os.path.join(os.getcwd(), "data", "feature_train.csv"))
#
# test = pd.read_csv(os.path.join(os.getcwd(), "data", "feature_train_test.csv"))

# train = pd.read_csv(os.path.join(os.getcwd(), "data", "action_feature_ratio.csv"))
#
# test = pd.read_csv(os.path.join(os.getcwd(), "data", "action_feature_ratio_test.csv"))

train = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action.csv"))

test = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action_test.csv"))

# train["age"] = train["age"].fillna("-1")
# train["sex"] = train["sex"].fillna("-1")
# test["age"] = test["age"].fillna("-1")
# test["sex"] = test["sex"].fillna("-1")

# predictors = ["cate", "brand", "view_value", "cart_value", "cancel_cart_value", "order_value",
#               "follow_value",
#               "click_value", "last_order", 'last_cart', 'last_cancel_cart','model_std']

predictors = ["user_buy_follow_ratio", "user_buy_cart_ratio",
              "user_buy_cancel_ratio",
              "user_buy_click_ratio", "user_buy_view_ratio", "user_len_buy", "user_len_cate",
              "user_len_sku", "user_len_brand", "sku_buy_follow_ratio", "sku_buy_cart_ratio",
              "sku_buy_cancel_ratio", "sku_buy_click_ratio", "sku_buy_view_ratio", "sku_len_buy",
              "sku_len_cate", "sku_len_user", "sku_len_brand",
              "cate", "brand", "view_value", "cart_value", "cancel_cart_value", "order_value",
              "follow_value", "click_value", 'last_order', 'last_cart', 'last_cancel_cart', 'model_std']

# predictors = ["a1", "a2", "a3", "age", "sex", "user_lv_cd", "reg_time", "user_buy_follow_ratio", "user_buy_cart_ratio",
#               "user_buy_cancel_ratio",
#               "user_buy_click_ratio", "user_buy_view_ratio", "user_len_buy", "user_len_cate",
#               "user_len_sku", "user_len_brand", "sku_buy_follow_ratio", "sku_buy_cart_ratio",
#               "sku_buy_cancel_ratio", "sku_buy_click_ratio", "sku_buy_view_ratio", "sku_len_buy",
#               "sku_len_cate", "sku_len_user", "sku_len_brand",
#               "cate", "brand", "view_value", "cart_value", "cancel_cart_value", "order_value",
#               "follow_value", "click_value", 'last_order', 'last_cart', 'last_cancel_cart', 'model_std',
#               "date", "num", "has_bad", "percent", "comment_rate"]

results = []

# train_len = 500000
# test_len = len(train) - 500000
# groud_truth = train["label"][500001:]
# for leaf_size in range(1, 500, 50):
#     for estimators_size in range(1, 1000, 100):
#         algorithm = RandomForestClassifier(min_samples_leaf=leaf_size,
#                                            n_estimators=estimators_size, random_state=50)
#         algorithm.fit(train[predictors][:500000], train["label"][:500000])
#         predict = algorithm.predict(train[predictors][500001:])
#         # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
#         right = 0
#         for i in range(len(groud_truth)):
#             x = 500001 + i
#             if groud_truth[x] == 1 and predict[i] == 1:
#                 right += 1
#         f11 = F11(right, train_len, test_len)
#         results.append((leaf_size, estimators_size, f11))
#         print (leaf_size, estimators_size, f11)
#         # 真实结果和预测结果进行比较，计算准确率
#
# # # 打印精度最大的那一个三元组
# result = max(results, key=lambda x: x[2])
# print result


alg = RandomForestClassifier(100)
alg.fit(train[predictors][:], train["label"][:])
predict = alg.predict(test[predictors][:])

print len(test)
test = test.loc[:, ['user_id', 'sku_id']]
test['label'] = predict
predict = test[test['label'] == 1].drop(['label'], axis=1)
print len(predict)
print predict.tail(5)
submission = pd.DataFrame({
    "user_id": predict["user_id"],
    "sku_id": predict["sku_id"]
})
submission.drop_duplicates('user_id', inplace=True)
submission.to_csv(os.path.join(os.getcwd(), 'submit', 'result.csv'), index=False, index_label=False)
