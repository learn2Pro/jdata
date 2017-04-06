# -*- coding: utf-8 -*-
# filename: features_test.py
__author__ = "learn2pro"

import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import datetime
import glob
import features

matplotlib.use('Agg')


# 通过时间划分训练集和测试集
def SplitTrainandTestData(users, date):
    test_data = users[users['date'] == date]
    train_data = users[users['date'] < date]
    return (train_data, test_data)


def strtodatetime(datestr, format):
    return datetime.datetime.strptime(datestr, format)


## 计算时间相差天数
def datediff(beginDate, endDate):
    format = "%Y-%m-%d"
    bd = strtodatetime(beginDate, format)
    ed = strtodatetime(endDate, format)
    count = (ed - bd).days
    return count


# 处理user_action
user_sku_actions = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Action_201604.csv"),
                               dtype={'user_id': object, 'sku_id': object, 'type': object, 'model_id': object,
                                      'cate': object, 'brand': object})
# for user_sku_action in user_sku_actions:
user_sku_action = features.handle_user_action(user_sku_actions)
print user_sku_action.head(10)
user_sku_action.loc[:, 'basic_feature'] = user_sku_action.apply(
    lambda x: features.get_basic_features(x['type'], x['date'], x['hour'], x['model_id'], "2016-04-06", "2016-04-16"), axis=1)
user_sku_action.loc[:, 'view_value'] = user_sku_action['basic_feature'].apply(lambda x: x[0])
user_sku_action.loc[:, 'cart_value'] = user_sku_action['basic_feature'].apply(lambda x: x[1])
user_sku_action.loc[:, 'cancel_cart_value'] = user_sku_action['basic_feature'].apply(lambda x: x[2])
user_sku_action.loc[:, 'order_value'] = user_sku_action['basic_feature'].apply(lambda x: x[3])
user_sku_action.loc[:, 'follow_value'] = user_sku_action['basic_feature'].apply(lambda x: x[4])
user_sku_action.loc[:, 'click_value'] = user_sku_action['basic_feature'].apply(lambda x: x[5])
user_sku_action.loc[:, 'last_order'] = user_sku_action['basic_feature'].apply(lambda x: x[6])
user_sku_action.loc[:, 'last_cart'] = user_sku_action['basic_feature'].apply(lambda x: x[7])
user_sku_action.loc[:, 'last_cancel_cart'] = user_sku_action['basic_feature'].apply(lambda x: x[8])
user_sku_action.loc[:, 'model_std'] = user_sku_action['basic_feature'].apply(lambda x: x[9])
user_sku_action = user_sku_action.drop(['basic_feature'], axis=1)
print user_sku_action.head(5)
# user_sku_action.loc[:, 'label'] = user_sku_action.apply(
#     lambda x: get_label(x['type'], x['date'], '2016-04-11',
#                         '2016-04-15'), axis=1)
user_sku_action = user_sku_action.drop(['date'], axis=1)
user_sku_action = user_sku_action.drop(['hour'], axis=1)
user_sku_action = user_sku_action.drop(['type'], axis=1)
user_sku_action = user_sku_action.drop(['model_id'], axis=1)
user_sku_action.to_csv(os.path.join(os.getcwd(), "data", "user_action_test.csv"), index=False, index_label=False)
# print user_sku_action[user_sku_action['label'] == 1].head(5)
