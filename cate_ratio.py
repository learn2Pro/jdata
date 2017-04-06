# -*- coding: utf-8 -*-
# filename: user_ratio.py
__author__ = "learn2pro"

import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import datetime
import glob

matplotlib.use('Agg')


def strtodatetime(datestr, format):
    return datetime.datetime.strptime(datestr, format)


## 计算时间相差天数
def datediff(beginDate, endDate):
    format = "%Y-%m-%d"
    bd = strtodatetime(beginDate, format)
    ed = strtodatetime(endDate, format)
    count = (ed - bd).days
    return count


def handle_user_action(user_action):
    user_action = user_action.drop(["time"], axis=1)
    user_action["model_id"] = user_action["model_id"].fillna("-1")
    user_action[['user_id', 'type', 'brand', 'sku_id']] += "|"
    user_action = user_action.groupby(['cate']).sum().reset_index()
    return user_action


def get_basic_features(user, type, brand, sku):
    try:
        sku = sku.split('|')[:-1]
        type = type.split('|')[:-1]
        brand = brand.split('|')[:-1]
        user = user.split('|')[:-1]
        basic_features = []
        if len(sku) == 0:
            return ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        len_buy, len_follow, len_cart, len_cancel, len_click, len_view = 0, 0, 0, 0, 0, 0
        for i in range(len(type)):
            if type[i] == '1':
                len_view += 1
            if type[i] == '2':
                len_cart += 1
            if type[i] == '3':
                len_cancel += 1
            if type[i] == '4':
                len_buy += 1
            if type[i] == '5':
                len_follow += 1
            if type[i] == '6':
                len_click += 1
        basic_features += [(0.0 if len_follow == 0 else (float(len_buy) / float(len_follow))),
                           (0.0 if len_cart == 0 else (float(len_buy) / float(len_cart))),
                           (0.0 if len_cancel == 0 else (float(len_buy) / float(len_cancel))),
                           (0.0 if len_click == 0 else (float(len_buy) / float(len_click))),
                           (0.0 if len_view == 0 else (float(len_buy) / float(len_view))),
                           len_buy, len(brand), len(sku), len(user)]
        return (basic_features)
    except Exception as e:
        print e
        return ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


# 处理user_action
user_sku_actions = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Action.csv"),
                               dtype={'user_id': object, 'sku_id': object, 'type': object, 'model_id': object,
                                      'cate': object, 'brand': object})
# for user_sku_action in user_sku_actions:
user_sku_action = handle_user_action(user_sku_actions)
print user_sku_action.head(10)

user_sku_action.loc[:, 'basic_feature'] = user_sku_action.apply(
    lambda x: get_basic_features(x['user_id'], x['type'], x['brand'], x['sku_id']),
    axis=1)
user_sku_action.loc[:, 'cate_buy_follow_ratio'] = user_sku_action['basic_feature'].apply(lambda x: x[0])
user_sku_action.loc[:, 'cate_buy_cart_ratio'] = user_sku_action['basic_feature'].apply(lambda x: x[1])
user_sku_action.loc[:, 'cate_buy_cancel_ratio'] = user_sku_action['basic_feature'].apply(lambda x: x[2])
user_sku_action.loc[:, 'cate_buy_click_ratio'] = user_sku_action['basic_feature'].apply(lambda x: x[3])
user_sku_action.loc[:, 'cate_buy_view_ratio'] = user_sku_action['basic_feature'].apply(lambda x: x[4])
user_sku_action.loc[:, 'cate_len_buy'] = user_sku_action['basic_feature'].apply(lambda x: x[5])
user_sku_action.loc[:, 'cate_len_brand'] = user_sku_action['basic_feature'].apply(lambda x: x[6])
user_sku_action.loc[:, 'cate_len_sku'] = user_sku_action['basic_feature'].apply(lambda x: x[7])
user_sku_action.loc[:, 'cate_len_user'] = user_sku_action['basic_feature'].apply(lambda x: x[8])

user_sku_action = user_sku_action.drop(['basic_feature'], axis=1)
user_sku_action = user_sku_action.drop(['model_id'], axis=1)
user_sku_action = user_sku_action.drop(['user_id'], axis=1)
user_sku_action = user_sku_action.drop(['type'], axis=1)
user_sku_action = user_sku_action.drop(['brand'], axis=1)
user_sku_action = user_sku_action.drop(['sku_id'], axis=1)
print user_sku_action.head(5)
user_sku_action.to_csv(os.path.join(os.getcwd(), "data", "cate_ratio.csv"), index=False, index_label=False)
