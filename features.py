# -*- coding: utf-8 -*-
# filename: features.py
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


## Input: hehavior_type, date, 划分时间
def get_label(x, y, begin, end):
    format = "%Y-%m-%d"
    try:
        bd = strtodatetime(begin, format)
        ed = strtodatetime(end, format)
        last_day = max(y.split('|'))
        last_day = strtodatetime(last_day, format)
        if last_day <= bd:
            return (0)
        days = y.split('|')
        behaviors = x.split('|')
        flag = 0
        for i in range(len(days)):
            day = strtodatetime(days[i], format)
            if day >= bd and day <= ed and behaviors[i] == '4':
                flag = 1
                break
        if flag:
            return (1)
        return (0)
    except:
        return (0)


## bahavior_type, date,
## Basic features including:
def get_basic_features(x, y, z, begin, end):
    try:
        y = y.split('|')
        x = x.split('|')
        z = z.split('|')
        basic_features = []
        if len(x) == 0:
            return ([0] * 7)
        model_std, view_value, cart_value, order_value, follow_value, click_value = 0, 0, 0, 0, 0, 0
        zArray = np.array(z[:-1], dtype=np.float)
        model_std = zArray.std()
        last_action = []
        for i in range(len(x) - 1):
            if y[i] <= begin and y[i] >= end:
                continue
            if x[i] == '1':
                view_value += np.exp(-0.3 * datediff(y[i], end))
            if x[i] == '2':
                cart_value += np.exp(-0.05 * datediff(y[i], end))
            if x[i] == '3':
                cart_value -= np.exp(-0.05 * datediff(y[i], end))
            if x[i] == '4':
                order_value += 1
            if x[i] == '5':
                follow_value += np.exp(-0.15 * datediff(y[i], end))
            if x[i] == '6':
                click_value += np.exp(-0.3 * datediff(y[i], end))
            last_action.append(datediff(y[i], end))
        if len(last_action) == 0:
            return ([0] * 7)
        ## todo 加入时间hour
        basic_features += [view_value, cart_value, order_value, follow_value, click_value, min(last_action), model_std]
        return (basic_features)
    except Exception as e:
        print "get_basic_features!!====" + e
        return ([0] * 7)


def handle_users(users, time):
    users.loc[users["age"] == "15岁以下".decode("utf-8"), "age"] = 0
    users.loc[users["age"] == "16-25岁".decode("utf-8"), "age"] = 1
    users.loc[users["age"] == "26-35岁".decode("utf-8"), "age"] = 2
    users.loc[users["age"] == "36-45岁".decode("utf-8"), "age"] = 3
    users.loc[users["age"] == "46-55岁".decode("utf-8"), "age"] = 4
    users.loc[users["age"] == "56岁以上".decode("utf-8"), "age"] = 5
    users.loc[users["age"] == "-1".decode("utf-8"), "age"] = -1
    users.loc[:, 'reg_time'] = users.apply(lambda x: datediff(x['user_reg_dt'].replace(r"/", r"-"), time), axis=1)
    users = users.drop(['user_reg_dt'], axis=1)
    return users


def handle_user_action(user_action):
    user_action["date"] = user_action.time.map(lambda x: x.split(' ')[0])
    user_action["hour"] = user_action.time.map(lambda x: x.split(' ')[1].split(':')[0])
    user_action["date"] = user_action["date"].fillna(user_action["date"].min())
    ## todo change to user_action["model_id"].median()
    user_action["model_id"] = user_action["model_id"].fillna(str(user_action["model_id"].median()))
    user_action = user_action.drop(["time"], axis=1)
    user_action[['date', 'hour', 'type', 'model_id']] += "|"
    user_action = user_action.groupby(['user_id', 'sku_id', 'cate', 'brand']).sum().reset_index()
    return user_action


# 处理user表
# users = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_User.csv"), encoding='gbk')
# users = handle_users(users, '2016-04-16')
# users.to_csv(os.path.join(os.getcwd(), "data", "users_test.csv"), index=False, index_label=False)
# print users.head(5)

# for filepath in glob.glob(os.path.join(os.getcwd(), "data", "JData_Action*.csv")):
# 处理user_action
user_sku_actions = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Action_201604.csv"),
                               dtype={'user_id': object, 'sku_id': object, 'type': object, 'model_id': object,
                                      'cate': object, 'brand': object})
# for user_sku_action in user_sku_actions:
user_sku_action = handle_user_action(user_sku_actions)
print user_sku_action.head(10)
user_sku_action.loc[:, 'basic_feature'] = user_sku_action.apply(
    lambda x: get_basic_features(x['type'], x['date'], x['model_id'], "2016-04-06", "2016-04-16"), axis=1)
user_sku_action.loc[:, 'view_value'] = user_sku_action['basic_feature'].apply(lambda x: x[0])
user_sku_action.loc[:, 'cart_value'] = user_sku_action['basic_feature'].apply(lambda x: x[1])
user_sku_action.loc[:, 'order_value'] = user_sku_action['basic_feature'].apply(lambda x: x[2])
user_sku_action.loc[:, 'follow_value'] = user_sku_action['basic_feature'].apply(lambda x: x[3])
user_sku_action.loc[:, 'click_value'] = user_sku_action['basic_feature'].apply(lambda x: x[4])
user_sku_action.loc[:, 'last_order'] = user_sku_action['basic_feature'].apply(lambda x: x[5])
user_sku_action.loc[:, 'model_std'] = user_sku_action['basic_feature'].apply(lambda x: x[6])
user_sku_action = user_sku_action.drop(['basic_feature'], axis=1)
print user_sku_action.head(5)
# user_sku_action = get_basic_features(user_sku_action['date'], user_sku_action['type'], "2016/04/11")
# user_sku_action.loc[:, 'label'] = user_sku_action.apply(
#     lambda x: get_label(x['type'], x['date'], '2016-04-11',
#                         '2016-04-15'), axis=1)
user_sku_action = user_sku_action.drop(['date'], axis=1)
user_sku_action = user_sku_action.drop(['hour'], axis=1)
user_sku_action = user_sku_action.drop(['type'], axis=1)
user_sku_action.to_csv(os.path.join(os.getcwd(), "data", "user_action_test.csv"), index=False, index_label=False)
# print user_sku_action[user_sku_action['label'] == 1].head(5)
