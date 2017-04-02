# -*- coding: utf-8 -*-
# filename: comment.py
__author__ = "learn2pro"

import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import datetime
import glob
import math

matplotlib.use('Agg')

from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


def strtodatetime(datestr, format):
    return datetime.datetime.strptime(datestr, format)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


## 计算时间相差天数
def datediff(beginDate, endDate):
    format = "%Y-%m-%d"
    bd = strtodatetime(beginDate, format)
    ed = strtodatetime(endDate, format)
    count = (ed - bd).days
    return count


def get_percent_change(dates, num, has_bad, commnet_rate):
    cr = commnet_rate.split('|')
    rslist = []
    num = num.split('|')
    has_bad = has_bad.split('|')
    date = dates.split('|')
    size = len(cr)
    percent = 0.0
    if size <= 2:
        return ([date[0], num[0], has_bad[0], percent])
    last_action = []
    for i in range(len(date) - 2):
        last_action.append(datediff(date[i], "2016-04-16"))
    percent = sigmoid(float(cr[0]) - float(cr[size - 2])) - sigmoid(datediff(date[0], date[size - 2]))
    rslist += [min(last_action), num[0], has_bad[0], percent]
    return (rslist)


comment = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Comment.csv"),
                      dtype={'dt': object, 'comment_num': object,
                             'has_bad_comment': object, 'bad_comment_rate': object})
# for comment in comments:
comment[['dt', 'comment_num', 'has_bad_comment', 'bad_comment_rate']] += '|'
comment = comment.groupby('sku_id').sum().reset_index()
comment.loc[:, 'feature'] = comment.apply(
    lambda x: get_percent_change(x['dt'], x['comment_num'], x['has_bad_comment'], x['bad_comment_rate']),
    axis=1)
print comment.head(5)
comment.loc[:, 'date'] = comment['feature'].apply(lambda x: x[0])
comment.loc[:, 'num'] = comment['feature'].apply(lambda x: x[1])
comment.loc[:, 'has_bad'] = comment['feature'].apply(lambda x: x[2])
comment.loc[:, 'rate'] = comment['feature'].apply(lambda x: x[3])
comment.loc[:, 'comment_rate'] = comment['bad_comment_rate'].apply(lambda x: x[0].split('|')[0])

comment = comment.drop(['feature'], axis=1)
comment = comment.drop(['dt'], axis=1)
comment = comment.drop(['comment_num'], axis=1)
comment = comment.drop(['has_bad_comment'], axis=1)
comment = comment.drop(['bad_comment_rate'], axis=1)

comment.to_csv(os.path.join(os.getcwd(), 'data', 'comment_test.csv'), index=False, index_label=False)
print comment.head(5)


# comment = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Comment.csv"))
#
# Product = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Product.csv"))
# for r in glob.glob("test*.csv"):
#         csv=pandas.read_csv(r)
#         csv.to_csv("test.txt",mode="a+")
