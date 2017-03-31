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


def strtodatetime(datestr, format):
    return datetime.datetime.strptime(datestr, format)


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
    size = len(cr)
    if size <= 2:
        return ([0] * 5)
    num = num.split('|')
    has_bad = has_bad.split('|')
    date = dates.split('|')
    daydis = datediff(date[0], date[size - 2])
    if daydis != 0:
        percent = (float(cr[0]) - float(cr[size - 2])) / datediff(date[0], date[size - 2])
        rslist += [date[0], num[0], has_bad[0], cr[0], percent]
        return percent
    else:
        return ([0] * 5)


comment = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Comment.csv"),
                      dtype={'dt': object, 'comment_num': object,
                             'has_bad_comment': object, 'bad_comment_rate': object})

comment[['dt', 'comment_num', 'has_bad_comment', 'bad_comment_rate']] += '|'
comment = comment.groupby('sku_id').sum().reset_index()
comment.loc[:, 'feature'] = comment.apply(
    lambda x: get_percent_change(x['dt'], x['comment_num'], x['has_bad_comment'], x['bad_comment_rate']), axis=1)

comment.loc[:, 'date'] = comment['feature'].apply(lambda x: x[0])
comment.loc[:, 'num'] = comment['feature'].apply(lambda x: x[1])
comment.loc[:, 'has_bad'] = comment['feature'].apply(lambda x: x[2])
comment.loc[:, 'commnet_rate'] = comment['feature'].apply(lambda x: x[3])
comment.loc[:, 'rate'] = comment['feature'].apply(lambda x: x[4])

comment = comment.drop(comment['feature'], axis=1)
comment = comment.drop(comment['dt'], axis=1)
comment = comment.drop(comment['comment_num'], axis=1)
comment = comment.drop(comment['has_bad_comment'], axis=1)
comment = comment.drop(comment['bad_comment_rate'], axis=1)

comment.to_csv(os.path.join(os.getcwd(), 'date', 'comment.csv'), index=False, index_label=False)
print comment.head(5)
