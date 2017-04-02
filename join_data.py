# -*- coding: utf-8 -*-
# filename: join_data.py
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

# users = pd.read_csv(os.path.join(os.getcwd(), "data", "users.csv"), dtype={'user_id': object})
# comments = pd.read_csv(os.path.join(os.getcwd(), "data", "comment.csv"))
# users_action = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action.csv"),
#                            dtype={'user_id': object, 'sku_id': object})
# products = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Product.csv"),
#                        dtype={'sku_id': object})
users = pd.read_csv(os.path.join(os.getcwd(), "data", "users_test.csv"), dtype={'user_id': object})
comments = pd.read_csv(os.path.join(os.getcwd(), "data", "comment_test.csv"))
users_action = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action_test.csv"),
                           dtype={'user_id': object, 'sku_id': object})
products = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Product.csv"),
                       dtype={'sku_id': object})

users_action = users_action.join(users.set_index('user_id'), on='user_id')
users_action = users_action.join(comments.set_index('sku_id'), on='sku_id')
users_action = users_action.join(products[['sku_id', 'attr1', 'attr2', 'attr3']].set_index('sku_id'), on='sku_id')

print users_action.head(5)
users_action.to_csv(os.path.join(os.getcwd(), "data", "feature_train_test.csv"), index=False, index_label=False)
