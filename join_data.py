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

users = pd.read_csv(os.path.join(os.getcwd(), "data", "users.csv"))
comments = pd.read_csv(os.path.join(os.getcwd(), "data", "comment.csv"))
users_action = pd.read_csv(os.path.join(os.getcwd(), "data", "action_feature_ratio.csv"))
products = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Product.csv"))

# users = pd.read_csv(os.path.join(os.getcwd(), "data", "users_test.csv"))
# comments = pd.read_csv(os.path.join(os.getcwd(), "data", "comment_test.csv"))
# users_action = pd.read_csv(os.path.join(os.getcwd(), "data", "action_feature_ratio_test.csv"))
# products = pd.read_csv(os.path.join(os.getcwd(), "data", "JData_Product.csv"))

users_action = users_action.join(users.set_index('user_id'), how='left', on='user_id')
users_action = users_action.join(comments.set_index('sku_id'), how='left', on='sku_id')
users_action = users_action.join(products[['sku_id', 'a1', 'a2', 'a3']].set_index('sku_id'), how='left',
                                 on='sku_id')

users_action.fillna('-1', inplace=True)

# users_action = products[['sku_id', 'a1', 'a2', 'a3']].join(comments.set_index('sku_id'), how='inner',
#                                                            on='sku_id')


print users_action.tail(5)
print len(users_action)
users_action.to_csv(os.path.join(os.getcwd(), "data", "feature_train.csv"), index=False, index_label=False)

# product_comment = users_action.join(comments.set_index('sku_id'), how='inner',
#                                                                     on='sku_id')
# print product_comment.tail(5)
# users_action.to_csv(os.path.join(os.getcwd(), "data", "product_comment.csv"), index=False, index_label=False)
