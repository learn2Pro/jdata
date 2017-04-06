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

user_action = pd.read_csv(os.path.join(os.getcwd(), "data", "user_action_test.csv"))
user_action['model_std'] = user_action['model_std'].fillna('0')
user_ratio = pd.read_csv(os.path.join(os.getcwd(), "data", "user_ratio.csv"))
sku_ratio = pd.read_csv(os.path.join(os.getcwd(), "data", "sku_ratio.csv"))
users_action = user_ratio.join(user_action.set_index('user_id'), how='inner', on='user_id')
users_action = sku_ratio.join(users_action.set_index('sku_id'), how='inner', on='sku_id')

print users_action.tail(5)
users_action.to_csv(os.path.join(os.getcwd(), "data", "action_feature_ratio_test.csv"), index=False, index_label=False)
