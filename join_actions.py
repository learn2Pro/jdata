# -*- coding: utf-8 -*-
# filename: join_actions.py
__author__ = "learn2pro"

import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import datetime
import glob

matplotlib.use('Agg')

cnt = 0
for r in glob.glob(os.path.join(os.getcwd(), "data", "JData_Action*.csv")):
    cnt += 1
    csv = pd.read_csv(r)
    if cnt == 1:
        csv.to_csv(os.path.join(os.getcwd(), "data", "JData_Action.csv"), mode="a+", index=False, index_label=False)
    else:
        csv.to_csv(os.path.join(os.getcwd(), "data", "JData_Action.csv"), mode="a+", index=False, index_label=False,
                   header=False)
