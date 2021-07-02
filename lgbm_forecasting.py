import os
import glob
import datetime

import numpy as np
import pandas as pd
from scipy import signal, stats, interpolate

import statsmodels.api as sm
import statsmodels

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold

import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 200)

class LGBMForecast():
    def __init__(self, max_depth=7):
        super(LGBMForecast, self).__init__()
        self.max_depth = max_depth
        self.num_leaves = int(.7 * 2 ** max_depth)

    def load_csv(self, filepath):
        files = glob.glob(filepath)
        lis = []
        for filename in files:
            parser = lambda x: pd.to_datetime(x, format='%Y%m%d%H')
            df = pd.read_csv(filename, index_col=0, parse_dates=True, date_parser=parser,
                             usecols=[0,5,6,7,8,9,10,11,
                                      15,16,17,18,
                                      120]
                             )
            lis.append(df)
            df = pd.concat(lis, axis=0)
            df.index.name = 'date'
        return df

    def summer_winter_split(self, filepath, train=True, to_csv=False):
        df = LGBMForecast.load_csv(filepath)
        if train:
            # prepare summer data
            df_s_17 = df['2017-04-01 00:00:00':'2017-09-30 23:00:00']
            df_s_18 = df['2018-04-01 00:00:00':'2018-09-30 23:00:00']
            df_s_19 = df['2019-04-01 00:00:00':'2019-09-30 23:00:00']
            df_s = pd.concat([df_s_17, df_s_18, df_s_19])
            # prepare winter data
            df_w_17_1 = df[:'2017-03-31 23:00:00']
            df_w_17_2 = df['2017-10-01 00:00:00':'2018-03-31 23:00:00']
            df_w_17 = pd.concat([df_w_17_1, df_w_17_2])
            df_w_18 = df['2018-10-01 00:00:00':'2019-03-31 23:00:00']
            df_w_19 = df['2019-10-01 00:00:00':]
            df_w = pd.concat([df_w_17, df_w_18, df_w_19])
            if to_csv:
                filename_split = os.path.join(filepath)
                df_s.to_csv(f'{filename_split[0]}_summer{filename_split[1]}')
                df_w.to_csv(f'{filename_split[0]}_winter{filename_split[1]}')
            return df_s, df_w
        else:
            df_s = df['2020-04-01 00:00:00':'2020-09-30 23:00:00']
            df_w_1 = df[:'2020-03-31 23:00:00']
            df_w_2 = df['2020-10-01 00:00:00':]
            df_w = pd.concat([df_w_1, df_w_2])
            if to_csv:
                filename_split = os.path.join(filepath)
                df_s.to_csv(f'{filename_split[0]}_summer{filename_split[1]}')
                df_w.to_csv(f'{filename_split[0]}_winter{filename_split[1]}')
            return df_s, df_w

    #def train(self, X, y, params, X_test, n_splits=5, n_test_folds=1):