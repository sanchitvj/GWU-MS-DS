# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:02:51 2022

@author: Upmanyu
"""

import pandas as pd
import numpy as np
import os

import wbgapi as wb

df_country = pd.read_csv('CountrUsedN(1).csv')

country_name_list = df_country['country'].tolist()

country_list = list(wb.economy.coder(country_name_list).values())
country_set = set(country_list)


# years of interest
year_list = range(1994, 2022)

# measures of interest
wb_measures_list = df_country['Index'].tolist()

# import the data from the API
raw_dat = wb.data.DataFrame(wb_measures_list, country_set, year_list)

raw_dat = raw_dat.reset_index()

raw_dat['series'] = raw_dat['series']+'_'+ raw_dat['economy']
raw_dat_t = raw_dat.drop(['economy'],axis=1).T

raw_dat_t.columns = list(raw_dat_t.iloc[0,:])

raw_dat_t = raw_dat_t.iloc[1:,:]

nan_sum = raw_dat_t.isna().sum()
raw_dat_t.dropna(thresh=len(raw_dat_t) - 3, axis=1,inplace=True)

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
raw_dat_inputed = pd.DataFrame(imputer.fit_transform(raw_dat_t))

raw_dat_inputed.columns = raw_dat_t.columns

raw_dat_inputed.filter(regex='EN.ATM.CO2E.KT',axis=1)/raw_dat_inputed.filter(regex='SP.POP.TOTL',axis=1)
"EN.ATM.CO2E.KT"










