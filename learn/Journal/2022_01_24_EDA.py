# %%
from MLBookCamp.DataRegister import DataRegister
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

dr = DataRegister("../../datasets")
raw_df = dr.get_df_from_key("udacity_diabetes")
raw_df.head()

# Null values
raw_df.isnull().sum()

# OUtcome proportion
(raw_df.Outcome==True).sum()/(raw_df.Outcome==False).sum()

# Distributions
attributes = raw_df.columns.values
scatter_matrix(raw_df[attributes], figsize=(12, 8))
#Correlation
corr_matrix = raw_df.corr()
corr_matrix["Outcome"].sort_values(ascending=False)

raw_df.describe()