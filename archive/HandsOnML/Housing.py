#This is the condense code of the Housing data set analaysis and model training
import os
import tarfile
import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
"""
////-> Load Data Functions
"""
HOUSING_PATH = os.path.join("../datasets", "housing")
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

"""
////-> Initial Analysis functions
"""
def plot_histogram(dataFrame):
    dataFrame.hist(bins=50, figsize=(20,15))
    plt.show()

def perform_stratified_sampling(dataFrame, column_based="median_income"):
    #Begin by stratifying the ColumnBased Attribute on the dataset.
    #In this case: "median_income"
    dataFrame["income_cat"] = pd.cut(dataFrame[column_based],
                                                 bins= [0,1.5,3.0,4.5, 6, np.inf],
                                                 labels=[1,2,3,4,5])
    dataFrame["income_cat"].hist()
    plt.show()

    #Now with the stratified category appended to the dataFrame
    #We proceed to perform StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_indices, test_indices in split.split(dataFrame, dataFrame['income_cat']):
        strat_train_set = dataFrame.loc[train_indices]
        strat_test_set = dataFrame.loc[test_indices]

    #let's compare the income_cat proportions in the test set:
    strat_proportions = strat_test_set['income_cat'].value_counts() / len(strat_test_set)
    print(strat_proportions)

    #Do the sampling using the random approach and compare the proportions on the sample
    random_train_set, random_test_set = train_test_split(dataFrame, test_size=0.2, random_state=42)
    random_proportions = random_test_set['income_cat'].value_counts() / len(random_test_set)
    print(random_proportions)

    #Remove the stratified  column, we just used it for this sampling approach
    for set_ in(strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set

def execute_initial_exploration_and_sampling(dataFrame):
    plot_histogram(dataFrame)
    strat_train_set, strat_test_set = perform_stratified_sampling(dataFrame)
    return strat_train_set, strat_test_set


"""
////-> DATA PRE-PROCESSING WITH PIPELINES
"""
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]


def prepare_data(train_set):
    #Separate features and labels
    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    #from our train data, separate numeric features from categorical features
    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)

    print('Features and Labels are ready for training')
    print(housing_prepared[0])
    return housing_prepared, housing_labels, full_pipeline

"""
////-> MAIN
"""
def main():
    print('hello')
    # Load data
    rawDataFrame = load_housing_data()

    # Initial Exploration
    strat_train_set, strat_test_set = execute_initial_exploration_and_sampling(rawDataFrame)

    #Pre-processing
    housing_prepared, housing_labels, full_pipeline = prepare_data(strat_train_set)
    print(type(housing_prepared))
    print(type(housing_labels))

    #Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # Try Linear Regression
    some_data = strat_train_set.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    print("Predictions:", lin_reg.predict(some_data_prepared))
    print("Labels:", list(some_labels))

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(f'lin_rmse: {lin_rmse}')

if __name__ == "__main__":
    main()