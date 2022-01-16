from nis import cat
from sys import displayhook
from unittest.case import DIFF_OMITTED
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

class EvaluationMetrics(object):
    def __init__(self, path):
        """
        raw_df  Contains the raw dataframe as first read from CSV file: WA_Fn-UseC_-Telco-Customer-Churn.csv
        """
        self.path = path
        self.raw_df = pd.read_csv(path)
    
    categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
    
    numerical = ['tenure', 'monthlycharges', 'totalcharges']

    def describe_data(self):
        print(f'number of rows:{len(self.raw_df)}')
        print(f'types :{self.raw_df}')
        displayhook(self.raw_df.head().T)
    

    def preprocess_dataframe(self):
        df = self.raw_df
        # TotalCharges must be numeric
        df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce")
        # Fill NAs with 0
        df.TotalCharges = df.TotalCharges.fillna(0)
        # Let’s make it uniform by lowercasing everything and replacing spaces with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        # Our target Column Churn should be numeric as well
        df.churn = (df.churn=="Yes").astype(int)
        displayhook(df.head().T)
        print(f'Churn Values:{df.churn.value_counts()}')

        return df

    def perform_EDA(self, df):
        # Check for null values
        displayhook(df.isnull().sum())
        # Check number of unique values on each categorical columns
        displayhook(df[self.categorical].nunique())
        # Churn Rates:  calculate the risk for each category
        global_mean = round(df.churn.mean(),3)
        print(f'global churn mean:{global_mean}')
    
        for col in self.categorical:
            df_group = df.groupby(by=col).churn.agg(['mean'])
            df_group['diff'] = df_group['mean']-global_mean
            df_group['risk'] = df_group['mean']/global_mean
            displayhook(df_group)
    
        def calculate_mi(series):                                      
            return mutual_info_score(series, df.churn)  
        
        # Mutual info
        df_mi = df[self.categorical].apply(calculate_mi)         
        df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI') 
        displayhook(df_mi)
        # Correlation    
        displayhook(df[self.numerical].corrwith(df.churn))
    
    def feature_engineering(self, df, columns):
        # Train Dictionary
        train_dict = df[columns].to_dict(orient='records')
        dv = DictVectorizer(sparse=False)
        dv.fit(train_dict)
        # Obtain the X_train feature matrix
        X_Vector = dv.transform(train_dict)
        return X_Vector
    
    def split_data(self, df, columns):
        # Split Full Data in training and test
        df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
        # Split Training data in training and validation
        df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)
    
        X_train = self.feature_engineering(df_train,columns)
        X_val = self.feature_engineering(df_val, columns)
        X_test = self.feature_engineering(df_test, columns)

        # Target data
        y_train = df_train.churn.values
        del df_train['churn']
        
        y_val = df_val.churn.values
        del df_val['churn']
        
        y_test = df_test.churn.values
        del df_test['churn']
        
        # Log how data is distributed
        print(f'Training Data X_train:{len(X_train)}, y_train:{len(y_train)}')
        print(f'Validation Data:{len(X_val)}, y_val:{len(y_val)}')
        print(f'Testing Data:{len(X_test)}, y_test:{len(y_test)}')

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_Logistic_Regression(self, X_train, y_train):
        model = LogisticRegression(solver='liblinear', random_state=1)
        model.fit(X_train, y_train)
        return model

    def get_decision_tree(self,X_train, y_train):
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        return model

    class DummyModel:    
        def predict_proba(self,y_val):
            return np.repeat(False, len(y_val))