from MLBookCamp.DataRegister import DataRegister
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
from sklearn.model_selection import KFold   
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

class BinaryEvaluationMetrics:
    """ Evaluation Metrics for binary Classification
    """
    def __init__(self, dataset_path):
        self.data_register = DataRegister(dataset_path)
    
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
    
    def add_model(self, key, model):
        self.data_register.register_model(key, model)
    
    def get_model_for(self, key):
        return self.data_register.register_model[key]

    def get_model_raw_predictions_for(self, key, X_val):
        if key == "Dummy":
            return np.repeat(0, len(X_val))
        if key == "Random":
            preds_prob = np.random.uniform(0,1,len(X_val))
            return preds_prob

        model = self.data_register.get_model_for(key)
        preds = ((model.predict_proba(X_val)[:,1]))
        return preds

    def get_model_predictions_for(self, key, X_val, t=0.5):
        if key == "Dummy":
            return np.repeat(False, len(X_val))
        if key == "Random":
            preds_prob = np.random.uniform(0,1,len(X_val))
            return (preds_prob >= t)

        model = self.data_register.get_model_for(key)
        preds = ((model.predict_proba(X_val)[:,1]) >= t)
        return preds
    

    def get_confusion_table_for(self, key, X_val, y_val):
        preds = self.get_model_predictions_for(key, X_val)
        # True Positives
        tp = (preds & y_val).sum()
        # False Positives
        fp = (preds & ~y_val).sum()
        # True Negatives
        tn = (~preds & ~y_val).sum()
        # False Negatives
        fn = (~preds & y_val).sum()
        df_confusion_table = pd.DataFrame([["+", tp, fn],["-",fp, tn]], columns=['', 'Guess +', 'Guess -'])
        return df_confusion_table
    
    def get_confusion_table_components(self, preds, y_val):
        # True Positives
        tp = (preds & y_val).sum()
        # False Positives
        fp = (preds & ~y_val).sum()
        # True Negatives
        tn = (~preds & ~y_val).sum()
        # False Negatives
        fn = (~preds & y_val).sum()

        return tp, fp, tn, fn

    def get_precision_recall(self, key, X_val, y_val):
        preds = self.get_model_predictions_for(key, X_val)
        # True Positives
        tp = (preds & y_val).sum()
        # False Positives
        fp = (preds & ~y_val).sum()
        # True Negatives
        tn = (~preds & ~y_val).sum()
        # False Negatives
        fn = (~preds & y_val).sum()
        # Actual positives
        ap = y_val.sum()

        precision = tp / (tp + fp)
        recall = tp/ap
        return precision, recall
    
    def get_f_scores(self,key, X_val, y_val, beta=None):
        precision, recall = self.get_precision_recall(key, X_val, y_val)
        if beta:
            # F beta score
            beta_score =(1+beta*beta) * ( (precision*recall)/(beta*beta)*(precision+recall) )
            return beta_score
        
        # F1 score
        return 2 * ( (precision*recall)/(precision+recall) )
    
    def get_positive_rates(self, key, X_val, y_val):
        preds = self.get_model_predictions_for(key, X_val)
        tp, fp, tn, fn = self.get_confusion_table_components(preds, y_val)
        # False positive rate (FPR): the fraction of false positives among all negative examples
        fpr = fp / (fp + tn)
        # True positive rate (TPR): the fraction of true positives among all positive examples
        tpr = tp / (tp + fn)

        return tpr, fpr
    
    def get_scores_dataframe(self, key, X_val, y_val):
        thresholds = np.linspace(0,1,101)
        scores = []
        for t in thresholds:
            preds = self.get_model_predictions_for(key, X_val, t=t)
            tp, fp, tn, fn = self.get_confusion_table_components(preds, y_val)            
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            scores.append((t, tp, fp, fn, tn, tpr, fpr))
        df = pd.DataFrame(scores)
        df.columns = ["Threshold", "tp","fp","fn","tn","tpr", "fpr"]
        return df
        
    def get_ideal_scores(self, y_val):
        pos_num = (y_val==1).sum()
        neg_num = (y_val==0).sum()
        # The ideal Values (0 or 1) 
        y_ideal = np.repeat([0,1],[neg_num, pos_num])
        # The ideal prediction probabilities, from 0, 0.1, 0.2... 1.0
        y_pred_ideal = np.linspace(0,1,neg_num + pos_num)

        # Build the score dataframe
        thresholds = np.linspace(0,1,101)
        scores = []
        for t in thresholds:
            preds = (y_pred_ideal >= t)
            tp, fp, tn, fn = self.get_confusion_table_components(preds, y_ideal)            
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            scores.append((t, tp, fp, fn, tn, tpr, fpr))
        df = pd.DataFrame(scores)
        df.columns = ["Threshold", "tp","fp","fn","tn","tpr", "fpr"]
        return df


    def train_logistic_regression(self, df, columns, y, C):
        cat = df[columns].to_dict(orient='records')
        dv = DictVectorizer(sparse=False)
        dv.fit(cat)
        X = dv.transform(cat)

        model = LogisticRegression(solver='liblinear', C=C) 
        model.fit(X, y)
        return dv, model

    def predict_logistic_regression(self, df, columns, dv, model):
        cat = df[columns].to_dict(orient='records')
        X = dv.transform(cat)

        y_pred = model.predict_proba(X)[:, 1]
        return y_pred

    def kfold_train_logistic_regression(self, nfolds, df, columns):    
        kfold = KFold(n_splits=nfolds, shuffle=True, random_state=1)
 
        for C in [0.001, 0.01, 0.1, 0.5, 1, 10]:
            aucs = []
            for train_idx, val_idx in kfold.split(df):
                X_train = df.iloc[train_idx]
                X_val  = df.iloc[val_idx]

                y_train = X_train.churn.values
                y_val = X_val.churn.values

                dv, model = self.train_logistic_regression(X_train,columns,y_train, C)
                y_pred = self.predict_logistic_regression(X_val, columns, dv, model)
                auc = roc_auc_score(y_val, y_pred)
                aucs.append(auc)
            
            print('C=%s, auc = %0.3f ± %0.3f' % (C, np.mean(aucs), np.std(aucs)))
        