# %%
from MLBookCamp.EvaluationMetrics import EvaluationMetrics
import pandas as pd
from sklearn.metrics import accuracy_score

evaluation_metrics = EvaluationMetrics(path='../../datasets/kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv')

preprocess_df = evaluation_metrics.preprocess_dataframe()

# Split Data
X_train, X_val, X_test, y_train, y_val, y_test = evaluation_metrics.split_data(preprocess_df,evaluation_metrics.categorical+evaluation_metrics.numerical)

# Split Data with reduced number of features
small_subset = ['contract', 'tenure', 'totalcharges']
X_small_train, X_small_val, X_small_test, y_train, y_val, y_test = evaluation_metrics.split_data(preprocess_df,small_subset)
    
# Build 4 models:
# Logistic Regression
model_lr= evaluation_metrics.get_Logistic_Regression(X_train, y_train)
# Logistic Regression with small set of features
model_small_lr= evaluation_metrics.get_Logistic_Regression(X_small_train, y_train)
# Decision Tree
model_tree = evaluation_metrics.get_decision_tree(X_train, y_train)
# Dummy Model
model_dummy = evaluation_metrics.DummyModel()

# %%
# Accuracy
y_pred = model_lr.predict_proba(X_val)[:, 1]      
churn = y_pred >= 0.5

y_pred_small = model_small_lr.predict_proba(X_small_val)[:, 1]      
churn_small = y_pred_small >= 0.5                                

y_pred_dummy = model_dummy.predict_proba(y_val)
churn_dummy = y_pred_dummy >= 0.5                      

y_pred_tree = model_tree.predict_proba(X_val)[:, 1]      
churn_tree = y_pred_tree >= 0.5

model_accuracy_results = [['LogisticRegression', accuracy_score(y_val, churn)],
['SmallLogisticRegression', accuracy_score(y_val, churn_small)],
['Dummy', accuracy_score(y_val, churn_dummy)],
['DecisionTree', accuracy_score(y_val, churn_tree)]]
df_model_accuracies = pd.DataFrame(model_accuracy_results, columns=['Model','Accuracy'])
df_model_accuracies


