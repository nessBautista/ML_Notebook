"""
Study of Binary Evaluation Metrics:
    
"""
# %%
from cProfile import label
from nis import cat
from sys import displayhook
import pandas as pd
import numpy as np
from MLBookCamp.BinaryEvaluationMetrics import BinaryEvaluationMetrics
import matplotlib.pyplot as plt
%matplotlib inline

# The path to the dataset
datasets_path = "../../datasets"
# Create binary evaluation object
helper =  BinaryEvaluationMetrics(datasets_path)

# Get raw_dataframe
raw_df = helper.data_register.get_df_from_key("telco_customers")
raw_df.head()

# Define a processing function
def preprocess_dataframe(raw_df):
    df = raw_df
    # TotalCharges must be numeric
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce")
    # Fill NAs with 0
    df.TotalCharges = df.TotalCharges.fillna(0)
    # Let’s make it uniform by lowercasing everything and replacing spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # Our target Column Churn should be numeric as well
    df.churn = (df.churn=="Yes").astype(int)
    #displayhook(df.head().T)
    print(f'Churn Values:{df.churn.value_counts()}')

    return df

# apply this function
processed_df = preprocess_dataframe(raw_df)
processed_df.head()

# Now lets define some constants
# All categorical columns, except the target: Churn
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
# All numerical columns
numerical = ['tenure', 'monthlycharges', 'totalcharges']

# Small subset of colums to train a smaller model
small_subset = ['contract', 'tenure', 'totalcharges']

# We can proceed to split the data

# Regular split
X_train, X_val, X_test, y_train, y_val, y_test = helper.split_data(processed_df,categorical+numerical)
print(X_train.shape)
# Small split
X_train_small, X_val_small, X_test_small, y_train_small, y_val_small, y_test_small = helper.split_data(processed_df,small_subset)
print(X_train_small.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# With data in place we can now build our models 
# to compare Evaluation metrics
lr_model = LogisticRegression(solver='liblinear', random_state=1)
lr_model.fit(X_train, y_train)
helper.add_model("LogisticRegression", lr_model)

small_lr_model = LogisticRegression(solver='liblinear', random_state=1)
small_lr_model.fit(X_train_small, y_train_small)
helper.add_model("SmallLogisticRegression", small_lr_model)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
helper.add_model("DecisionTree", tree_model)

# Get predictions
# Logistic Regression
lr_model_preds = helper.get_model_predictions_for("LogisticRegression", X_val)

# Small Logistic Regression
small_lr_model_preds = helper.get_model_predictions_for("SmallLogisticRegression", X_val_small)

# Tree Predictions
tree_model_preds = helper.get_model_predictions_for("DecisionTree", X_val)

#Dummy Predictions
dummy_preds = helper.get_model_predictions_for("Dummy", X_val)


accuracies = [
    ["Logistic Regression", accuracy_score(y_val, lr_model_preds)],
    ["SmallLogisticRegression", accuracy_score(y_val, small_lr_model_preds)],
    ["DecisionTree", accuracy_score(y_val, tree_model_preds)],
    ["Dummy", accuracy_score(y_val, dummy_preds)],
            ]
df_accuracies = pd.DataFrame(accuracies, columns=['Model','Accuracy'])
df_accuracies

# Build Confusion tables for the logistic regression Model
# Remember we are using the validation set in each model
# Lets focus on the Logistic Regression Model
df_cf_lr = helper.get_confusion_table_for("LogisticRegression", X_val, y_val)
df_cf_lr
df_cf_small_lr = helper.get_confusion_table_for("SmallLogisticRegression", X_val_small, y_val_small)
df_cf_small_lr

precision, recall = helper.get_precision_recall("LogisticRegression", X_val, y_val)
print(f'precision:{precision}, recall:{recall}')
f1_score = helper.get_f_scores("LogisticRegression", X_val, y_val)
print(f'f1 score:{f1_score}')
f5_score = helper.get_f_scores("LogisticRegression", X_val, y_val, 10)
print(f'f5_score:{f5_score}')
f05_score = helper.get_f_scores("LogisticRegression", X_val, y_val, 0.5)
print(f'f0.5_score:{f05_score}')

# ROC

# To calculate the ROC (Receiver Operating Characteristc)
# We need the positive rates 
tpr, fpr = helper.get_positive_rates("LogisticRegression", X_val, y_val)
print(f'FPR:{fpr}, TPR:{tpr}')

# The positive rates thoughout different thresholds
df_scores = helper.get_scores_dataframe("LogisticRegression", X_val, y_val)
displayhook(df_scores)
#plt.plot(df_scores.Threshold,df_scores.tpr, label="TPR")
#plt.plot(df_scores.Threshold,df_scores.fpr, label="FPR")
#plt.legend()

# Now lets compare with the scores of a Random model
df_scores_random = helper.get_scores_dataframe("Random", X_val, y_val)
#plt.plot(df_scores_random.Threshold,df_scores_random.tpr, label="TPR")
#plt.plot(df_scores_random.Threshold,df_scores_random.fpr, label="FPR")
#plt.legend()

# Compare with the ideal model
df_ideal_scores = helper.get_ideal_scores(y_val)
#plt.plot(df_ideal_scores.Threshold,df_ideal_scores.tpr, label="TPR")
#plt.plot(df_ideal_scores.Threshold,df_ideal_scores.fpr, label="FPR")
#plt.legend()

# ROC Curve
#plt.plot(df_scores.fpr, df_scores.tpr, label="Model")
#plt.plot(df_scores_random.fpr, df_scores_random.tpr, label="Random")
#plt.plot(df_ideal_scores.fpr, df_ideal_scores.tpr, label="Ideal")
#plt.legend()

# ROC comparison
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# Compare ROC of the Logistic Regression, small Logistic Regression and Decision tree
lr_model_preds = helper.get_model_raw_predictions_for("LogisticRegression", X_val) 
lr_small_model_preds = helper.get_model_raw_predictions_for("SmallLogisticRegression", X_val_small) 
tree_model_preds = helper.get_model_raw_predictions_for("DecisionTree", X_val) 

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_val, lr_model_preds)
fpr_small, tpr_small, thresholds_small = roc_curve(y_val, lr_small_model_preds)
fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_val, tree_model_preds)

plt.plot(fpr_lr, tpr_lr, c='b')
plt.plot(fpr_small, tpr_small, c='g')
plt.plot(fpr_tree, tpr_tree, c='orange')

# You can also get the AUC
print(f'Logistic Regression AUC:{auc(fpr_lr, tpr_lr)}')
print(f'Small Logistic Regression AUC:{auc(fpr_small, tpr_small)}')
print(f'Decision Tree AUC:{auc(fpr_tree, tpr_tree)}')


# COmparing AUC with Probabilities
neg = lr_model_preds[y_val==0]
pos = lr_model_preds[y_val==1]
np.random.seed(1)
neg_choice = np.random.randint(low=0, high=len(neg), size=10000)
pos_choice = np.random.randint(low=0, high=len(pos), size=10000)
(pos[pos_choice] > neg[neg_choice]).mean()


