# %%
from nis import cat
from MLBookCamp.BinaryEvaluationMetrics import BinaryEvaluationMetrics
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
from sklearn.metrics import roc_auc_score
# The path to the dataset
datasets_path = "../../datasets"
# Columns divided by categorical and numerical
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
# All numerical columns
numerical = ['tenure', 'monthlycharges', 'totalcharges']
# Create binary evaluation object
binEval =  BinaryEvaluationMetrics(datasets_path)

# Get raw_dataframe
df_raw = binEval.data_register.get_df_from_key("telco_customers")
df_raw.head()

# Processed Telco customer df
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
df_proc = preprocess_dataframe(df_raw)
df_proc.head()

# Find which is the best C
binEval.kfold_train_logistic_regression(5, df_proc,categorical+numerical)

# use the best C

# split data
from sklearn.model_selection import train_test_split
df_X = df_proc[categorical+numerical]
df_y = df_proc.churn
X_train, X_test, y_train, y_test = train_test_split(df_X,df_y, test_size=0.33)

# train model
dv, model = binEval.train_logistic_regression(X_train, categorical+numerical, y=y_train,C=0.5)

# predict model
y_preds = binEval.predict_logistic_regression(X_test, categorical+numerical, dv, model)
auc = roc_auc_score(y_test, y_preds)
auc



