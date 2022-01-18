# %%
from audioop import add
from re import A
from MLBookCamp.EvaluationMetrics import EvaluationMetrics


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
evaluation_metrics = EvaluationMetrics(path='../../datasets/kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv')
raw_df = evaluation_metrics.raw_df

# Analyze how imbalance is the dataset
neg, pos = np.bincount(raw_df["Churn"]=="Yes")
total = neg + pos 
print("examples:\n    Total: {}\n    Positive: {}  ({:.2f}% of total)\n".format(total, pos, 100*pos/total))
test_df = raw_df.copy()

def plot_positives_against_negatives(df, target_column, positive_val,feature1, feature2):
    #Set target column to Boolean Values
    df[target_column] = df[target_column]== positive_val
    #Make sure feature columns are numeric
    df[feature1] = pd.to_numeric(df[feature1], errors="coerce")
    
    #df[feature1] = df[feature1].fillna(1)
    #df[feature1]=df[feature1].mask(df[feature1]==0).fillna(df[feature1].mean())
    for val in df[feature1]:
        if val == 0:
            print(val)
    #print(f'zeros:{(df[feature1])}')
    df[feature2] = pd.to_numeric(df[feature2], errors="coerce")
    
    #df[feature2] = df[feature2].fillna(1)
    #df[feature2] = df[feature2].replace(to_replace=0, value=1)

    # Obtain positive and negative values
    mask_array = np.array(df[target_column])
    df_pos = pd.DataFrame(df[mask_array], columns=df.columns)
    df_neg = pd.DataFrame(df[~mask_array], columns=df.columns)
    print(df_pos[feature1].max(), df_neg[feature1].max())
    print(df_pos[feature2].max(), df_neg[feature2].max())
    #plot distributions
    sns.jointplot(x=df_pos[feature1], y=df_pos[feature2], kind="hex")
    plt.suptitle("Positive distribution")
    sns.jointplot(x=df_neg[feature1], y=df_neg[feature2], kind="hex")
    plt.suptitle("Negative distribution")
    
    
plot_positives_against_negatives(test_df,"Churn", "Yes", "MonthlyCharges", "TotalCharges")

# %%
from MLBookCamp.BinaryEvaluationMetrics import BinaryEvaluationMetrics
import pandas as pd
datasets_path = "../../datasets"

# Create binary evaluation object
binary_eval = BinaryEvaluationMetrics(datasets_path)

# Get Telco dataframe to get started
raw_df = binary_eval.data_register.get_df_from_key('telco_customers')
# Register the raw dataframe as reference
binary_eval.data_register.register_raw_dataframe('telco_customers', raw_df)
raw_df.head()

# Define processing function
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


binary_eval.data_register.register_processing_function("process_func_telco", preprocess_dataframe)

# NOw process raw data
telco_args={"raw_df":raw_df}
processed_df = binary_eval.data_register.apply_function("process_func_telco", telco_args)
# Also register the processed data frame for reference
binary_eval.data_register.register_processed_dataframe("processed_df_telco", processed_df)
processed_df.head()




