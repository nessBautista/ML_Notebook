from ast import arg
import pandas as pd

class DataRegister:
    """ Serves as helper by storing
    - dataframes: 
        - raw_dataframes
        - processed dataframes
    - Predictions
    - Metrics:
        - Accuracy
        - F-beta, Precision, Recall, etc
        - AUC
    """
    def __init__(self, datasets_path):     
        # The path to the datasets
        self.datasets_path = datasets_path
        # Raw dataframe registers
        self.raw_df_register = {}
        # processed dataframe registers: After Feature Engineering
        self.processed_df_register = {}
        # Register ML Models
        self.model_register = {}
        # The register associating short keys to a CSV file
        self.csv_register = {
            "telco_customers":"kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "diabetes":"kaggle/pima-indians-diabetes.data.csv"}
    
        # Register of processing functions
        self.processing_functions_register={}

    # Get Keys
    def get_csv_keys(self):
        keys = []
        for key in self.csv_register.keys():
            keys.append(key)
        return keys
    
    # Build a dataframe from a key
    def get_df_from_key(self, key):
        filepath = self.datasets_path + "/" + self.csv_register[key]
        df = pd.read_csv(filepath)
        return df

    # Register a raw dataframe
    def register_raw_dataframe(self, key, value):
        if not key in self.raw_df_register:
            self.raw_df_register[key] = value
            return True
        else:
            return False

    # Remove a raw dataframe
    def remove_raw_dataframe(self, key, value):
        if key in self.raw_df_register:
            self.raw_df_register[key] = None
            return True
        else:
            return False
    
    # Register a processed dataframe
    def register_processed_dataframe(self, key, value):
        if not key in self.processed_df_register:
            self.processed_df_register[key] = value
            return True
        else:
            return False

    # Remove a raw dataframe
    def remove_processed_dataframe(self, key, value):
        if key in self.processed_df_register:
            self.processed_df_register[key] = None
            return True
        else:
            return False
    
    # Register a function
    def register_processing_function(self, key, value):
        if not key in self.processing_functions_register:
            self.processing_functions_register[key] = value
            return True
        else:
            return False
    
    # Remove a function
    def remove_processing_function(self, key, value):
        if key in self.processing_functions_register:
            self.processing_functions_register[key] = None
            return True
        else:
            return False
    
    #Apply the function
    def apply_function(self, key, arguments):
        function = self.processing_functions_register[key]
        return function(**arguments)
        
    # Model register  
    def register_model(self,key, value):
        self.model_register[key] = value
    
    def get_model_for(self, key):
        return self.model_register[key]