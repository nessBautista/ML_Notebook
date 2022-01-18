
from curses.ascii import EM
from genericpath import exists
from queue import Empty
from EvaluationMetrics import EvaluationMetrics
from DataRegister import DataRegister
import pandas as pd
# TODO: Make this an enviroment variable
datasets_path = './datasets'

csv_path='./datasets/kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv'
def test_model_init():
    model = EvaluationMetrics(path=csv_path)
    assert model is not None

def test_model_raw_df():
    model = EvaluationMetrics(path=csv_path)
    assert model.raw_df is not None

# Data register test
def test_data_register_init():
    dr = DataRegister(datasets_path)

def test_data_register_get_csv_keys():
    dr = DataRegister(datasets_path)
    keys = dr.get_csv_keys()
    assert keys is not Empty

def test_data_register_retrieve_dataframe_from_key():
    key = "diabetes"
    dr = DataRegister(datasets_path)
    df = dr.get_df_from_key(key)
    df is not None

def test_data_register_add_raw_dataframe():
    dr = DataRegister(datasets_path)
    key = "diabetes"
    raw_df = dr.get_df_from_key(key)
    result = dr.register_raw_dataframe(key, raw_df)
    assert result is True

def test_data_try_double_registration_of_raw_df():
    dr = DataRegister(datasets_path)
    key = "diabetes"
    raw_df = dr.get_df_from_key(key)
    result = dr.register_raw_dataframe(key, raw_df)
    result = dr.register_raw_dataframe(key, raw_df)
    assert result is False

def test_data_register_remove_raw_df():
    dr = DataRegister(datasets_path)
    key = "diabetes"
    raw_df = dr.get_df_from_key(key)
    result = dr.register_raw_dataframe(key, raw_df)
    result = dr.remove_raw_dataframe(key, raw_df)
    assert result is True

def test_data_register_add_processed_dataframe():
    dr = DataRegister(datasets_path)
    key = "diabetes"
    raw_df = dr.get_df_from_key(key)
    result = dr.register_processed_dataframe(key, raw_df)
    assert result is True

def test_data_register_remove_processed_df():
    dr = DataRegister(datasets_path)
    key = "diabetes"
    raw_df = dr.get_df_from_key(key)
    result = dr.register_processed_dataframe(key, raw_df)
    result = dr.remove_processed_dataframe(key, raw_df)
    assert result is True

# Processing functions
def test_data_register_add_processing_function():
    # Get dr object
    dr = DataRegister(datasets_path)
    # Register raw dataframe
    key = "AB_dataframe"
    test_df = pd.DataFrame([[1, 1],[2,2]], columns=["A", "B"])
    dr.register_raw_dataframe(key, test_df)
    
    # Define processing function and register
    def addColumn(**kwargs):
        kwargs["df"][kwargs["column_name"]]= kwargs["column_values"]
        return kwargs["df"]
    dr.register_processing_function("addColumn", addColumn)
    
    new_df = dr.processing_functions_register["addColumn"](df=test_df,column_values=[3,3], column_name="C")
    assert new_df["C"] is not Empty

def test_data_register_apply_processing():
    dr = DataRegister(datasets_path)
    # Create test dataframe
    test_df = pd.DataFrame([[1, 1],[2,2]], columns=["A", "B"])
    # define processing function and register
    def addColumn(df, column_name, column_values):
            df["column_name"]= column_values
            return df
    dr.register_processing_function("addColumn", addColumn)
    # Prepare arguments and apply processing
    args={'df':test_df, 'column_name':'C', 'column_values':[4,5]}
    processed_df = dr.apply_function("addColumn", args)
    assert processed_df is not Empty