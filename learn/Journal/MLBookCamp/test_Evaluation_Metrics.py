
from EvaluationMetrics import EvaluationMetrics
csv_path='./datasets/kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv'
def test_model_init():
    model = EvaluationMetrics(path=csv_path)
    assert model is not None

def test_model_raw_df():
    model = EvaluationMetrics(path=csv_path)
    assert model.raw_df is not None
