import os
import json
import subprocess
import sys
import pathlib
import tarfile
import pickle

# import proper metric
from sklearn.metrics import mean_squared_error

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("xgboost")
install("pandas")

import xgboost
import pandas as pd

if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    model = pickle.load(open("xgboost-model", "rb"))

    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    
    X_test = xgboost.DMatrix(df.values)

    predictions = model.predict(X_test).round()

    # change metric here
    score = mean_squared_error(y_test, predictions)
    
    print("\nTest score :", score)

    # Change metric here
    report_dict = {
            "regression_metrics": {
                "mse": {"value": score, "standard_deviation": "NaN"},
            },
        }


    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

