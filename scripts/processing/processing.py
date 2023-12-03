import glob
import numpy as np
import os
import argparse
import logging
import boto3
import pathlib
import pandas as pd
import logging
from pickle import dump,load

import tempfile
import joblib

from sklearn.preprocessing import LabelEncoder

from pickle import dump


from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


TARGET_COLUMN = "class"

if __name__ == "__main__":
    logger.debug("Starting preprocessing...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True, default='/opt/ml/preprocessing/input')
    parser.add_argument("--transformation-path", type=str, required=True)
    args = parser.parse_args()
    
    s3_client = boto3.client('s3')
    
    transformation_save_path = args.transformation_path
    
    bucket_name=transformation_save_path.split("/")[2]
    print(f"transformation save bucket_name = {bucket_name}")

    transformations_key = os.path.join(*transformation_save_path.split("/")[3:]) + "/transformations"
    print(f"transformation save key = {transformations_key}")

    # loading of input data from S3 bucket passed as --input-data argument
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    fn = f"{base_dir}/data/loaded_dataset.csv" # change this is you are using data different than .csv 
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)
    
    # reading loaded input data
    df = pd.read_csv(fn) # change this is you are using data different than .csv 
    os.unlink(fn)
    
    # PREPROCESSING PART HERE

    # dropping not needed columns
    df.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True)

    df_temp = df

    # encode class labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_temp['class'])
    df_temp['class'] = y_encoded
    
    dump(le, open('le.pkl', 'wb'))

    # WRITE LABEL_ENCODER TO S3
    dump(le, open('le.pkl', 'wb'))
    
    le_key = "Label_encoder"
    
    s3_client.upload_file("le.pkl", bucket_name, le_key)
    
    print(f"transformation save le_key = {le_key}")
        
    scaler_key = f"{transformations_key}/scaler_transformation.pkl"
    
    df = df_temp

    X_train, X_test, y_train, y_test = train_test_split(df.drop(TARGET_COLUMN, axis=1), df[TARGET_COLUMN], test_size = 0.33)

    X_train.insert(0, "target", y_train)
    X_test.insert(0, "target", y_test)

    logger.info("Saving datasets to %s", base_dir)
    pd.DataFrame(X_train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(X_test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

