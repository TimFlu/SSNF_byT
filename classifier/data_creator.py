import pandas as pd
import os

def main():
    # Read in preprocessed data
    data_eb_test = pd.read_parquet("SSNF2/preprocess/data_eb_test.parquet")
    data_eb_train = pd.read_parquet("SSNF2/preprocess/data_eb_train.parquet")
    mc_eb_test = pd.read_parquet("SSNF2/preprocess/mc_eb_test.parquet")
    mc_eb_train = pd.read_parquet("SSNF2/preprocess/mc_eb_train.parquet")

    # create label column
    data_eb_test["label"] = ["data"] * data_eb_test.shape[0]
    data_eb_train["label"] = ["data"] * data_eb_train.shape[0]
    mc_eb_test["label"] = ["mc"] * mc_eb_test.shape[0]
    mc_eb_train["label"] = ["mc"] * mc_eb_train.shape[0]

    # delete weights column
    del data_eb_test["weight"]
    del data_eb_train["weight"]
    del mc_eb_test["weight"]
    del mc_eb_train["weight"]

    # concat the training and test data
    test_data = pd.concat([data_eb_test, mc_eb_test], axis=0)
    train_data = pd.concat([data_eb_train, mc_eb_train], axis=0)

    # save data in folder as parquet
    folder_name = "SSNF2/classifier/data/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created succesfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    
    test_data.to_parquet("SSNF2/classifier/data/test_data.parquet")
    train_data.to_parquet("SSNF2/classifier/data/train_data.parquet")

if __name__ == "__main__":
    main()