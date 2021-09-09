import pandas as pd
from config import Config

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
test_df = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))

def get_features(df):
    return df.iloc[:,:-1]    #x = df.iloc[:,:-1]


train_features = get_features(train_df)
test_features = get_features(test_df)

train_features.to_csv(str(Config.FEATURES_PATH/"train_features.csv"),index=None)
test_features.to_csv(str(Config.FEATURES_PATH/"test_features.csv"),index=None)

train_df.iloc[:,-1].to_csv(str(Config.FEATURES_PATH/"train_labels.csv"),index=None)
test_df.iloc[:,-1].to_csv(str(Config.FEATURES_PATH/"test_labels.csv"),index=None)