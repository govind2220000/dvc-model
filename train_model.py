import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
Y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

model = LinearRegression()
model.fit(X_train,Y_train)

pickle.dump(model , open(str(Config.MODELS_PATH/"lr_model.pickle"),'wb'))