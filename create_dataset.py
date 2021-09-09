import gdown
import pandas as pd
from config import Config
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


np.random.seed(Config.RANDOM_SEED)

Config.ORIGINAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

gdown.download(
    "https://drive.google.com/uc?id=1Mt9xtyINCVwGotGnkb3N4Yq3gAmsU3sV",
    str(Config.ORIGINAL_DATASET_PATH),
)
dataset = pd.read_csv(str(Config.ORIGINAL_DATASET_PATH),sep=r'\s*,\s*',
                           header=0, encoding='ascii', engine='python')

dataset.drop(columns=['Serial No.'],inplace=True)
scaler = StandardScaler()
#lst = ["GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research"]
'''for i in lst:
    df[i] = df[i].fit_transform(df[i])'''
df_copy = scaler.fit_transform(dataset[["GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research"]])

df = pd.DataFrame({'GRE Score': df_copy[:, 0], 'TOEFL Score': df_copy[:, 1], 'University Rating': df_copy[:, 2], 'SOP': df_copy[:, 3], 'LOR': df_copy[:, 4], 'CGPA': df_copy[:, 5], 'Research': df_copy[:, 6]})
df['Chance Of Admit'] = dataset["Chance of Admit"]


#print(df_copy.index)
#print(df.index)
#df_copy["Chance of Admit"] = df["Chance of Admit"]

df_train , df_test = train_test_split(df,test_size=0.2, random_state=Config.RANDOM_SEED,)

df_train.to_csv(str(Config.DATASET_PATH/"train.csv"),index=None)
df_test.to_csv(str(Config.DATASET_PATH/"test.csv"),index=None)

