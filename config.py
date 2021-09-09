from pathlib import  Path
import os

if not os.path.exists("assets"):
    os.makedirs("assets")

class Config:
    RANDOM_SEED = 1234
    ASSETS_PATH = Path("./assets")
    ORIGINAL_DATASET_PATH = ASSETS_PATH / "original_dataset" / "Admission_Predict.csv"
    DATASET_PATH = ASSETS_PATH / "data"
    FEATURES_PATH = ASSETS_PATH/ "features"
    MODELS_PATH = ASSETS_PATH / "models"
    METRICS_FILE_PATH = ASSETS_PATH / "metrics.json"
