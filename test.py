import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set the folder containing kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = r"C:\Users\Jayesh\Downloads"

# Initialize API
api = KaggleApi()
api.authenticate()

# Dataset details
dataset = "emmarex/plantdisease"  # Plant Disease dataset
download_path = r"D:\My folder 2025\python for data science\projects\AI-Powered Crop Disease Detection & Yield Predictor\data\raw"

# Download and unzip
api.dataset_download_files(dataset, path=download_path, unzip=True)

print(f"Dataset downloaded and extracted to: {download_path}")
