# For data manipulation
import pandas as pd
import sklearn
# For folder creation
import os
# For data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hf space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for dataset and output paths
api = HfApi(token = os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/ani-maddy98/tourism-package-prediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("dataset loaded successfully")

# Clean the Gender variable
tourism_dataset['Gender'] = tourism_dataset['Gender'].replace('Fe Male','Female')

# Clean the Marital Status variable
tourism_dataset["MaritalStatus"] = tourism_dataset["MaritalStatus"].replace("Unmarried","Single")

# Define target variable for prediction task
target = "ProdTaken"

# List of numeric features in the dataset
numeric_features = [
    'Age',
    'MonthlyIncome',
    'NumberOfPersonVisiting',
    'NumberOfTrips',
    'NumberOfChildrenVisiting'
]

# List of categorical features in the dataset
categorical_features = [
    'CityTier',
    'Occupation',
    'Gender',
    'ProductPitched',
    'PreferredPropertyStar',
    'MaritalStatus',
    'Passport',
    'OwnCar',
    'Designation'
]

# Define predictor matrix (X) using selected numerical, and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 42)

X_train.to_csv("Xtrain.csv")
X_test.to_csv("Xtest.csv")
y_train.to_csv("ytrain.csv")
y_test.to_csv("ytest.csv")

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="ani-maddy98/tourism-package-prediction",
        repo_type="dataset",
    )
