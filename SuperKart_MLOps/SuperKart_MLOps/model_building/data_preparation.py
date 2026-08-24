
import os
import pandas as pd

from huggingface_hub import hf_hub_download

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_REPO = (
    "ranadipmajumdar/"
    "Superkart_MLOps-dataset"
)

TARGET = "Product_Store_Sales_Total"

OUTPUT_DIR = "data/processed"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DOWNLOAD RAW DATA FROM HUGGING FACE
# ============================================================

print("Downloading raw dataset from Hugging Face...")

raw_file = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="SuperKart.csv",
    repo_type="dataset"
)

print("Raw dataset downloaded.")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    raw_file
)

print(
    "Original dataset shape:",
    df.shape
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()


# ============================================================
# HANDLE MISSING PRODUCT WEIGHT
# ============================================================

df["Product_Weight"] = df[
    "Product_Weight"
].fillna(
    df["Product_Weight"].median()
)


# ============================================================
# REMOVE UNNECESSARY IDENTIFIER COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "Product_Id",
        "Store_Id"
    ],
    errors="ignore"
)


# ============================================================
# REMOVE MISSING TARGET VALUES
# ============================================================

df = df.dropna(
    subset=[TARGET]
)


print(
    "Cleaned dataset shape:",
    df.shape
)


# ============================================================
# SPLIT FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# CREATE TRAIN DATASET
# ============================================================

train_df = X_train.copy()

train_df[TARGET] = y_train.values


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_df = X_test.copy()

test_df[TARGET] = y_test.values


# ============================================================
# SAVE DATASETS
# ============================================================

train_path = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)

train_df.to_csv(
    train_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


print("\nTraining data saved:")
print(train_path)

print("\nTesting data saved:")
print(test_path)

print("\nData preparation completed successfully.")
