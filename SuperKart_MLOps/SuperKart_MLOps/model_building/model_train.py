
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Product_Store_Sales_Total"

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_DIR = "models"


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training and testing data...")

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)

print(
    "Training shape:",
    train_df.shape
)

print(
    "Testing shape:",
    test_df.shape
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X_train = train_df.drop(
    columns=[TARGET]
)

y_train = train_df[TARGET]

X_test = test_df.drop(
    columns=[TARGET]
)

y_test = test_df[TARGET]


# ============================================================
# IDENTIFY FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# BASELINE MODELS
# ============================================================

models = {

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# TRAIN BASELINE MODELS
# ============================================================

results = []

trained_models = {}


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[
        model_name
    ] = pipeline

    print(
        f"{model_name} RMSE:",
        rmse
    )


# ============================================================
# RANDOM FOREST TUNING
# ============================================================

print(
    "\nStarting Random Forest tuning..."
)

rf_param_grid = {

    "model__n_estimators": [
        100,
        200,
        300
    ],

    "model__max_depth": [
        None,
        10,
        20,
        30
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ],

    "model__max_features": [
        "sqrt",
        "log2",
        1.0
    ]
}


rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_grid,
    n_iter=10,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1
)


rf_search.fit(
    X_train,
    y_train
)


rf_predictions = rf_search.best_estimator_.predict(
    X_test
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


results.append({
    "Model": "Tuned Random Forest",
    "MAE": rf_mae,
    "RMSE": rf_rmse,
    "R2": rf_r2
})


print(
    "Tuned Random Forest RMSE:",
    rf_rmse
)


# ============================================================
# XGBOOST TUNING
# ============================================================

print(
    "\nStarting XGBoost tuning..."
)

xgb_param_grid = {

    "model__n_estimators": [
        100,
        200,
        300
    ],

    "model__learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1
    ],

    "model__max_depth": [
        3,
        4,
        5,
        6
    ],

    "model__subsample": [
        0.7,
        0.8,
        1.0
    ],

    "model__colsample_bytree": [
        0.7,
        0.8,
        1.0
    ]
}


xgb_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_param_grid,
    n_iter=10,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1
)


xgb_search.fit(
    X_train,
    y_train
)


xgb_predictions = xgb_search.best_estimator_.predict(
    X_test
)

xgb_mae = mean_absolute_error(
    y_test,
    xgb_predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_predictions
    )
)

xgb_r2 = r2_score(
    y_test,
    xgb_predictions
)


results.append({
    "Model": "Tuned XGBoost",
    "MAE": xgb_mae,
    "RMSE": xgb_rmse,
    "R2": xgb_r2
})


print(
    "Tuned XGBoost RMSE:",
    xgb_rmse
)


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="RMSE"
).reset_index(
    drop=True
)


print(
    "\nFinal Model Comparison:"
)

print(
    results_df
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]


if best_model_name == "Tuned Random Forest":

    best_model = rf_search.best_estimator_

elif best_model_name == "Tuned XGBoost":

    best_model = xgb_search.best_estimator_

else:

    best_model = trained_models[
        best_model_name
    ]


best_predictions = best_model.predict(
    X_test
)

best_mae = mean_absolute_error(
    y_test,
    best_predictions
)

best_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        best_predictions
    )
)

best_r2 = r2_score(
    y_test,
    best_predictions
)


print(
    "\nSelected Model:",
    best_model_name
)

print(
    "MAE:",
    best_mae
)

print(
    "RMSE:",
    best_rmse
)

print(
    "R2:",
    best_r2
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "superkart_sales_model.joblib"
)

joblib.dump(
    best_model,
    model_path
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model_name": best_model_name,

    "target": TARGET,

    "mae": float(best_mae),

    "rmse": float(best_rmse),

    "r2": float(best_r2),

    "features": X_train.columns.tolist(),

    "numeric_features": numeric_features,

    "categorical_features": categorical_features,

    "random_state": 42
}


metadata_path = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)


with open(
    metadata_path,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print(
    "\n✅ Model training completed."
)

print(
    "Model saved:",
    model_path
)

print(
    "Metadata saved:",
    metadata_path
)
