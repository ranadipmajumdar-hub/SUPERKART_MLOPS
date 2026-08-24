
import os

from huggingface_hub import HfApi


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_REPO = (
    "ranadipmajumdar/"
    "Superkart_MLOps-model"
)

MODEL_DIR = "models"


# ============================================================
# INITIALIZE API
# ============================================================

api = HfApi(
    token=os.environ["HF_TOKEN"]
)


# ============================================================
# VERIFY MODEL FILES
# ============================================================

model_file = os.path.join(
    MODEL_DIR,
    "superkart_sales_model.joblib"
)

metadata_file = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)


if not os.path.exists(model_file):

    raise FileNotFoundError(
        "Trained model was not found."
    )


if not os.path.exists(metadata_file):

    raise FileNotFoundError(
        "Model metadata was not found."
    )


# ============================================================
# UPLOAD MODEL
# ============================================================

print(
    "Uploading trained model..."
)

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="superkart_sales_model.joblib",
    repo_id=MODEL_REPO,
    repo_type="model"
)


# ============================================================
# UPLOAD METADATA
# ============================================================

print(
    "Uploading model metadata..."
)

api.upload_file(
    path_or_fileobj=metadata_file,
    path_in_repo="model_metadata.json",
    repo_id=MODEL_REPO,
    repo_type="model"
)


print(
    "\n✅ Model registration completed successfully."
)

print(
    "Model repository:"
)

print(
    f"https://huggingface.co/{MODEL_REPO}"
)
