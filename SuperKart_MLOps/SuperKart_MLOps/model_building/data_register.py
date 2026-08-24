
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "ranadipmajumdar/Superkart_MLOps-dataset"
repo_type = "dataset"

# Initialize API client
api = HfApi()

# Step 1: Check if the dataset repository exists
try:

    api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type
    )

    print(
        f"Dataset repository '{repo_id}' already exists. "
        "Using it."
    )

except RepositoryNotFoundError:

    print(
        f"Dataset repository '{repo_id}' not found. "
        "Creating new dataset repository..."
    )

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )

    print(
        f"Dataset repository '{repo_id}' created."
    )

# Step 2: Upload the dataset
api.upload_folder(
    folder_path="SuperKart_MLOps/data",
    repo_id=repo_id,
    repo_type=repo_type
)

print(
    "SuperKart dataset uploaded successfully."
)

print(
    f"Dataset URL:"
)

print(
    f"https://huggingface.co/datasets/{repo_id}"
)
