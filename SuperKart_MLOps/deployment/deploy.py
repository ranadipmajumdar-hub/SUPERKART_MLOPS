
import os

from huggingface_hub import HfApi


# ============================================================
# CONFIGURATION
# ============================================================

SPACE_REPO = (
    "ranadipmajumdar/"
    "Superkart_MLOps"
)

SPACE_DIR = "space_deployment"


# ============================================================
# INITIALIZE API
# ============================================================

api = HfApi(
    token=os.environ["HF_TOKEN"]
)


# ============================================================
# VERIFY SPACE
# ============================================================

print(
    "Checking Hugging Face Space..."
)

space_info = api.space_info(
    repo_id=SPACE_REPO
)

print(
    "Space found:",
    space_info.id
)


# ============================================================
# UPLOAD SPACE FILES
# ============================================================

print(
    "\nUploading Streamlit application..."
)

api.upload_folder(
    folder_path=SPACE_DIR,
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo=""
)


print(
    "\n✅ Streamlit deployment completed."
)

print(
    "Space:"
)

print(
    f"https://huggingface.co/spaces/{SPACE_REPO}"
)
