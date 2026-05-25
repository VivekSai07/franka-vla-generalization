import os
import sys
import json
from huggingface_hub import HfApi, login

# ==========================================
# CONFIGURATION — set these via environment variables in .env
# ==========================================
# Local path inside the container (mounted from host ~/vla/datasets/)
LOCAL_DATASET_DIR = os.getenv(
    "DATASET_DIR",
    "/vla/datasets/franka_libero_stacking_v1"
)

# The target Hugging Face repository  (<username>/<repo_name>)
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# Token read from environment — never hardcode
HF_TOKEN = os.getenv("HF_TOKEN", "")

def main():
    print("Starting dataset push...")

    if not HF_REPO_ID:
        print("ERROR: HF_REPO_ID env var is not set. Example: yourname/franka_libero_stacking_v1")
        sys.exit(1)

    # 1. Authentication
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("HF_TOKEN not set — falling back to cached credentials (run huggingface-cli login first)")

    if not os.path.exists(LOCAL_DATASET_DIR):
        print(f"ERROR: Dataset directory not found: {LOCAL_DATASET_DIR}")
        sys.exit(1)

    # 2. Read the codebase_version from info.json
    info_path = os.path.join(LOCAL_DATASET_DIR, "meta", "info.json")
    if not os.path.exists(info_path):
        print(f"ERROR: Could not find info.json at {info_path}")
        sys.exit(1)

    with open(info_path, "r") as f:
        info = json.load(f)

    # LeRobot expects data to be on a specific branch matching codebase_version (e.g., "v3.0")
    branch_name = info.get("codebase_version", "main")
    print(f"Detected LeRobot dataset version: {branch_name}")

    api = HfApi()

    # 3. Create the repo if it doesn't exist
    print(f"Verifying repository: {HF_REPO_ID}...")
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", private=True, exist_ok=True)

    # 4. Create the specific version branch
    try:
        api.create_branch(repo_id=HF_REPO_ID, repo_type="dataset", branch=branch_name, exist_ok=True)
        print(f"Branch '{branch_name}' ready.")
    except Exception as e:
        print(f"Branch note: {e}")

    # 5. Upload to that branch
    print(f"Uploading dataset to branch '{branch_name}'...")
    try:
        api.upload_folder(
            folder_path=LOCAL_DATASET_DIR,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=branch_name,
            commit_message=f"Upload LeRobot dataset ({branch_name})"
        )
        print(f"Upload complete. View at: https://huggingface.co/datasets/{HF_REPO_ID}/tree/{branch_name}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()