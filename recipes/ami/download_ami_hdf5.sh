#!/bin/bash

pip install huggingface_hub tqdm -q

python3 - <<EOF
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm
import os

REPO_ID = "hcliu/ami-fcasa"
TARGET_DIR = os.path.dirname(__file__)  # current dir recipes/ami

# available subfolders: ["hdf5", "cv", "tr", "tt"]
TARGET_SUBFOLDERS = ["hdf5", "tt"] # hdf5 preprocessed for cv and tr; .wavs for tt.

all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")

for subfolder in TARGET_SUBFOLDERS:
    print(f"\nDownloading files in: {subfolder}")
    files_to_download = [f for f in all_files if f.startswith(subfolder)]

    for file_path in tqdm(files_to_download, desc=subfolder):
        local_path = os.path.join(TARGET_DIR, file_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=file_path,
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False,
            cache_dir=None
        )
EOF