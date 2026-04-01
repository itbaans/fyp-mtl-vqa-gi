import os
import sys
import subprocess
import shutil

# Replace this with your actual Kaggle dataset ID (e.g., "username/dataset-name")
KAGGLE_DATASET = "itbaansawan/fyp-gi-vqa-mtl-data"

def ensure_dependencies():
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        
    try:
        import dotenv
    except ImportError:
        print("Installing python-dotenv...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])

def download_data():
    from dotenv import load_dotenv
    import kagglehub
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if credentials are in the environment
    has_creds = 'KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ
    
    if not has_creds:
        print("Checking authentication... (If this is a private dataset, your browser will open to log you in)")
        try:
            kagglehub.login()
        except Exception as e:
            print("Login skipped or failed (if the dataset is public, this is fine).")
    else:
        print("Authenticating automatically using credentials found in .env!")

    print(f"Downloading dataset '{KAGGLE_DATASET}' via KaggleHub...")
    
    # KaggleHub downloads to a global cache.
    cached_path = kagglehub.dataset_download(KAGGLE_DATASET)
    
    # Move the contents of the cache directly into our local ./data directory
    local_data_dir = os.path.join(os.getcwd(), "data")
    
    print(f"Data downloaded to cache: {cached_path}")
    print(f"Copying files to local repository directory: {local_data_dir}...")
    
    # Copy from cache to our data folder
    os.makedirs(local_data_dir, exist_ok=True)
    shutil.copytree(cached_path, local_data_dir, dirs_exist_ok=True)
    
    print("Download and extraction complete. Data is ready in the './data' directory!")

if __name__ == "__main__":
    if KAGGLE_DATASET == "username/dataset-name":
        print("Please edit download_data.py and set 'KAGGLE_DATASET' to your actual Kaggle dataset ID!")
        sys.exit(1)
        
    ensure_dependencies()
    download_data()
