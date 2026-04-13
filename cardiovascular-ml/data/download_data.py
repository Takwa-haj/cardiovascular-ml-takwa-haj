"""
download_data.py
================
Script to download all datasets for the CVD ML project.
Author: Takwa Haj

Usage:
    python data/download_data.py

This script automatically downloads the UCI Heart Disease dataset.
For Framingham and Kaggle CVD datasets, it prints manual download instructions.
"""

import os
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download_uci():
    """Download UCI Heart Disease dataset directly from UCI ML Repository."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    dest = os.path.join(DATA_DIR, "uci_heart.csv")

    if os.path.exists(dest):
        print(f"  ✅ UCI already exists: {dest}")
        return

    print("  Downloading UCI Heart Disease dataset...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✅ Saved to {dest}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        print(f"     Manual URL: {url}")


def print_kaggle_instructions():
    """Print manual download instructions for Kaggle datasets."""
    print("""
  ── Framingham Heart Study ───────────────────────────────────────
  1. Go to: https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset
  2. Click Download
  3. Rename the file to: framingham.csv
  4. Place it in this folder: data/

  ── Kaggle CVD Dataset (70,000 patients) ─────────────────────────
  1. Go to: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
  2. Click Download
  3. Extract the zip — you'll get: cardio_train.csv
  4. Place it in this folder: data/

  💡 Tip: Install the Kaggle CLI for faster downloads:
     pip install kaggle
     kaggle datasets download aasheesh200/framingham-heart-study-dataset
     kaggle datasets download sulianova/cardiovascular-disease-dataset
""")


if __name__ == "__main__":
    print("\n🫀 CVD ML Project — Dataset Download Script")
    print("=" * 50)

    print("\n[1/3] UCI Heart Disease (auto-download):")
    download_uci()

    print("\n[2/3] Framingham Heart Study (manual):")
    print_kaggle_instructions()

    print("\n[3/3] Checking data folder contents:")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if files:
        for f in files:
            size = os.path.getsize(os.path.join(DATA_DIR, f)) // 1024
            print(f"  📄 {f} ({size} KB)")
    else:
        print("  ⚠️  No CSV files found yet — follow instructions above")

    print("\n✅ Done. Place all CSV files in the data/ folder, then run the notebooks.")
