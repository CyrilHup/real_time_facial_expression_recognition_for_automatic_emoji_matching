"""
Script to download and prepare LeapGestRecog dataset from Kaggle
=================================================================
"""

import os
import zipfile
from pathlib import Path


def download_leapgestrecog():
    """
    Download LeapGestRecog dataset from Kaggle.
    
    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Setup Kaggle API credentials:
       - Go to https://www.kaggle.com/settings
       - Click "Create New API Token"
       - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)
    """
    
    print("="*70)
    print("  LEAPGESTRECOG DATASET DOWNLOADER")
    print("="*70)
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("✓ Kaggle API installed")
    except ImportError:
        print("✗ Kaggle API not installed")
        print("\nPlease install: pip install kaggle")
        return
    
    # Check if credentials exist
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print(f"✗ Kaggle credentials not found at {kaggle_json}")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
        return
    
    print(f"✓ Kaggle credentials found")
    
    # Create data directory
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    output_dir = data_dir / 'leapGestRecog'
    
    # Check if already downloaded
    if output_dir.exists() and len(list(output_dir.glob('*'))) > 0:
        print(f"\n✓ Dataset already exists at {output_dir}")
        print(f"  Found {len(list(output_dir.glob('*')))} items")
        
        response = input("\nRe-download? (y/n): ").lower()
        if response != 'y':
            print("Using existing dataset.")
            return
    
    print(f"\nDownloading to: {output_dir}")
    print("This may take a few minutes...")
    
    try:
        # Download dataset
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print("\nDownloading leapgestrecog dataset...")
        api.dataset_download_files(
            'gti-upm/leapgestrecog',
            path=str(data_dir),
            unzip=True
        )
        
        print("✓ Download complete!")
        
        # Verify structure
        if output_dir.exists():
            folders = [f for f in output_dir.iterdir() if f.is_dir()]
            print(f"\n✓ Dataset extracted successfully")
            print(f"  Found {len(folders)} class folders:")
            for folder in sorted(folders)[:5]:
                num_images = len(list(folder.glob('*.png')))
                print(f"    - {folder.name}: {num_images} images")
            if len(folders) > 5:
                print(f"    ... and {len(folders) - 5} more")
        
        print("\n" + "="*70)
        print("DATASET READY!")
        print("="*70)
        print(f"Location: {output_dir}")
        print("\nNext steps:")
        print("  1. Run: python train_hand_gesture.py")
        print("  2. Wait for training to complete (~30-60 min)")
        print("  3. Model will be saved to ./checkpoints/hand_gesture_model.pth")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/gti-upm/leapgestrecog")
        print("2. Click 'Download' button")
        print(f"3. Extract to: {output_dir}")


if __name__ == "__main__":
    download_leapgestrecog()
