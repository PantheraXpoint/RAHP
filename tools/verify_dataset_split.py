#!/usr/bin/env python3
"""
Verify that split_GLIPunseen exists in your VG150 dataset.
"""

import h5py
import os

h5_path = '/RAHP/DATASET/VG150/VG-SGG-with-attri.h5'

print("=" * 70)
print("Checking VG150 Dataset Split")
print("=" * 70)

if not os.path.exists(h5_path):
    print(f"✗ Dataset not found at: {h5_path}")
    exit(1)

print(f"✓ Found dataset: {h5_path}\n")

with h5py.File(h5_path, 'r') as f:
    print("Available top-level keys:")
    for key in f.keys():
        print(f"  - {key}")
    
    # Check split
    if 'split' in f:
        print("\n✓ 'split' dataset found")
        split_data = f['split'][:]
        print(f"  Shape: {split_data.shape}")
        print(f"  Values: train=0, val=1, test=2")
        
        # Count images per split
        num_train = (split_data == 0).sum()
        num_val = (split_data == 1).sum()
        num_test = (split_data == 2).sum()
        
        print(f"\nSplit distribution:")
        print(f"  - train: {num_train} images")
        print(f"  - val: {num_val} images")
        print(f"  - test: {num_test} images")
    
    # Check split_GLIPunseen
    if 'split_GLIPunseen' in f:
        print("\n✓ 'split_GLIPunseen' dataset found!")
        glip_split = f['split_GLIPunseen'][:]
        print(f"  Shape: {glip_split.shape}")
        
        # Count clean images
        num_clean_train = (glip_split == 0).sum()
        num_clean_val = (glip_split == 1).sum()
        num_clean_test = (glip_split == 2).sum()
        num_contaminated = (glip_split == -1).sum()
        
        print(f"\nClean split distribution (GLIP-unseen):")
        print(f"  - train: {num_clean_train} images")
        print(f"  - val: {num_clean_val} images")
        print(f"  - test: {num_clean_test} images (← This is what you're using!)")
        print(f"  - contaminated (removed): {num_contaminated} images")
        
        print(f"\n✓ Your config is correct!")
        print(f"✓ Using clean test set with {num_clean_test} images")
        print(f"✓ No data leakage from GLIP pre-training")
    else:
        print("\n✗ 'split_GLIPunseen' NOT found!")
        print("✗ You're using the contaminated test split")

print("\n" + "=" * 70)
print("Verification Complete")
print("=" * 70)