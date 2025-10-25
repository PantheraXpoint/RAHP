#!/usr/bin/env python3
"""
Generate base_triplet_labels_all.pt and base_pair_labels_all.pt for RAHP.

Based on RAHP paper (AAAI 2025) Appendix D.
"""

import torch
import json
import os

print("=" * 70)
print("Generating base_triplet_labels_all.pt and base_pair_labels_all.pt")
print("=" * 70)

# CORRECTED PATH - Use actual location
vg_cate_info_path = '/RAHP/DATA/vg/vg_cate_info.json'
print(f"\n[1/3] Loading VG150 categories from: {vg_cate_info_path}")

if not os.path.exists(vg_cate_info_path):
    raise FileNotFoundError(f"Cannot find: {vg_cate_info_path}")

with open(vg_cate_info_path, 'r') as f:
    cate_info = json.load(f)

predicates = cate_info["pred_cate"]
entities = cate_info["ent_cate"]

print(f"✓ Loaded {len(predicates)} predicates")
print(f"✓ Loaded {len(entities)} entities")

# ==================== 1. Base Predicates ====================
print(f"\n[2/3] Generating base_triplet_labels_all.pt")

# Base predicates from RAHP paper Appendix D (36 predicates)
base_predicates = [
    'above', 'against', 'at', 'attached to', 'behind', 'belonging to', 
    'between', 'carrying', 'covered in', 'covering', 'for', 'from', 
    'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 
    'looking at', 'made of', 'near', 'of', 'on', 'over', 'parked on', 
    'playing', 'riding', 'sitting on', 'standing on', 'to', 'under', 
    'walking on', 'watching', 'wearing', 'wears', 'with'
]

# Get indices (1-indexed for background)
base_pred_indices = []
for pred in base_predicates:
    if pred in predicates:
        idx = predicates.index(pred) + 1  # +1 for background
        base_pred_indices.append(idx)
    else:
        print(f"⚠ Warning: '{pred}' not found in VG150 categories!")

base_triplet_labels_all = torch.tensor(base_pred_indices, dtype=torch.long)

print(f"✓ Found {len(base_pred_indices)}/36 base predicates")
print(f"✓ Indices: {base_pred_indices[:5]}... (first 5)")
print(f"✓ Shape: {base_triplet_labels_all.shape}")

# Novel predicates (14 zero-shot predicates)
novel_predicates = [
    'across', 'along', 'and', 'eating', 'flying in', 'laying on', 
    'lying on', 'mounted on', 'on back of', 'painted on', 'part of', 
    'says', 'using', 'walking in'
]

print(f"✓ Novel predicates (zero-shot): {len(novel_predicates)}")

# ==================== 2. Base Entity Pairs ====================
print(f"\n[3/3] Generating base_pair_labels_all.pt")

# Create all possible entity pairs (placeholder approach)
# Note: Ideally, extract actual training pairs from annotations
all_pairs = []
for i in range(len(entities)):
    for j in range(len(entities)):
        if i != j:  # Exclude self-pairs
            all_pairs.append([i + 1, j + 1])  # +1 for background

base_pair_labels_all = torch.tensor(all_pairs, dtype=torch.long)

print(f"✓ Generated {len(all_pairs)} entity pairs")
print(f"✓ Shape: {base_pair_labels_all.shape}")
print(f"⚠ Note: Using ALL pairs as base (conservative approach)")
print(f"⚠ For accurate metrics, extract actual training pairs")

# ==================== 3. Save Files ====================
print(f"\nSaving files...")

# Save to DATA directory (where vg_cate_info.json is)
output_dir = '/RAHP/DATA/vg/'
os.makedirs(output_dir, exist_ok=True)

triplet_path = os.path.join(output_dir, 'base_triplet_labels_all.pt')
pair_path = os.path.join(output_dir, 'base_pair_labels_all.pt')

torch.save(base_triplet_labels_all, triplet_path)
torch.save(base_pair_labels_all, pair_path)

print(f"✓ Saved: {triplet_path}")
print(f"✓ Saved: {pair_path}")

# Verify
if os.path.exists(triplet_path) and os.path.exists(pair_path):
    triplet_size = os.path.getsize(triplet_path)
    pair_size = os.path.getsize(pair_path)
    print(f"\n✓ Files created successfully!")
    print(f"  - base_triplet_labels_all.pt: {triplet_size} bytes")
    print(f"  - base_pair_labels_all.pt: {pair_size} bytes")
    
    # Quick verification
    loaded_triplet = torch.load(triplet_path)
    loaded_pair = torch.load(pair_path)
    print(f"\n✓ Verification:")
    print(f"  - Triplet tensor: {loaded_triplet.shape}")
    print(f"  - Pair tensor: {loaded_pair.shape}")
else:
    print("\n✗ Error: Files were not created!")

print("\n" + "=" * 70)
print("Generation Complete!")
print("=" * 70)
print("\nNext steps:")
print("1. Update sgg_eval.py paths (run update script)")
print("2. Re-run evaluation")
print("=" * 70)