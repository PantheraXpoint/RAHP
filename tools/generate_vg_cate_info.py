#!/usr/bin/env python3
"""
Generate vg_cate_info.json from VG-SGG-dicts-with-attri.json
This script extracts entity and predicate categories from the VG dataset dictionary file.
"""

import json
import os
import argparse
from pathlib import Path

def generate_vg_cate_info(vg_dicts_path, output_path):
    """
    Generate vg_cate_info.json from VG-SGG-dicts-with-attri.json
    
    Args:
        vg_dicts_path (str): Path to VG-SGG-dicts-with-attri.json
        output_path (str): Path to save vg_cate_info.json
    """
    
    print(f"Reading VG dictionary file: {vg_dicts_path}")
    
    # Read the VG-SGG-dicts-with-attri.json file
    with open(vg_dicts_path, 'r') as f:
        vg_data = json.load(f)
    
    # Extract entity categories from idx_to_label
    print("Extracting entity categories...")
    ent_cate = list(vg_data['idx_to_label'].values())
    print(f"Found {len(ent_cate)} entity categories")
    
    # Extract predicate categories from idx_to_predicate  
    print("Extracting predicate categories...")
    pred_cate = list(vg_data['idx_to_predicate'].values())
    print(f"Found {len(pred_cate)} predicate categories")
    
    # Create the required structure
    vg_cate_info = {
        'ent_cate': ent_cate,
        'pred_cate': pred_cate
    }
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the file
    print(f"Saving vg_cate_info.json to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(vg_cate_info, f, indent=4)
    
    print("✅ Successfully generated vg_cate_info.json")
    print(f"   - Entity categories: {len(ent_cate)}")
    print(f"   - Predicate categories: {len(pred_cate)}")
    
    # Print first few examples
    print("\nFirst 10 entity categories:")
    for i, cat in enumerate(ent_cate[:10]):
        print(f"   {i+1}. {cat}")
    
    print("\nFirst 10 predicate categories:")
    for i, cat in enumerate(pred_cate[:10]):
        print(f"   {i+1}. {cat}")

def main():
    parser = argparse.ArgumentParser(description="Generate vg_cate_info.json from VG-SGG-dicts-with-attri.json")
    parser.add_argument(
        "--vg_dicts_path", 
        default="/sensys/new/RAHP/DATASET/VG150/VG-SGG-dicts-with-attri.json",
        help="Path to VG-SGG-dicts-with-attri.json file"
    )
    parser.add_argument(
        "--output_path",
        default="/sensys/new/RAHP/DATA/vg/vg_cate_info.json", 
        help="Path to save vg_cate_info.json file"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    vg_dicts_path = os.path.abspath(args.vg_dicts_path)
    output_path = os.path.abspath(args.output_path)
    
    # Check if input file exists
    if not os.path.exists(vg_dicts_path):
        print(f"❌ Error: Input file not found: {vg_dicts_path}")
        return
    
    # Generate the file
    generate_vg_cate_info(vg_dicts_path, output_path)

if __name__ == "__main__":
    main()