#!/usr/bin/env python3
"""Combine train and test data into single file for proper splitting"""
import sys
import json
from pathlib import Path

def main():
    # Paths
    train_path = Path("data/processed/geppri_train1_pairs.jsonl")
    test_path = Path("data/processed/geppri_test1_pairs.jsonl")
    output_path = Path("data/processed/geppri_all_pairs.jsonl")
    
    if not train_path.exists():
        print(f"Error: {train_path} not found")
        print("Please run build_geppri_pair_dataset.py first")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"Error: {test_path} not found")
        print("Please run build_geppri_pair_dataset.py first")
        sys.exit(1)
    
    # Read all records
    all_records = []
    
    with open(train_path) as f:
        for line in f:
            all_records.append(json.loads(line))
    
    with open(test_path) as f:
        for line in f:
            all_records.append(json.loads(line))
    
    # Write combined
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for record in all_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Combined {len(all_records)} records")
    print(f"Saved to: {output_path}")
    
    # Count labels
    pos = sum(1 for r in all_records if r['label'] == 1)
    neg = len(all_records) - pos
    print(f"Positive: {pos}")
    print(f"Negative: {neg}")

if __name__ == "__main__":
    main()
