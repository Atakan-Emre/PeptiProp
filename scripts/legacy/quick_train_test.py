#!/usr/bin/env python3
"""
Quick test script for the improved PeptGAINET model
Run this to verify the model works before full training
"""
from __future__ import annotations

import sys
import os
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from pathlib import Path

# Test data creation
def create_dummy_batch(batch_size=4, node_dim=32, max_nodes=50):
    """Create a dummy batch for testing"""
    batch = {
        "peptide_x": torch.randn(batch_size, max_nodes, node_dim),
        "peptide_adj": torch.randn(batch_size, max_nodes, max_nodes),
        "peptide_mask": torch.ones(batch_size, max_nodes),
        "protein_x": torch.randn(batch_size, max_nodes, node_dim),
        "protein_adj": torch.randn(batch_size, max_nodes, max_nodes),
        "protein_mask": torch.ones(batch_size, max_nodes),
        "labels": torch.randint(0, 2, (batch_size,)),
        "pair_id": [f"pair_{i}" for i in range(batch_size)]
    }
    return batch

def test_original_model():
    """Test the original PeptGAINET model"""
    print("Testing original PeptGAINET...")
    from peptidquantum.peptgainet import PeptGAINET
    
    model = PeptGAINET(node_dim=32)
    batch = create_dummy_batch()
    
    with torch.no_grad():
        out = model(batch)
    
    print(f"  Output shape: {out['prob'].shape}")
    print(f"  Output range: [{out['prob'].min():.3f}, {out['prob'].max():.3f}]")
    print("  [OK] Original model works!")
    return True

def test_improved_model():
    """Test the improved PeptGAINET model"""
    print("\nTesting Improved PeptGAINET...")
    from peptidquantum.peptgainet import ImprovedPeptGAINET
    
    model = ImprovedPeptGAINET(node_dim=32)
    batch = create_dummy_batch()
    
    with torch.no_grad():
        out = model(batch)
    
    print(f"  Output shape: {out['prob'].shape}")
    print(f"  Output range: [{out['prob'].min():.3f}, {out['prob'].max():.3f}]")
    print("  [OK] Improved model works!")
    return True

def test_v2_model():
    """Test the PeptGAINETV2 model"""
    print("\nTesting PeptGAINETV2...")
    from peptidquantum.peptgainet import PeptGAINETV2
    
    model = PeptGAINETV2(node_dim=32)
    batch = create_dummy_batch()
    
    with torch.no_grad():
        out = model(batch)
    
    print(f"  Output shape: {out['prob'].shape}")
    print(f"  Output range: [{out['prob'].min():.3f}, {out['prob'].max():.3f}]")
    print("  [OK] V2 model works!")
    return True

def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    from peptidquantum.peptgainet import ImprovedPeptGAINET
    
    model = ImprovedPeptGAINET(node_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.BCELoss()
    
    batch = create_dummy_batch()
    
    # Forward pass
    out = model(batch)
    loss = criterion(out['prob'], batch['labels'].float())
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Loss: {loss.item():.4f}")
    print("  [OK] Training step works!")
    return True

def main():
    print("=" * 50)
    print("PeptGAINET Model Test Suite")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    try:
        # Test all models
        test_original_model()
        test_improved_model()
        test_v2_model()
        test_training_step()
        
        print("\n" + "="*50)
        print("[OK] All tests passed! Models are ready for training.")
        print("="*50)
        
        # Check if data exists
        train_path = Path("data/processed/geppri_train1_pairs.jsonl")
        test_path = Path("data/processed/geppri_test1_pairs.jsonl")
        
        if train_path.exists() and test_path.exists():
            print("\n[OK] Training data found!")
            print("  You can now run the full training with:")
            print("  python scripts/train_peptgainet_improved.py \\")
            print("    --train-jsonl data/processed/geppri_train1_pairs.jsonl \\")
            print("    --valid-jsonl data/processed/geppri_test1_pairs.jsonl \\")
            print("    --epochs 50 --batch-size 32 --lr 5e-4")
        else:
            print("\n[WARNING] Training data not found.")
            print("  Please run the data preparation first:")
            print("  python scripts/build_geppri_pair_dataset.py")
            
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
