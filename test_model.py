#!/usr/bin/env python3
"""Test the fixed improved model"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from peptidquantum.peptgainet.model_fixed import ImprovedPeptGAINET

def test_model():
    """Test model with dummy data"""
    print("Testing ImprovedPeptGAINET...")
    
    # Create dummy batch
    batch_size = 4
    max_nodes = 50
    node_dim = 65  # Typical dimension from dataset
    
    batch = {
        "peptide_x": torch.randn(batch_size, max_nodes, node_dim),
        "peptide_adj": torch.randn(batch_size, max_nodes, max_nodes).abs(),
        "peptide_mask": torch.ones(batch_size, max_nodes),
        "protein_x": torch.randn(batch_size, max_nodes, node_dim),
        "protein_adj": torch.randn(batch_size, max_nodes, max_nodes).abs(),
        "protein_mask": torch.ones(batch_size, max_nodes),
        "labels": torch.randint(0, 2, (batch_size,))
    }
    
    # Create model
    model = ImprovedPeptGAINET(node_dim=node_dim)
    
    # Test forward pass
    try:
        with torch.no_grad():
            output = model(batch)
        
        print(f"  Output shape: {output['prob'].shape}")
        print(f"  Output range: [{output['prob'].min():.3f}, {output['prob'].max():.3f}]")
        print("  [OK] Model works correctly!")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    batch_size = 4
    max_nodes = 50
    node_dim = 65
    
    batch = {
        "peptide_x": torch.randn(batch_size, max_nodes, node_dim),
        "peptide_adj": torch.randn(batch_size, max_nodes, max_nodes).abs(),
        "peptide_mask": torch.ones(batch_size, max_nodes),
        "protein_x": torch.randn(batch_size, max_nodes, node_dim),
        "protein_adj": torch.randn(batch_size, max_nodes, max_nodes).abs(),
        "protein_mask": torch.ones(batch_size, max_nodes),
        "labels": torch.randint(0, 2, (batch_size,)).float()
    }
    
    model = ImprovedPeptGAINET(node_dim=node_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.BCELoss()
    
    try:
        # Forward
        output = model(batch)
        loss = criterion(output['prob'], batch['labels'])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Loss: {loss.item():.4f}")
        print("  [OK] Training step works!")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("PeptidQuantum Model Test")
    print("="*60)
    
    success = True
    success = test_model() and success
    success = test_training_step() and success
    
    print("\n" + "="*60)
    if success:
        print("[OK] All tests passed!")
        print("="*60)
        print("\nYou can now run training with:")
        print("  python run.bat")
        print("or")
        print("  python train.py --data data/processed/geppri_all_pairs.jsonl")
    else:
        print("[ERROR] Some tests failed!")
        print("="*60)
        sys.exit(1)
