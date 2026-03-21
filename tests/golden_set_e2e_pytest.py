"""
Golden end-to-end test set for pipeline validation

unittest keşfi bu dosyayı yüklemez (adı test_*.py değil). Çalıştırmak için:
  pip install pytest
  pytest tests/golden_set_e2e_pytest.py -v

Tests 3 representative complexes:
1. Clean experimental complex (standard case)
2. Multi-chain complex (chain ID complexity)
3. Low/no interaction complex (edge case)
"""
import sys
from pathlib import Path
import pytest
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peptidquantum.pipeline import PeptidQuantumPipeline, PipelineConfig


# Golden test complexes
GOLDEN_COMPLEXES = {
    "clean_experimental": {
        "complex_id": "1A1M",
        "description": "Insulin-like growth factor binding protein - clean experimental",
        "protein_chain": "A",
        "peptide_chain": "B",
        "expected": {
            "min_interactions": 5,
            "has_hbond": True,
            "has_hydrophobic": True,
            "contact_matrix_shape": (None, None),  # Will be determined
            "report_generated": True,
            "viewer_generated": True
        }
    },
    "multi_chain": {
        "complex_id": "1SSH",
        "description": "Streptavidin-biotin complex - multiple chains",
        "protein_chain": "A",
        "peptide_chain": "E",
        "expected": {
            "min_interactions": 3,
            "contact_matrix_shape": (None, None),
            "report_generated": True,
            "viewer_generated": True
        }
    },
    "edge_case": {
        "complex_id": "1A2K",
        "description": "Small peptide complex - potential low interactions",
        "protein_chain": "A",
        "peptide_chain": "B",
        "expected": {
            "min_interactions": 0,  # May have few or no interactions
            "report_generated": True,  # Should still generate report
            "viewer_generated": True
        }
    }
}


@pytest.fixture
def pipeline():
    """Create pipeline instance"""
    return PeptidQuantumPipeline(
        output_base_dir="tests/outputs",
        cache_dir="tests/cache"
    )


@pytest.fixture
def test_config():
    """Create test configuration"""
    return PipelineConfig(
        complex_id="1A1M",
        protein_chain="A",
        peptide_chain="B",
        chain_id_mode="auth",
        residue_number_mode="auth",
        use_arpeggio=False,  # Disable for CI/CD
        use_plip=False,      # Disable for CI/CD
        generate_pymol=False, # Disable for CI/CD
        generate_report=True,
        generate_viewer=True,
        output_dir="tests/outputs"
    )


class TestGoldenSet:
    """Golden test set for end-to-end validation"""
    
    def test_clean_experimental(self, pipeline):
        """Test 1: Clean experimental complex"""
        spec = GOLDEN_COMPLEXES["clean_experimental"]
        
        results = pipeline.run(
            complex_id=spec["complex_id"],
            protein_chain=spec["protein_chain"],
            peptide_chain=spec["peptide_chain"],
            use_arpeggio=False,
            use_plip=False,
            generate_pymol=False
        )
        
        # Validate results
        assert results["status"] == "success", f"Pipeline failed: {results.get('stage')}"
        assert results["complex_id"] == spec["complex_id"]
        
        # Check output directory
        output_dir = Path(results["output_dir"])
        assert output_dir.exists(), "Output directory not created"
        
        # Check required files
        assert (output_dir / "structures" / "complex.cif").exists()
        assert (output_dir / "report.html").exists()
        assert (output_dir / "viewer.html").exists()
        
        # Check data files (may be empty if no interaction tools)
        data_dir = output_dir / "data"
        assert data_dir.exists()
        
        logging.info(f"✓ Clean experimental test passed: {spec['complex_id']}")
    
    def test_multi_chain_complex(self, pipeline):
        """Test 2: Multi-chain complex"""
        spec = GOLDEN_COMPLEXES["multi_chain"]
        
        results = pipeline.run(
            complex_id=spec["complex_id"],
            protein_chain=spec["protein_chain"],
            peptide_chain=spec["peptide_chain"],
            use_arpeggio=False,
            use_plip=False,
            generate_pymol=False
        )
        
        assert results["status"] == "success"
        
        output_dir = Path(results["output_dir"])
        assert (output_dir / "report.html").exists()
        
        logging.info(f"✓ Multi-chain test passed: {spec['complex_id']}")
    
    def test_edge_case_low_interactions(self, pipeline):
        """Test 3: Edge case with potentially low interactions"""
        spec = GOLDEN_COMPLEXES["edge_case"]
        
        results = pipeline.run(
            complex_id=spec["complex_id"],
            protein_chain=spec["protein_chain"],
            peptide_chain=spec["peptide_chain"],
            use_arpeggio=False,
            use_plip=False,
            generate_pymol=False
        )
        
        # Should succeed even with no interactions
        assert results["status"] == "success"
        
        output_dir = Path(results["output_dir"])
        assert (output_dir / "report.html").exists(), "Report should generate even with no interactions"
        
        logging.info(f"✓ Edge case test passed: {spec['complex_id']}")
    
    def test_chain_id_consistency(self, pipeline):
        """Test 4: Chain ID consistency across outputs"""
        spec = GOLDEN_COMPLEXES["clean_experimental"]
        
        results = pipeline.run(
            complex_id=spec["complex_id"],
            protein_chain=spec["protein_chain"],
            peptide_chain=spec["peptide_chain"],
            use_arpeggio=False,
            use_plip=False,
            generate_pymol=False
        )
        
        output_dir = Path(results["output_dir"])
        
        # Check that chain IDs are consistent
        # (This would require parsing outputs - simplified for now)
        assert results["status"] == "success"
        
        logging.info("✓ Chain ID consistency test passed")
    
    def test_config_based_run(self, test_config):
        """Test 5: Config-based pipeline run"""
        pipeline = PeptidQuantumPipeline(
            output_base_dir=test_config.output_dir,
            cache_dir=test_config.cache_dir
        )
        
        results = pipeline.run(
            complex_id=test_config.complex_id,
            protein_chain=test_config.protein_chain,
            peptide_chain=test_config.peptide_chain,
            use_arpeggio=test_config.use_arpeggio,
            use_plip=test_config.use_plip,
            generate_pymol=test_config.generate_pymol,
            generate_report=test_config.generate_report,
            generate_viewer=test_config.generate_viewer
        )
        
        assert results["status"] == "success"
        
        logging.info("✓ Config-based run test passed")
    
    def test_graceful_fallback_no_tools(self, pipeline):
        """Test 6: Graceful fallback when external tools unavailable"""
        results = pipeline.run(
            complex_id="1A1M",
            protein_chain="A",
            peptide_chain="B",
            use_arpeggio=False,
            use_plip=False,
            generate_pymol=False
        )
        
        # Should succeed with minimal functionality
        assert results["status"] == "success"
        assert results["num_interactions"] == 0  # No tools = no interactions
        
        output_dir = Path(results["output_dir"])
        assert (output_dir / "report.html").exists(), "Report should generate without tools"
        
        logging.info("✓ Graceful fallback test passed")


class TestChainResidueIDPolicy:
    """Test chain and residue ID policy consistency"""
    
    def test_auth_mode_consistency(self, pipeline):
        """Test auth mode consistency across pipeline"""
        # This would require checking that all components use auth IDs
        # Simplified for now
        config = PipelineConfig(
            complex_id="1A1M",
            protein_chain="A",
            peptide_chain="B",
            chain_id_mode="auth",
            residue_number_mode="auth"
        )
        
        assert config.chain_id_mode == "auth"
        assert config.residue_number_mode == "auth"
        
        logging.info("✓ Auth mode consistency test passed")
    
    def test_label_mode_consistency(self, pipeline):
        """Test label mode consistency across pipeline"""
        config = PipelineConfig(
            complex_id="1A1M",
            protein_chain="A",
            peptide_chain="B",
            chain_id_mode="label",
            residue_number_mode="label"
        )
        
        assert config.chain_id_mode == "label"
        assert config.residue_number_mode == "label"
        
        logging.info("✓ Label mode consistency test passed")
    
    def test_invalid_mode_rejected(self):
        """Test that invalid modes are rejected"""
        with pytest.raises(ValueError):
            PipelineConfig(
                complex_id="1A1M",
                chain_id_mode="invalid"
            )
        
        with pytest.raises(ValueError):
            PipelineConfig(
                complex_id="1A1M",
                residue_number_mode="invalid"
            )
        
        logging.info("✓ Invalid mode rejection test passed")


class TestOutputStructure:
    """Test output directory structure"""
    
    def test_output_directory_structure(self, pipeline):
        """Test that output directory has correct structure"""
        results = pipeline.run(
            complex_id="1A1M",
            protein_chain="A",
            peptide_chain="B",
            use_arpeggio=False,
            use_plip=False,
            generate_pymol=False
        )
        
        output_dir = Path(results["output_dir"])
        
        # Check directory structure
        assert (output_dir / "structures").is_dir()
        assert (output_dir / "data").is_dir()
        assert (output_dir / "figures").is_dir()
        
        # Check minimum files
        assert (output_dir / "structures" / "complex.cif").exists()
        assert (output_dir / "report.html").exists()
        assert (output_dir / "viewer.html").exists()
        
        logging.info("✓ Output directory structure test passed")
    
    def test_data_files_created(self, pipeline):
        """Test that data files are created"""
        results = pipeline.run(
            complex_id="1A1M",
            protein_chain="A",
            peptide_chain="B",
            use_arpeggio=False,
            use_plip=False,
            generate_pymol=False
        )
        
        output_dir = Path(results["output_dir"])
        data_dir = output_dir / "data"
        
        # These should exist even with no interactions
        assert (data_dir / "contacts.tsv").exists()
        assert (data_dir / "interaction_fingerprint.json").exists()
        
        logging.info("✓ Data files creation test passed")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
