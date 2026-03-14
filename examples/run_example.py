"""
Example usage of PeptidQuantum pipeline

This script demonstrates how to run the complete pipeline on a sample complex.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peptidquantum.pipeline import PeptidQuantumPipeline, PipelineConfig


def example_1_basic():
    """Example 1: Basic usage with PDB ID"""
    print("="*60)
    print("Example 1: Basic Pipeline Usage")
    print("="*60)
    
    # Initialize pipeline
    pipeline = PeptidQuantumPipeline(
        output_base_dir="outputs",
        cache_dir="data/cache"
    )
    
    # Run pipeline
    results = pipeline.run(
        complex_id="1A1M",  # Example: Insulin-like growth factor binding protein
        protein_chain="A",
        peptide_chain="B",
        pocket_radius=8.0,
        use_arpeggio=True,
        use_plip=True,
        generate_pymol=True,
        generate_report=True,
        generate_viewer=True
    )
    
    print("\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Complex ID: {results['complex_id']}")
    print(f"  Output: {results['output_dir']}")
    print(f"  Interactions: {results['num_interactions']}")
    
    return results


def example_2_local_file():
    """Example 2: Using local CIF file"""
    print("\n" + "="*60)
    print("Example 2: Local CIF File")
    print("="*60)
    
    # Path to local CIF file
    cif_file = Path("data/structures/example.cif")
    
    if not cif_file.exists():
        print(f"CIF file not found: {cif_file}")
        print("Skipping this example.")
        return None
    
    pipeline = PeptidQuantumPipeline()
    
    results = pipeline.run(
        cif_path=cif_file,
        protein_chain="A",
        peptide_chain="B"
    )
    
    return results


def example_3_config_file():
    """Example 3: Using configuration file"""
    print("\n" + "="*60)
    print("Example 3: Configuration File")
    print("="*60)
    
    # Create config
    config = PipelineConfig(
        complex_id="1ABC",
        protein_chain="A",
        peptide_chain="B",
        pocket_radius=8.0,
        use_arpeggio=True,
        use_plip=True,
        generate_pymol=False,  # Skip PyMOL for faster testing
        generate_report=True,
        output_dir="outputs"
    )
    
    # Save config
    config_file = Path("examples/example_config.json")
    config_file.parent.mkdir(exist_ok=True)
    config.to_json(config_file)
    print(f"Config saved to: {config_file}")
    
    # Load and use config
    loaded_config = PipelineConfig.from_json(config_file)
    
    pipeline = PeptidQuantumPipeline(
        output_base_dir=loaded_config.output_dir,
        cache_dir=loaded_config.cache_dir
    )
    
    results = pipeline.run(
        complex_id=loaded_config.complex_id,
        protein_chain=loaded_config.protein_chain,
        peptide_chain=loaded_config.peptide_chain,
        pocket_radius=loaded_config.pocket_radius,
        use_arpeggio=loaded_config.use_arpeggio,
        use_plip=loaded_config.use_plip,
        generate_pymol=loaded_config.generate_pymol,
        generate_report=loaded_config.generate_report
    )
    
    return results


def example_4_minimal():
    """Example 4: Minimal pipeline (no external tools)"""
    print("\n" + "="*60)
    print("Example 4: Minimal Pipeline (No External Tools)")
    print("="*60)
    
    pipeline = PeptidQuantumPipeline()
    
    # Run with minimal dependencies
    results = pipeline.run(
        complex_id="1A1M",
        protein_chain="A",
        peptide_chain="B",
        use_arpeggio=False,  # Disable if not installed
        use_plip=False,      # Disable if not installed
        generate_pymol=False, # Disable if not installed
        generate_report=True, # HTML report always works
        generate_viewer=True  # 3Dmol.js viewer always works
    )
    
    print("\nNote: This example works with minimal dependencies")
    print("Install Arpeggio, PLIP, and PyMOL for full functionality")
    
    return results


def example_5_auto_detect():
    """Example 5: Auto-detect chains"""
    print("\n" + "="*60)
    print("Example 5: Auto-Detect Chains")
    print("="*60)
    
    pipeline = PeptidQuantumPipeline()
    
    # Don't specify chains - let pipeline auto-detect
    results = pipeline.run(
        complex_id="1A1M",
        # protein_chain and peptide_chain not specified
        # Pipeline will classify based on chain length
    )
    
    return results


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" "*20 + "PeptidQuantum Examples")
    print("="*70)
    
    examples = [
        ("Basic Usage", example_1_basic),
        ("Local File", example_2_local_file),
        ("Config File", example_3_config_file),
        ("Minimal Pipeline", example_4_minimal),
        ("Auto-Detect Chains", example_5_auto_detect)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n\nRunning: {name}")
            result = example_func()
            results[name] = result
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            results[name] = None
            continue
    
    # Summary
    print("\n\n" + "="*70)
    print(" "*25 + "Summary")
    print("="*70)
    
    for name, result in results.items():
        if result and result.get('status') == 'success':
            print(f"✓ {name}: Success")
            print(f"  Output: {result['output_dir']}")
        elif result:
            print(f"✗ {name}: Failed at {result.get('stage', 'unknown')}")
        else:
            print(f"- {name}: Skipped")
    
    print("\n" + "="*70)
    print("\nTo view results, open the report.html files in your browser:")
    for name, result in results.items():
        if result and result.get('status') == 'success':
            report_path = Path(result['output_dir']) / "report.html"
            if report_path.exists():
                print(f"  • {report_path}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Run specific example or all
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            "1": example_1_basic,
            "2": example_2_local_file,
            "3": example_3_config_file,
            "4": example_4_minimal,
            "5": example_5_auto_detect
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print("Available: 1, 2, 3, 4, 5")
    else:
        main()
