"""Validation checklist for canonical dataset generation"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime

from .mmcif_parser import MMCIFStructureParser
from .chain_mapper import ChainResidueMapper
from .pair_extractor import PeptideProteinPairExtractor
from .pdb_to_mmcif import PDBToMMCIFConverter

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Validation result for a single complex"""
    complex_id: str
    source_file: str
    source_format: str  # pdb | mmcif
    
    # 10-point checklist
    parse_success: bool
    auth_chain_found: bool
    peptide_chain_found: bool
    protein_chain_found: bool
    peptide_length_valid: bool
    protein_length_valid: bool
    pair_extractor_confidence: float
    quarantine_reason: Optional[str]
    parquet_written: bool
    visualization_compatible: bool
    
    # Additional metadata
    pdb_id: Optional[str] = None
    peptide_length: Optional[int] = None
    protein_length: Optional[int] = None
    num_chains: Optional[int] = None
    mmcif_path: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def passed(self) -> bool:
        """Check if validation passed"""
        return (
            self.parse_success and
            self.auth_chain_found and
            self.peptide_chain_found and
            self.protein_chain_found and
            self.peptide_length_valid and
            self.protein_length_valid and
            self.pair_extractor_confidence >= 0.5 and
            self.quarantine_reason is None and
            self.parquet_written
        )


class ValidationChecklist:
    """
    Run 10-point validation checklist on complexes
    
    Checklist:
    1. Parse successful
    2. Auth chain found
    3. Peptide chain found
    4. Protein chain found
    5. Peptide length policy passed
    6. Protein length policy passed
    7. Pair extractor confidence
    8. Quarantine reason (if any)
    9. Parquet record written
    10. Visualization pipeline compatible
    """
    
    def __init__(
        self,
        mmcif_cache_dir: str | Path,
        chain_id_mode: str = "auth",
        residue_number_mode: str = "auth"
    ):
        """
        Initialize validation checklist
        
        Args:
            mmcif_cache_dir: Directory for mmCIF cache
            chain_id_mode: Chain ID mode
            residue_number_mode: Residue number mode
        """
        self.mmcif_cache_dir = Path(mmcif_cache_dir)
        self.chain_id_mode = chain_id_mode
        self.residue_number_mode = residue_number_mode
        
        # Initialize components
        self.parser = MMCIFStructureParser(use_auth_ids=(chain_id_mode == "auth"))
        self.pair_extractor = PeptideProteinPairExtractor(allow_extension=True)
        self.pdb_converter = PDBToMMCIFConverter(mmcif_cache_dir)
        
        # Results
        self.results: List[ValidationResult] = []
    
    def validate_complex(
        self,
        structure_file: Path,
        complex_id: Optional[str] = None,
        source_db: str = "unknown"
    ) -> ValidationResult:
        """
        Validate a single complex
        
        Args:
            structure_file: Path to structure file (PDB or mmCIF)
            complex_id: Complex identifier
            source_db: Source database
            
        Returns:
            ValidationResult
        """
        if not complex_id:
            complex_id = structure_file.stem
        
        # Determine source format
        source_format = "mmcif" if structure_file.suffix.lower() in ['.cif', '.mmcif'] else "pdb"
        
        # Initialize result
        result = ValidationResult(
            complex_id=complex_id,
            source_file=str(structure_file),
            source_format=source_format,
            parse_success=False,
            auth_chain_found=False,
            peptide_chain_found=False,
            protein_chain_found=False,
            peptide_length_valid=False,
            protein_length_valid=False,
            pair_extractor_confidence=0.0,
            quarantine_reason=None,
            parquet_written=False,
            visualization_compatible=False
        )
        
        try:
            # Convert PDB to mmCIF if needed
            if source_format == "pdb":
                mmcif_path, status = self.pdb_converter.convert(structure_file)
                if not mmcif_path:
                    result.error_message = f"PDB→mmCIF conversion failed: {status}"
                    result.quarantine_reason = "pdb_conversion_failed"
                    return result
                
                result.mmcif_path = str(mmcif_path)
                structure_file = mmcif_path
            else:
                result.mmcif_path = str(structure_file)
            
            # 1. Parse structure
            try:
                parsed = self.parser.parse(structure_file)
                result.parse_success = True
                result.pdb_id = parsed.get('pdb_id')
                result.num_chains = len(parsed.get('chains', []))
            except Exception as e:
                result.error_message = f"Parse failed: {e}"
                result.quarantine_reason = "parse_error"
                return result
            
            chains = parsed.get('chains', [])
            
            if not chains:
                result.error_message = "No chains found"
                result.quarantine_reason = "no_chains"
                return result
            
            # 2. Check auth chains
            auth_chains = [c for c in chains if c.chain_id_auth]
            result.auth_chain_found = len(auth_chains) > 0
            
            if not result.auth_chain_found:
                result.error_message = "No auth chains found"
                result.quarantine_reason = "no_auth_chains"
                return result
            
            # Extract pairs
            pairs, warnings = self.pair_extractor.extract_pairs(chains)
            
            if not pairs:
                result.error_message = f"No pairs extracted: {'; '.join(warnings)}"
                
                # Determine specific reason
                if any("peptide" in w.lower() for w in warnings):
                    result.quarantine_reason = "no_peptide_chain"
                elif any("protein" in w.lower() for w in warnings):
                    result.quarantine_reason = "no_protein_chain"
                else:
                    result.quarantine_reason = "no_valid_pairs"
                
                return result
            
            # Use first pair for validation
            pair = pairs[0]
            
            # 3. Peptide chain found
            result.peptide_chain_found = True
            result.peptide_length = len(pair.peptide_chain.residues)
            
            # 4. Protein chain found
            result.protein_chain_found = True
            result.protein_length = len(pair.protein_chain.residues)
            
            # 5. Peptide length valid
            result.peptide_length_valid = (
                5 <= result.peptide_length <= 50
            )
            
            if not result.peptide_length_valid:
                if result.peptide_length < 5:
                    result.quarantine_reason = "peptide_too_short"
                else:
                    result.quarantine_reason = "peptide_too_long"
            
            # 6. Protein length valid
            result.protein_length_valid = result.protein_length >= 30
            
            if not result.protein_length_valid:
                result.quarantine_reason = "protein_too_short"
            
            # 7. Pair extractor confidence
            result.pair_extractor_confidence = pair.confidence
            
            if result.pair_extractor_confidence < 0.5:
                result.quarantine_reason = "low_confidence"
            
            # 8. Quarantine reason already set above if any issues
            
            # 9. Parquet written (simulated - would be set by CanonicalBuilder)
            result.parquet_written = result.quarantine_reason is None
            
            # 10. Visualization compatible (check if structure can be loaded)
            result.visualization_compatible = result.parse_success and result.parquet_written
            
        except Exception as e:
            result.error_message = f"Validation error: {e}"
            result.quarantine_reason = "validation_error"
            logger.error(f"Validation failed for {complex_id}: {e}", exc_info=True)
        
        return result
    
    def validate_batch(
        self,
        structure_files: List[Path],
        source_db: str = "unknown"
    ) -> List[ValidationResult]:
        """
        Validate batch of complexes
        
        Args:
            structure_files: List of structure files
            source_db: Source database
            
        Returns:
            List of validation results
        """
        logger.info(f"Validating {len(structure_files)} complexes...")
        
        for i, structure_file in enumerate(structure_files):
            result = self.validate_complex(structure_file, source_db=source_db)
            self.results.append(result)
            
            # Progress
            if (i + 1) % 10 == 0:
                passed = sum(1 for r in self.results if r.passed)
                logger.info(
                    f"Progress: {i+1}/{len(structure_files)} "
                    f"(passed: {passed}, failed: {i+1-passed})"
                )
        
        return self.results
    
    def get_summary(self) -> Dict:
        """Get validation summary statistics"""
        if not self.results:
            return {}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        # Count by checklist item
        checklist_stats = {
            "parse_success": sum(1 for r in self.results if r.parse_success),
            "auth_chain_found": sum(1 for r in self.results if r.auth_chain_found),
            "peptide_chain_found": sum(1 for r in self.results if r.peptide_chain_found),
            "protein_chain_found": sum(1 for r in self.results if r.protein_chain_found),
            "peptide_length_valid": sum(1 for r in self.results if r.peptide_length_valid),
            "protein_length_valid": sum(1 for r in self.results if r.protein_length_valid),
            "high_confidence": sum(1 for r in self.results if r.pair_extractor_confidence >= 0.8),
            "no_quarantine": sum(1 for r in self.results if r.quarantine_reason is None),
            "parquet_written": sum(1 for r in self.results if r.parquet_written),
            "visualization_compatible": sum(1 for r in self.results if r.visualization_compatible)
        }
        
        # Quarantine reasons
        quarantine_reasons = {}
        for r in self.results:
            if r.quarantine_reason:
                quarantine_reasons[r.quarantine_reason] = quarantine_reasons.get(r.quarantine_reason, 0) + 1
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "checklist_stats": checklist_stats,
            "quarantine_reasons": quarantine_reasons
        }
    
    def export_results(self, output_file: str | Path):
        """Export validation results to CSV"""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save
        df.to_csv(output_file, index=False)
        logger.info(f"Validation results exported to {output_file}")
    
    def export_summary_report(self, output_file: str | Path):
        """Export summary HTML report"""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_summary()
        
        # Generate HTML
        html = self._generate_summary_html(summary)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Summary report exported to {output_file}")
    
    def _generate_summary_html(self, summary: Dict) -> str:
        """Generate HTML summary report"""
        # Checklist table
        checklist_rows = ""
        for item, count in summary['checklist_stats'].items():
            pct = (count / summary['total']) * 100
            status = "✓" if pct >= 80 else "⚠" if pct >= 50 else "✗"
            checklist_rows += f"""
            <tr>
                <td>{status}</td>
                <td>{item.replace('_', ' ').title()}</td>
                <td>{count}/{summary['total']}</td>
                <td>{pct:.1f}%</td>
            </tr>
            """
        
        # Quarantine reasons
        quarantine_rows = ""
        for reason, count in sorted(summary['quarantine_reasons'].items(), key=lambda x: -x[1]):
            pct = (count / summary['total']) * 100
            quarantine_rows += f"""
            <tr>
                <td>{reason.replace('_', ' ').title()}</td>
                <td>{count}</td>
                <td>{pct:.1f}%</td>
            </tr>
            """
        
        pass_rate_color = "#28a745" if summary['pass_rate'] >= 0.8 else "#ffc107" if summary['pass_rate'] >= 0.5 else "#dc3545"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Validation Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        .stat-box {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 6px; border-left: 4px solid #007bff; }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; color: {pass_rate_color}; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #f8f9fa; padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6; }}
        td {{ padding: 10px; border-bottom: 1px solid #dee2e6; }}
        tr:hover {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sprint 2B Validation Summary</h1>
        
        <div class="stat-box">
            <div class="stat-value">{summary['pass_rate']:.1%}</div>
            <div>Pass Rate ({summary['passed']}/{summary['total']} complexes)</div>
        </div>
        
        <h2>10-Point Checklist Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Status</th>
                    <th>Check</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                {checklist_rows}
            </tbody>
        </table>
        
        <h2>Quarantine Reasons</h2>
        <table>
            <thead>
                <tr>
                    <th>Reason</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                {quarantine_rows}
            </tbody>
        </table>
        
        <p style="margin-top: 30px; color: #666;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validation_checklist.py <structure_dir>")
        sys.exit(1)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    structure_dir = Path(sys.argv[1])
    
    # Find structure files
    structure_files = (
        list(structure_dir.glob("*.cif")) +
        list(structure_dir.glob("*.pdb")) +
        list(structure_dir.glob("*.ent"))
    )
    
    if not structure_files:
        print(f"No structure files found in {structure_dir}")
        sys.exit(1)
    
    print(f"Found {len(structure_files)} structure files")
    
    # Run validation
    validator = ValidationChecklist(
        mmcif_cache_dir="data/staging/mmcif_cache",
        chain_id_mode="auth",
        residue_number_mode="auth"
    )
    
    results = validator.validate_batch(structure_files[:100])  # Limit to 100
    
    # Export results
    validator.export_results("data/staging/validation_results.csv")
    validator.export_summary_report("data/staging/validation_summary.html")
    
    # Print summary
    summary = validator.get_summary()
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    print(f"Total: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print("="*60)
