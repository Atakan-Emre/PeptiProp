"""Quarantine manager for problematic structures - flag, don't delete"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class QuarantineReason(Enum):
    """Reasons for quarantine"""
    CHAIN_MAPPING_FAILED = "chain_mapping_failed"
    MISSING_PEPTIDE_CHAIN = "missing_peptide_chain"
    MISSING_PROTEIN_CHAIN = "missing_protein_chain"
    PEPTIDE_LIGAND_CONFUSION = "peptide_ligand_confusion"
    NO_INTERFACE_DETECTED = "no_interface_detected"
    STRUCTURE_PARSE_ERROR = "structure_parse_error"
    AUTH_LABEL_CONFLICT = "auth_label_conflict"
    PEPTIDE_TOO_SHORT = "peptide_too_short"
    PEPTIDE_TOO_LONG = "peptide_too_long"
    PROTEIN_TOO_SHORT = "protein_too_short"
    AMBIGUOUS_CHAIN_TYPE = "ambiguous_chain_type"
    DUPLICATE_CHAIN_IDS = "duplicate_chain_ids"
    MISSING_COORDINATES = "missing_coordinates"
    INVALID_RESIDUES = "invalid_residues"
    OTHER = "other"


@dataclass
class QuarantineRecord:
    """Record of quarantined structure"""
    complex_id: str
    pdb_id: str
    source_db: str
    reason: QuarantineReason
    details: str
    timestamp: str
    structure_file: Optional[str] = None
    metadata: Optional[Dict] = None


class QuarantineManager:
    """
    Manage quarantined structures
    
    Philosophy: Flag, don't delete
    - Problematic structures go to quarantine/
    - Reasons are logged
    - Can be reviewed later
    - Quality reports show quarantine breakdown
    """
    
    def __init__(self, quarantine_dir: str | Path):
        """
        Initialize quarantine manager
        
        Args:
            quarantine_dir: Directory for quarantined structures
        """
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        # Quarantine log
        self.log_file = self.quarantine_dir / "quarantine_log.json"
        self.records: List[QuarantineRecord] = []
        
        # Load existing records
        self._load_records()
    
    def quarantine(
        self,
        complex_id: str,
        pdb_id: str,
        source_db: str,
        reason: QuarantineReason,
        details: str,
        structure_file: Optional[str | Path] = None,
        metadata: Optional[Dict] = None
    ) -> QuarantineRecord:
        """
        Quarantine a structure
        
        Args:
            complex_id: Complex identifier
            pdb_id: PDB ID
            source_db: Source database
            reason: Quarantine reason
            details: Detailed explanation
            structure_file: Path to structure file (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            QuarantineRecord
        """
        # Create record
        record = QuarantineRecord(
            complex_id=complex_id,
            pdb_id=pdb_id,
            source_db=source_db,
            reason=reason,
            details=details,
            timestamp=datetime.now().isoformat(),
            structure_file=str(structure_file) if structure_file else None,
            metadata=metadata or {}
        )
        
        self.records.append(record)
        
        # Log
        logger.warning(
            f"Quarantined {complex_id} ({pdb_id}): "
            f"{reason.value} - {details}"
        )
        
        # Save
        self._save_records()
        
        return record
    
    def is_quarantined(self, complex_id: str) -> bool:
        """Check if complex is quarantined"""
        return any(r.complex_id == complex_id for r in self.records)
    
    def get_quarantine_reason(self, complex_id: str) -> Optional[QuarantineReason]:
        """Get quarantine reason for complex"""
        for record in self.records:
            if record.complex_id == complex_id:
                return record.reason
        return None
    
    def get_statistics(self) -> Dict:
        """Get quarantine statistics"""
        if not self.records:
            return {
                "total_quarantined": 0,
                "by_reason": {},
                "by_source": {}
            }
        
        # Count by reason
        by_reason = {}
        for record in self.records:
            reason = record.reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        # Count by source
        by_source = {}
        for record in self.records:
            source = record.source_db
            by_source[source] = by_source.get(source, 0) + 1
        
        return {
            "total_quarantined": len(self.records),
            "by_reason": by_reason,
            "by_source": by_source
        }
    
    def get_records_by_reason(self, reason: QuarantineReason) -> List[QuarantineRecord]:
        """Get all records with specific reason"""
        return [r for r in self.records if r.reason == reason]
    
    def export_report(self, output_file: str | Path):
        """Export quarantine report"""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.get_statistics()
        
        # Generate HTML report
        html = self._generate_html_report(stats)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Quarantine report exported to {output_file}")
    
    def _load_records(self):
        """Load quarantine records from log"""
        if not self.log_file.exists():
            return
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                record = QuarantineRecord(
                    complex_id=item['complex_id'],
                    pdb_id=item['pdb_id'],
                    source_db=item['source_db'],
                    reason=QuarantineReason(item['reason']),
                    details=item['details'],
                    timestamp=item['timestamp'],
                    structure_file=item.get('structure_file'),
                    metadata=item.get('metadata')
                )
                self.records.append(record)
            
            logger.info(f"Loaded {len(self.records)} quarantine records")
            
        except Exception as e:
            logger.error(f"Failed to load quarantine records: {e}")
    
    def _save_records(self):
        """Save quarantine records to log"""
        data = []
        
        for record in self.records:
            data.append({
                'complex_id': record.complex_id,
                'pdb_id': record.pdb_id,
                'source_db': record.source_db,
                'reason': record.reason.value,
                'details': record.details,
                'timestamp': record.timestamp,
                'structure_file': record.structure_file,
                'metadata': record.metadata
            })
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _generate_html_report(self, stats: Dict) -> str:
        """Generate HTML quarantine report"""
        # Reason breakdown
        reason_rows = ""
        for reason, count in sorted(stats['by_reason'].items(), key=lambda x: -x[1]):
            pct = (count / stats['total_quarantined']) * 100
            reason_rows += f"""
            <tr>
                <td>{reason.replace('_', ' ').title()}</td>
                <td>{count}</td>
                <td>{pct:.1f}%</td>
            </tr>
            """
        
        # Source breakdown
        source_rows = ""
        for source, count in sorted(stats['by_source'].items(), key=lambda x: -x[1]):
            pct = (count / stats['total_quarantined']) * 100
            source_rows += f"""
            <tr>
                <td>{source}</td>
                <td>{count}</td>
                <td>{pct:.1f}%</td>
            </tr>
            """
        
        # Recent quarantines
        recent_rows = ""
        for record in sorted(self.records, key=lambda x: x.timestamp, reverse=True)[:20]:
            recent_rows += f"""
            <tr>
                <td>{record.complex_id}</td>
                <td>{record.pdb_id}</td>
                <td>{record.source_db}</td>
                <td>{record.reason.value.replace('_', ' ').title()}</td>
                <td>{record.details[:100]}...</td>
                <td>{record.timestamp[:10]}</td>
            </tr>
            """
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quarantine Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #C73E1D;
            border-bottom: 3px solid #C73E1D;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #333;
            margin-top: 30px;
        }}
        .stat-box {{
            background: #fff3cd;
            border-left: 4px solid #C73E1D;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #C73E1D;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #dee2e6;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Quarantine Report</h1>
        
        <div class="stat-box">
            <div class="stat-value">{stats['total_quarantined']}</div>
            <div>Total Quarantined Structures</div>
        </div>
        
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
                {reason_rows}
            </tbody>
        </table>
        
        <h2>Source Database Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Source</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                {source_rows}
            </tbody>
        </table>
        
        <h2>Recent Quarantines (Last 20)</h2>
        <table>
            <thead>
                <tr>
                    <th>Complex ID</th>
                    <th>PDB ID</th>
                    <th>Source</th>
                    <th>Reason</th>
                    <th>Details</th>
                    <th>Date</th>
                </tr>
            </thead>
            <tbody>
                {recent_rows}
            </tbody>
        </table>
        
        <p style="margin-top: 30px; color: #666; font-size: 0.9em;">
            <strong>Note:</strong> Quarantined structures are not deleted. They are flagged for review
            and excluded from the canonical dataset. Review quarantine reasons to improve data quality.
        </p>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    # Test quarantine manager
    logging.basicConfig(level=logging.INFO)
    
    manager = QuarantineManager("data/staging/quarantine")
    
    # Add some test records
    manager.quarantine(
        complex_id="1ABC_A_B",
        pdb_id="1ABC",
        source_db="propedia",
        reason=QuarantineReason.MISSING_PEPTIDE_CHAIN,
        details="Peptide chain B not found in structure"
    )
    
    manager.quarantine(
        complex_id="2XYZ_A_C",
        pdb_id="2XYZ",
        source_db="pepbdb",
        reason=QuarantineReason.PEPTIDE_TOO_SHORT,
        details="Peptide chain C only 3 residues (< 5 minimum)"
    )
    
    manager.quarantine(
        complex_id="3DEF_A_B",
        pdb_id="3DEF",
        source_db="biolip2",
        reason=QuarantineReason.AUTH_LABEL_CONFLICT,
        details="Auth and label chain IDs conflict"
    )
    
    # Statistics
    stats = manager.get_statistics()
    print("\nQuarantine Statistics:")
    print(f"  Total: {stats['total_quarantined']}")
    print(f"  By reason: {stats['by_reason']}")
    print(f"  By source: {stats['by_source']}")
    
    # Export report
    manager.export_report("data/staging/quarantine/quarantine_report.html")
    print("\nReport exported to data/staging/quarantine/quarantine_report.html")
