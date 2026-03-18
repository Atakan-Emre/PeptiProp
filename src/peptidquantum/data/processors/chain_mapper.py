"""Chain and residue ID mapper for auth/label consistency"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MappingStatus(Enum):
    """Chain/residue mapping status"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"


@dataclass
class ChainMapping:
    """Chain ID mapping between auth and label"""
    chain_id_auth: str
    chain_id_label: str
    entity_id: Optional[str]
    entity_type: str  # polymer, non-polymer, water
    mapping_status: MappingStatus
    notes: str = ""


@dataclass
class ResidueMapping:
    """Residue number mapping between auth and label"""
    chain_id_auth: str
    residue_number_auth: int
    residue_number_label: int
    resname: str
    mapping_status: MappingStatus
    notes: str = ""


class ChainResidueMapper:
    """
    Map chain and residue IDs between auth and label modes
    
    Critical for consistency across:
    - TSV exports
    - Contact maps
    - PyMOL labels
    - HTML reports
    - Graph builders
    """
    
    def __init__(self, default_mode: str = "auth"):
        """
        Initialize mapper
        
        Args:
            default_mode: Default ID mode ('auth' or 'label')
        """
        if default_mode not in ["auth", "label"]:
            raise ValueError(f"Invalid mode: {default_mode}")
        
        self.default_mode = default_mode
        self.chain_mappings: Dict[str, ChainMapping] = {}
        self.residue_mappings: Dict[Tuple[str, int], ResidueMapping] = {}
    
    def add_chain_mapping(
        self,
        chain_id_auth: str,
        chain_id_label: str,
        entity_id: Optional[str] = None,
        entity_type: str = "polymer"
    ) -> MappingStatus:
        """
        Add chain mapping
        
        Args:
            chain_id_auth: Author chain ID
            chain_id_label: Label chain ID
            entity_id: Entity ID
            entity_type: Entity type
            
        Returns:
            Mapping status
        """
        # Check for conflicts
        if chain_id_auth in self.chain_mappings:
            existing = self.chain_mappings[chain_id_auth]
            if existing.chain_id_label != chain_id_label:
                logger.warning(
                    f"Chain mapping conflict: {chain_id_auth} -> "
                    f"{existing.chain_id_label} vs {chain_id_label}"
                )
                status = MappingStatus.AMBIGUOUS
                notes = f"Conflict with existing mapping"
            else:
                status = MappingStatus.SUCCESS
                notes = "Duplicate mapping (consistent)"
        else:
            status = MappingStatus.SUCCESS
            notes = ""
        
        mapping = ChainMapping(
            chain_id_auth=chain_id_auth,
            chain_id_label=chain_id_label,
            entity_id=entity_id,
            entity_type=entity_type,
            mapping_status=status,
            notes=notes
        )
        
        self.chain_mappings[chain_id_auth] = mapping
        
        return status
    
    def add_residue_mapping(
        self,
        chain_id_auth: str,
        residue_number_auth: int,
        residue_number_label: int,
        resname: str
    ) -> MappingStatus:
        """
        Add residue mapping
        
        Args:
            chain_id_auth: Author chain ID
            residue_number_auth: Author residue number
            residue_number_label: Label residue number
            resname: Residue name
            
        Returns:
            Mapping status
        """
        key = (chain_id_auth, residue_number_auth)
        
        # Check for conflicts
        if key in self.residue_mappings:
            existing = self.residue_mappings[key]
            if existing.residue_number_label != residue_number_label:
                logger.warning(
                    f"Residue mapping conflict: {chain_id_auth}:{residue_number_auth} -> "
                    f"{existing.residue_number_label} vs {residue_number_label}"
                )
                status = MappingStatus.AMBIGUOUS
                notes = f"Conflict with existing mapping"
            else:
                status = MappingStatus.SUCCESS
                notes = "Duplicate mapping (consistent)"
        else:
            status = MappingStatus.SUCCESS
            notes = ""
        
        mapping = ResidueMapping(
            chain_id_auth=chain_id_auth,
            residue_number_auth=residue_number_auth,
            residue_number_label=residue_number_label,
            resname=resname,
            mapping_status=status,
            notes=notes
        )
        
        self.residue_mappings[key] = mapping
        
        return status
    
    def get_chain_id(self, chain_id_auth: str, mode: Optional[str] = None) -> Optional[str]:
        """
        Get chain ID in specified mode
        
        Args:
            chain_id_auth: Author chain ID
            mode: ID mode ('auth' or 'label', default: self.default_mode)
            
        Returns:
            Chain ID in specified mode
        """
        mode = mode or self.default_mode
        
        if mode == "auth":
            return chain_id_auth
        
        elif mode == "label":
            mapping = self.chain_mappings.get(chain_id_auth)
            if mapping:
                return mapping.chain_id_label
            else:
                logger.warning(f"No mapping found for chain {chain_id_auth}")
                return chain_id_auth  # Fallback to auth
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def get_residue_number(
        self,
        chain_id_auth: str,
        residue_number_auth: int,
        mode: Optional[str] = None
    ) -> Optional[int]:
        """
        Get residue number in specified mode
        
        Args:
            chain_id_auth: Author chain ID
            residue_number_auth: Author residue number
            mode: ID mode ('auth' or 'label', default: self.default_mode)
            
        Returns:
            Residue number in specified mode
        """
        mode = mode or self.default_mode
        
        if mode == "auth":
            return residue_number_auth
        
        elif mode == "label":
            key = (chain_id_auth, residue_number_auth)
            mapping = self.residue_mappings.get(key)
            if mapping:
                return mapping.residue_number_label
            else:
                logger.warning(
                    f"No mapping found for residue {chain_id_auth}:{residue_number_auth}"
                )
                return residue_number_auth  # Fallback to auth
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def validate_mappings(self) -> Tuple[bool, List[str]]:
        """
        Validate all mappings
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check for ambiguous chain mappings
        ambiguous_chains = [
            m.chain_id_auth for m in self.chain_mappings.values()
            if m.mapping_status == MappingStatus.AMBIGUOUS
        ]
        
        if ambiguous_chains:
            errors.append(f"Ambiguous chain mappings: {ambiguous_chains}")
        
        # Check for ambiguous residue mappings
        ambiguous_residues = [
            f"{m.chain_id_auth}:{m.residue_number_auth}"
            for m in self.residue_mappings.values()
            if m.mapping_status == MappingStatus.AMBIGUOUS
        ]
        
        if ambiguous_residues:
            errors.append(f"Ambiguous residue mappings: {ambiguous_residues[:10]}...")
        
        # Check for missing mappings
        if not self.chain_mappings:
            errors.append("No chain mappings found")
        
        is_valid = len(errors) == 0
        
        return is_valid, errors
    
    def get_mapping_summary(self) -> Dict:
        """Get mapping summary statistics"""
        chain_statuses = {}
        for mapping in self.chain_mappings.values():
            status = mapping.mapping_status.value
            chain_statuses[status] = chain_statuses.get(status, 0) + 1
        
        residue_statuses = {}
        for mapping in self.residue_mappings.values():
            status = mapping.mapping_status.value
            residue_statuses[status] = residue_statuses.get(status, 0) + 1
        
        return {
            "total_chains": len(self.chain_mappings),
            "chain_statuses": chain_statuses,
            "total_residues": len(self.residue_mappings),
            "residue_statuses": residue_statuses,
            "default_mode": self.default_mode
        }
    
    def export_mappings(self, output_dir: str | Path):
        """Export mappings to CSV files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export chain mappings
        import csv
        
        chain_file = output_dir / "chain_mappings.csv"
        with open(chain_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'chain_id_auth', 'chain_id_label', 'entity_id',
                'entity_type', 'mapping_status', 'notes'
            ])
            
            for mapping in self.chain_mappings.values():
                writer.writerow([
                    mapping.chain_id_auth,
                    mapping.chain_id_label,
                    mapping.entity_id or '',
                    mapping.entity_type,
                    mapping.mapping_status.value,
                    mapping.notes
                ])
        
        logger.info(f"Chain mappings exported to {chain_file}")
        
        # Export residue mappings
        residue_file = output_dir / "residue_mappings.csv"
        with open(residue_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'chain_id_auth', 'residue_number_auth', 'residue_number_label',
                'resname', 'mapping_status', 'notes'
            ])
            
            for mapping in self.residue_mappings.values():
                writer.writerow([
                    mapping.chain_id_auth,
                    mapping.residue_number_auth,
                    mapping.residue_number_label,
                    mapping.resname,
                    mapping.mapping_status.value,
                    mapping.notes
                ])
        
        logger.info(f"Residue mappings exported to {residue_file}")


class MappingValidator:
    """Validate mapping consistency across outputs"""
    
    @staticmethod
    def check_consistency(
        mapper: ChainResidueMapper,
        outputs: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Check if outputs use consistent ID mode
        
        Args:
            mapper: ChainResidueMapper instance
            outputs: List of output dictionaries with chain/residue IDs
            
        Returns:
            (is_consistent, error_messages)
        """
        errors = []
        
        # Check each output
        for i, output in enumerate(outputs):
            output_name = output.get('name', f'output_{i}')
            
            # Check chain IDs
            if 'chain_id' in output:
                chain_id = output['chain_id']
                
                # Verify it exists in mapper
                if chain_id not in mapper.chain_mappings:
                    # Check if it's a label ID
                    found = False
                    for mapping in mapper.chain_mappings.values():
                        if mapping.chain_id_label == chain_id:
                            found = True
                            break
                    
                    if not found:
                        errors.append(
                            f"{output_name}: Unknown chain ID '{chain_id}'"
                        )
            
            # Check residue numbers
            if 'residues' in output:
                for res in output['residues']:
                    chain_id = res.get('chain_id')
                    res_num = res.get('residue_number')
                    
                    if chain_id and res_num:
                        key = (chain_id, res_num)
                        if key not in mapper.residue_mappings:
                            errors.append(
                                f"{output_name}: Unknown residue {chain_id}:{res_num}"
                            )
        
        is_consistent = len(errors) == 0
        
        return is_consistent, errors


if __name__ == "__main__":
    # Test mapper
    logging.basicConfig(level=logging.INFO)
    
    mapper = ChainResidueMapper(default_mode="auth")
    
    # Add some mappings
    mapper.add_chain_mapping("A", "A", entity_id="1", entity_type="polymer")
    mapper.add_chain_mapping("B", "B", entity_id="2", entity_type="polymer")
    
    mapper.add_residue_mapping("A", 1, 1, "MET")
    mapper.add_residue_mapping("A", 2, 2, "ALA")
    mapper.add_residue_mapping("B", 1, 1, "GLY")
    
    # Validate
    is_valid, errors = mapper.validate_mappings()
    print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Summary
    summary = mapper.get_mapping_summary()
    print(f"\nSummary:")
    print(f"  Chains: {summary['total_chains']}")
    print(f"  Residues: {summary['total_residues']}")
    print(f"  Default mode: {summary['default_mode']}")
