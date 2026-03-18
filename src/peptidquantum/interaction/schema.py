"""Unified interaction schema for Arpeggio and PLIP outputs"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class InteractionType(Enum):
    """Standardized interaction types"""
    HBOND = "hydrogen_bond"
    SALT_BRIDGE = "salt_bridge"
    HYDROPHOBIC = "hydrophobic"
    PI_STACKING = "pi_stacking"
    CATION_PI = "cation_pi"
    VDW = "van_der_waals"
    HALOGEN = "halogen_bond"
    METAL = "metal_coordination"
    IONIC = "ionic"
    CARBONYL = "carbonyl"
    DONOR_PI = "donor_pi"
    HALOGEN_PI = "halogen_pi"
    CARBON_PI = "carbon_pi"
    WEAK_HBOND = "weak_hydrogen_bond"
    WATER_BRIDGE = "water_bridge"
    
    @classmethod
    def from_arpeggio(cls, arpeggio_type: str) -> Optional['InteractionType']:
        """Map Arpeggio interaction type to standard"""
        mapping = {
            'INTER': cls.VDW,
            'INTRA_SELECTION': cls.VDW,
            'SELECTION_WATER': cls.WATER_BRIDGE,
            'INTRA_NON_SELECTION': cls.VDW,
            'PROXIMAL': cls.VDW,
            'CLASH': cls.VDW,
            'COVALENT': None,  # Skip covalent
            'VDW_CLASH': cls.VDW,
            'VDW': cls.VDW,
            'HBOND': cls.HBOND,
            'WEAK_HBOND': cls.WEAK_HBOND,
            'HALOGEN_BOND': cls.HALOGEN,
            'IONIC': cls.IONIC,
            'METAL_COMPLEX': cls.METAL,
            'AROMATIC': cls.PI_STACKING,
            'HYDROPHOBIC': cls.HYDROPHOBIC,
            'CARBONYL': cls.CARBONYL,
            'POLAR': cls.HBOND,
            'WEAK_POLAR': cls.WEAK_HBOND,
        }
        return mapping.get(arpeggio_type.upper())
    
    @classmethod
    def from_plip(cls, plip_type: str) -> Optional['InteractionType']:
        """Map PLIP interaction type to standard"""
        mapping = {
            'hbond': cls.HBOND,
            'saltbridge': cls.SALT_BRIDGE,
            'hydrophobic': cls.HYDROPHOBIC,
            'pistacking': cls.PI_STACKING,
            'pication': cls.CATION_PI,
            'halogen': cls.HALOGEN,
            'waterbridge': cls.WATER_BRIDGE,
            'metal': cls.METAL,
        }
        return mapping.get(plip_type.lower())


@dataclass
class StandardizedInteraction:
    """Standardized interaction format"""
    # Protein side
    protein_chain: str
    protein_residue_id: int
    protein_residue_name: str
    
    # Peptide side
    peptide_chain: str
    peptide_residue_id: int
    peptide_residue_name: str

    # Optional atom-level detail
    protein_atom: Optional[str] = None
    peptide_atom: Optional[str] = None
    
    # Interaction details
    interaction_type: InteractionType = InteractionType.VDW
    distance: Optional[float] = None
    angle: Optional[float] = None
    
    # Metadata
    source_tool: str = "unknown"  # "arpeggio" or "plip"
    confidence: float = 1.0
    raw_type: Optional[str] = None  # Original type from tool
    
    def __str__(self):
        return (f"{self.protein_chain}:{self.protein_residue_name}{self.protein_residue_id}"
                f"({self.protein_atom or 'any'}) "
                f"<-{self.interaction_type.value}-> "
                f"{self.peptide_chain}:{self.peptide_residue_name}{self.peptide_residue_id}"
                f"({self.peptide_atom or 'any'})")
    
    @property
    def residue_pair_key(self) -> tuple:
        """Unique key for residue pair (ignoring atoms)"""
        return (
            self.protein_chain, self.protein_residue_id,
            self.peptide_chain, self.peptide_residue_id,
            self.interaction_type
        )
    
    @property
    def atom_pair_key(self) -> tuple:
        """Unique key for atom pair"""
        return (
            self.protein_chain, self.protein_residue_id, self.protein_atom,
            self.peptide_chain, self.peptide_residue_id, self.peptide_atom,
            self.interaction_type
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'protein_chain': self.protein_chain,
            'protein_residue_id': self.protein_residue_id,
            'protein_residue_name': self.protein_residue_name,
            'protein_atom': self.protein_atom,
            'peptide_chain': self.peptide_chain,
            'peptide_residue_id': self.peptide_residue_id,
            'peptide_residue_name': self.peptide_residue_name,
            'peptide_atom': self.peptide_atom,
            'interaction_type': self.interaction_type.value,
            'distance': self.distance,
            'angle': self.angle,
            'source_tool': self.source_tool,
            'confidence': self.confidence,
            'raw_type': self.raw_type
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StandardizedInteraction':
        """Create from dictionary"""
        data = data.copy()
        if 'interaction_type' in data and isinstance(data['interaction_type'], str):
            data['interaction_type'] = InteractionType(data['interaction_type'])
        return cls(**data)


@dataclass
class InteractionSet:
    """Collection of interactions for a complex"""
    complex_id: str
    interactions: List[StandardizedInteraction]
    
    def filter_by_type(self, interaction_type: InteractionType) -> List[StandardizedInteraction]:
        """Get interactions of specific type"""
        return [i for i in self.interactions if i.interaction_type == interaction_type]
    
    def filter_by_protein_chain(self, chain_id: str) -> List[StandardizedInteraction]:
        """Get interactions for specific protein chain"""
        return [i for i in self.interactions if i.protein_chain == chain_id]
    
    def filter_by_peptide_chain(self, chain_id: str) -> List[StandardizedInteraction]:
        """Get interactions for specific peptide chain"""
        return [i for i in self.interactions if i.peptide_chain == chain_id]
    
    def get_unique_residue_pairs(self) -> set:
        """Get unique residue pairs"""
        return {i.residue_pair_key for i in self.interactions}
    
    def get_interaction_types(self) -> set:
        """Get all interaction types present"""
        return {i.interaction_type for i in self.interactions}
    
    def count_by_type(self) -> dict:
        """Count interactions by type"""
        counts = {}
        for interaction in self.interactions:
            itype = interaction.interaction_type
            counts[itype] = counts.get(itype, 0) + 1
        return counts
    
    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        import pandas as pd
        return pd.DataFrame([i.to_dict() for i in self.interactions])
    
    def save_tsv(self, output_file: str):
        """Save to TSV file"""
        df = self.to_dataframe()
        df.to_csv(output_file, sep='\t', index=False)
    
    def save_json(self, output_file: str):
        """Save to JSON file"""
        import json
        data = {
            'complex_id': self.complex_id,
            'num_interactions': len(self.interactions),
            'interaction_types': [t.value for t in self.get_interaction_types()],
            'interactions': [i.to_dict() for i in self.interactions]
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
