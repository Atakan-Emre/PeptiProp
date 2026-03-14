"""Core data models for protein-peptide complexes"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


class StructureSource(Enum):
    """Source of structure data"""
    EXPERIMENTAL = "experimental"
    PREDICTED = "predicted"
    HYBRID = "hybrid"


class StructureOrigin(Enum):
    """Origin database/tool"""
    RCSB = "RCSB"
    PROPEDIA = "PROPEDIA"
    ALPHAFOLD3 = "AF3"
    ALPHAFOLD_SERVER = "AF_Server"
    ALPHAFOLD_DB = "AF_DB"


@dataclass
class Residue:
    """Single residue in a chain"""
    chain_id: str
    residue_number: int
    residue_name: str
    insertion_code: str = ""
    
    # Coordinates (CA atom)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Properties
    secondary_structure: Optional[str] = None
    solvent_accessibility: Optional[float] = None
    b_factor: Optional[float] = None
    
    def __str__(self):
        return f"{self.chain_id}:{self.residue_name}{self.residue_number}{self.insertion_code}"
    
    @property
    def residue_id(self):
        """Unique residue identifier"""
        return (self.chain_id, self.residue_number, self.insertion_code)


@dataclass
class Chain:
    """Protein or peptide chain"""
    chain_id: str
    chain_type: str  # "protein" or "peptide"
    sequence: str
    residues: List[Residue] = field(default_factory=list)
    
    def __len__(self):
        return len(self.sequence)
    
    def get_residue(self, residue_number: int, insertion_code: str = "") -> Optional[Residue]:
        """Get residue by number"""
        for res in self.residues:
            if res.residue_number == residue_number and res.insertion_code == insertion_code:
                return res
        return None


@dataclass
class Interaction:
    """Single interaction between residues"""
    protein_chain: str
    protein_residue_id: int
    protein_residue_name: str
    
    peptide_chain: str
    peptide_residue_id: int
    peptide_residue_name: str
    
    interaction_type: str
    atom_pair: Optional[str] = None
    distance: Optional[float] = None
    angle: Optional[float] = None
    
    source_tool: str = "unknown"
    confidence: float = 1.0
    
    def __str__(self):
        return (f"{self.protein_chain}:{self.protein_residue_name}{self.protein_residue_id} "
                f"<-{self.interaction_type}-> "
                f"{self.peptide_chain}:{self.peptide_residue_name}{self.peptide_residue_id}")


@dataclass
class Complex:
    """Protein-peptide complex"""
    complex_id: str
    
    # Chains
    protein_chains: List[Chain] = field(default_factory=list)
    peptide_chains: List[Chain] = field(default_factory=list)
    
    # Structure metadata
    structure_source: StructureSource = StructureSource.EXPERIMENTAL
    structure_origin: StructureOrigin = StructureOrigin.RCSB
    
    # Confidence scores
    confidence: Optional[float] = None  # pLDDT or similar
    interface_confidence: Optional[float] = None
    
    # Assembly info
    assembly_used: str = "biological_assembly"
    
    # File paths
    structure_file: Optional[str] = None
    pocket_file: Optional[str] = None
    
    # Interactions
    interactions: List[Interaction] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict = field(default_factory=dict)
    
    def get_protein_chain(self, chain_id: str) -> Optional[Chain]:
        """Get protein chain by ID"""
        for chain in self.protein_chains:
            if chain.chain_id == chain_id:
                return chain
        return None
    
    def get_peptide_chain(self, chain_id: str) -> Optional[Chain]:
        """Get peptide chain by ID"""
        for chain in self.peptide_chains:
            if chain.chain_id == chain_id:
                return chain
        return None
    
    def add_interaction(self, interaction: Interaction):
        """Add interaction to complex"""
        self.interactions.append(interaction)
    
    @property
    def num_interactions(self) -> int:
        """Total number of interactions"""
        return len(self.interactions)
    
    @property
    def interaction_types(self) -> set:
        """Unique interaction types"""
        return {i.interaction_type for i in self.interactions}
