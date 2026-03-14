"""Pipeline configuration management"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class PipelineConfig:
    """Configuration for PeptidQuantum pipeline"""
    
    # Input
    complex_id: Optional[str] = None
    cif_path: Optional[str | Path] = None
    
    # Chain specification
    protein_chain: Optional[str] = None
    peptide_chain: Optional[str] = None
    
    # Processing options
    pocket_radius: float = 8.0
    
    # Chain and residue ID policy (critical for consistency)
    chain_id_mode: str = "auth"  # "auth" or "label"
    residue_number_mode: str = "auth"  # "auth" or "label"
    
    # Interaction extraction
    use_arpeggio: bool = True
    use_plip: bool = True
    
    # Visualization
    generate_pymol: bool = True
    generate_contact_maps: bool = True
    generate_peptide_2d: bool = True
    
    # Output
    generate_report: bool = True
    generate_viewer: bool = True
    output_dir: str | Path = "outputs"
    cache_dir: str | Path = "data/cache"
    
    # Advanced options (for future use)
    use_alphafold: bool = False
    use_molstar: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.complex_id and not self.cif_path:
            raise ValueError("Must provide either complex_id or cif_path")
        
        if self.cif_path:
            self.cif_path = Path(self.cif_path)
        
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        
        # Validate ID modes
        if self.chain_id_mode not in ["auth", "label"]:
            raise ValueError(f"chain_id_mode must be 'auth' or 'label', got: {self.chain_id_mode}")
        
        if self.residue_number_mode not in ["auth", "label"]:
            raise ValueError(f"residue_number_mode must be 'auth' or 'label', got: {self.residue_number_mode}")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PipelineConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str | Path) -> 'PipelineConfig':
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'complex_id': self.complex_id,
            'cif_path': str(self.cif_path) if self.cif_path else None,
            'protein_chain': self.protein_chain,
            'peptide_chain': self.peptide_chain,
            'pocket_radius': self.pocket_radius,
            'chain_id_mode': self.chain_id_mode,
            'residue_number_mode': self.residue_number_mode,
            'use_arpeggio': self.use_arpeggio,
            'use_plip': self.use_plip,
            'generate_pymol': self.generate_pymol,
            'generate_contact_maps': self.generate_contact_maps,
            'generate_peptide_2d': self.generate_peptide_2d,
            'generate_report': self.generate_report,
            'generate_viewer': self.generate_viewer,
            'output_dir': str(self.output_dir),
            'cache_dir': str(self.cache_dir),
            'use_alphafold': self.use_alphafold,
            'use_molstar': self.use_molstar
        }
    
    def to_json(self, json_path: str | Path):
        """Save config to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.cif_path and not Path(self.cif_path).exists():
            raise FileNotFoundError(f"CIF file not found: {self.cif_path}")
        
        if self.pocket_radius <= 0:
            raise ValueError("pocket_radius must be positive")
        
        return True
