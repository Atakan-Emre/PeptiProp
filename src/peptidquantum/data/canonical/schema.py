"""Canonical schema definitions for PeptidQuantum dataset"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class SourceDatabase(Enum):
    """Source database for complex"""
    PROPEDIA = "propedia"
    PEPBDB = "pepbdb"
    BIOLIP2 = "biolip2"
    CAMP = "camp"
    GEPPRI = "geppri"


class SplitTag(Enum):
    """Dataset split assignment"""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    EXTERNAL = "external"


class QualityFlag(Enum):
    """Quality assessment flag"""
    CLEAN = "clean"
    WARNING = "warning"
    QUARANTINE = "quarantine"


class EntityType(Enum):
    """Biological entity type"""
    PROTEIN = "protein"
    PEPTIDE = "peptide"


@dataclass
class ComplexRecord:
    """Canonical complex record"""
    complex_id: str
    source_db: SourceDatabase
    pdb_id: str
    structure_source: str  # experimental | predicted | hybrid
    structure_format: str  # mmcif | pdb
    resolution: Optional[float]
    protein_chain_id: str
    peptide_chain_id: str
    chain_id_mode: str  # auth | label
    residue_number_mode: str  # auth | label
    peptide_length: int
    protein_length: int
    split_tag: SplitTag
    quality_flag: QualityFlag
    structure_file: Optional[str] = None


@dataclass
class ChainRecord:
    """Canonical chain record"""
    complex_id: str
    chain_id_auth: str
    chain_id_label: str
    entity_type: EntityType
    sequence: str
    length: int


@dataclass
class ResidueRecord:
    """Canonical residue record"""
    complex_id: str
    chain_id: str
    residue_number_auth: int
    residue_number_label: int
    resname: str
    is_interface: bool
    is_pocket: bool
    x: float
    y: float
    z: float
    secondary_structure: Optional[str] = None


@dataclass
class InteractionRecord:
    """Canonical interaction record"""
    complex_id: str
    protein_chain: str
    protein_residue: int
    peptide_chain: str
    peptide_residue: int
    interaction_type: str
    distance: float
    angle: Optional[float]
    tool_source: str  # arpeggio | plip | merged
    confidence: float


@dataclass
class ProvenanceRecord:
    """Provenance and metadata record"""
    complex_id: str
    original_source_url: str
    download_date: str
    parser_version: str
    normalization_version: str
    notes: str


class CanonicalSchema:
    """Schema definitions and validation"""
    
    SCHEMA_VERSION = "1.0.0"
    
    # Peptide length policy
    PEPTIDE_LENGTH_MIN = 5
    PEPTIDE_LENGTH_MAX_CORE = 30
    PEPTIDE_LENGTH_MAX_EXTENSION = 50
    
    # Protein minimum length
    PROTEIN_LENGTH_MIN = 30
    
    # Default ID modes
    DEFAULT_CHAIN_ID_MODE = "auth"
    DEFAULT_RESIDUE_NUMBER_MODE = "auth"
    
    @staticmethod
    def validate_complex(record: ComplexRecord) -> tuple[bool, str]:
        """
        Validate complex record
        
        Returns:
            (is_valid, error_message)
        """
        # Check peptide length
        if record.peptide_length < CanonicalSchema.PEPTIDE_LENGTH_MIN:
            return False, f"Peptide too short: {record.peptide_length} < {CanonicalSchema.PEPTIDE_LENGTH_MIN}"
        
        if record.peptide_length > CanonicalSchema.PEPTIDE_LENGTH_MAX_EXTENSION:
            return False, f"Peptide too long: {record.peptide_length} > {CanonicalSchema.PEPTIDE_LENGTH_MAX_EXTENSION}"
        
        # Check protein length
        if record.protein_length < CanonicalSchema.PROTEIN_LENGTH_MIN:
            return False, f"Protein too short: {record.protein_length} < {CanonicalSchema.PROTEIN_LENGTH_MIN}"
        
        # Check ID modes
        if record.chain_id_mode not in ["auth", "label"]:
            return False, f"Invalid chain_id_mode: {record.chain_id_mode}"
        
        if record.residue_number_mode not in ["auth", "label"]:
            return False, f"Invalid residue_number_mode: {record.residue_number_mode}"
        
        # Check structure format
        if record.structure_format not in {"mmcif", "pdb"}:
            return False, f"Invalid structure_format: {record.structure_format} (must be mmcif or pdb)"
        
        return True, ""
    
    @staticmethod
    def is_core_peptide(length: int) -> bool:
        """Check if peptide length is in core range"""
        return CanonicalSchema.PEPTIDE_LENGTH_MIN <= length <= CanonicalSchema.PEPTIDE_LENGTH_MAX_CORE
    
    @staticmethod
    def is_extension_peptide(length: int) -> bool:
        """Check if peptide length is in extension range"""
        return CanonicalSchema.PEPTIDE_LENGTH_MAX_CORE < length <= CanonicalSchema.PEPTIDE_LENGTH_MAX_EXTENSION
    
    @staticmethod
    def get_parquet_schema() -> dict:
        """Get Parquet schema definitions"""
        return {
            "complexes": {
                "complex_id": "string",
                "source_db": "string",
                "pdb_id": "string",
                "structure_source": "string",
                "structure_format": "string",
                "resolution": "float",
                "protein_chain_id": "string",
                "peptide_chain_id": "string",
                "chain_id_mode": "string",
                "residue_number_mode": "string",
                "peptide_length": "int32",
                "protein_length": "int32",
                "split_tag": "string",
                "quality_flag": "string",
                "structure_file": "string"
            },
            "chains": {
                "complex_id": "string",
                "chain_id_auth": "string",
                "chain_id_label": "string",
                "entity_type": "string",
                "sequence": "string",
                "length": "int32"
            },
            "residues": {
                "complex_id": "string",
                "chain_id": "string",
                "residue_number_auth": "int32",
                "residue_number_label": "int32",
                "resname": "string",
                "is_interface": "bool",
                "is_pocket": "bool",
                "x": "float",
                "y": "float",
                "z": "float",
                "secondary_structure": "string"
            },
            "interactions": {
                "complex_id": "string",
                "protein_chain": "string",
                "protein_residue": "int32",
                "peptide_chain": "string",
                "peptide_residue": "int32",
                "interaction_type": "string",
                "distance": "float",
                "angle": "float",
                "tool_source": "string",
                "confidence": "float"
            },
            "provenance": {
                "complex_id": "string",
                "original_source_url": "string",
                "download_date": "string",
                "parser_version": "string",
                "normalization_version": "string",
                "notes": "string"
            }
        }
