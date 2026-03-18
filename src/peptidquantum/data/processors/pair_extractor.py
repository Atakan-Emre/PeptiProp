"""Peptide-protein pair extractor with length policy enforcement"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .mmcif_parser import ChainInfo

logger = logging.getLogger(__name__)


class PeptideCategory(Enum):
    """Peptide length category"""
    TOO_SHORT = "too_short"  # <5 aa
    CORE = "core"  # 5-30 aa
    EXTENSION = "extension"  # 31-50 aa
    TOO_LONG = "too_long"  # >50 aa


class ChainType(Enum):
    """Chain type classification"""
    PROTEIN = "protein"
    PEPTIDE = "peptide"
    AMBIGUOUS = "ambiguous"
    EXCLUDED = "excluded"


@dataclass
class PeptideProteinPair:
    """Peptide-protein complex pair"""
    protein_chain: ChainInfo
    peptide_chain: ChainInfo
    peptide_category: PeptideCategory
    confidence: float  # 0-1
    notes: str = ""


class PeptideProteinPairExtractor:
    """
    Extract peptide-protein pairs from structure
    
    Length Policy (aligned with BioLiP2 and PepBDB):
    - Peptide <5 aa: EXCLUDE
    - Peptide 5-30 aa: CORE (BioLiP2 standard)
    - Peptide 31-50 aa: EXTENSION (PepBDB compatible)
    - Peptide >50 aa: EXCLUDE
    - Protein <30 aa: EXCLUDE/QUARANTINE
    """
    
    # Length thresholds
    PEPTIDE_MIN_LENGTH = 5
    PEPTIDE_CORE_MAX_LENGTH = 30
    PEPTIDE_EXTENSION_MAX_LENGTH = 50
    PROTEIN_MIN_LENGTH = 30
    
    def __init__(
        self,
        allow_extension: bool = True,
        strict_protein_length: bool = True
    ):
        """
        Initialize extractor
        
        Args:
            allow_extension: Allow extension set (31-50 aa)
            strict_protein_length: Enforce protein minimum length
        """
        self.allow_extension = allow_extension
        self.strict_protein_length = strict_protein_length
    
    def extract_pairs(
        self,
        chains: List[ChainInfo],
        protein_chain_id: Optional[str] = None,
        peptide_chain_id: Optional[str] = None
    ) -> Tuple[List[PeptideProteinPair], List[str]]:
        """
        Extract peptide-protein pairs
        
        Args:
            chains: List of chains from structure
            protein_chain_id: Explicit protein chain ID (optional)
            peptide_chain_id: Explicit peptide chain ID (optional)
            
        Returns:
            (pairs, warnings)
        """
        warnings = []
        
        # Classify chains
        chain_classifications = {}
        for chain in chains:
            chain_type, category = self._classify_chain(chain)
            chain_classifications[chain.chain_id_auth] = (chain_type, category)
        
        # Extract pairs
        pairs = []
        
        if protein_chain_id and peptide_chain_id:
            # Explicit pair specification
            protein_chain = self._find_chain(chains, protein_chain_id)
            peptide_chain = self._find_chain(chains, peptide_chain_id)
            
            if protein_chain and peptide_chain:
                pair = self._create_pair(protein_chain, peptide_chain, warnings)
                if pair:
                    pairs.append(pair)
            else:
                if not protein_chain:
                    warnings.append(f"Protein chain not found: {protein_chain_id}")
                if not peptide_chain:
                    warnings.append(f"Peptide chain not found: {peptide_chain_id}")
        
        else:
            # Auto-detect pairs
            protein_chains = []
            peptide_chains = []
            
            for chain in chains:
                chain_type, category = chain_classifications[chain.chain_id_auth]
                
                if chain_type == ChainType.PROTEIN:
                    protein_chains.append(chain)
                elif chain_type == ChainType.PEPTIDE:
                    peptide_chains.append((chain, category))
            
            # Create pairs
            if not protein_chains:
                warnings.append("No protein chains found")
            if not peptide_chains:
                warnings.append("No peptide chains found")
            
            for protein_chain in protein_chains:
                for peptide_chain, category in peptide_chains:
                    pair = self._create_pair(protein_chain, peptide_chain, warnings)
                    if pair:
                        pairs.append(pair)
        
        return pairs, warnings
    
    def _classify_chain(self, chain: ChainInfo) -> Tuple[ChainType, Optional[PeptideCategory]]:
        """
        Classify chain as protein or peptide
        
        Args:
            chain: Chain to classify
            
        Returns:
            (chain_type, peptide_category)
        """
        length = len(chain.residues)
        
        # Too short for protein
        if length < self.PEPTIDE_MIN_LENGTH:
            return ChainType.EXCLUDED, PeptideCategory.TOO_SHORT
        
        # Peptide range
        elif length <= self.PEPTIDE_CORE_MAX_LENGTH:
            return ChainType.PEPTIDE, PeptideCategory.CORE
        
        elif length <= self.PEPTIDE_EXTENSION_MAX_LENGTH:
            if self.allow_extension:
                return ChainType.PEPTIDE, PeptideCategory.EXTENSION
            else:
                return ChainType.AMBIGUOUS, None
        
        # Protein range
        elif length >= self.PROTEIN_MIN_LENGTH:
            return ChainType.PROTEIN, None
        
        # Ambiguous (between peptide extension and protein minimum)
        else:
            return ChainType.AMBIGUOUS, None
    
    def _create_pair(
        self,
        protein_chain: ChainInfo,
        peptide_chain: ChainInfo,
        warnings: List[str]
    ) -> Optional[PeptideProteinPair]:
        """Create peptide-protein pair with validation"""
        
        # Validate protein
        protein_length = len(protein_chain.residues)
        if self.strict_protein_length and protein_length < self.PROTEIN_MIN_LENGTH:
            warnings.append(
                f"Protein chain {protein_chain.chain_id_auth} too short: "
                f"{protein_length} < {self.PROTEIN_MIN_LENGTH}"
            )
            return None
        
        # Validate peptide
        peptide_length = len(peptide_chain.residues)
        
        if peptide_length < self.PEPTIDE_MIN_LENGTH:
            warnings.append(
                f"Peptide chain {peptide_chain.chain_id_auth} too short: "
                f"{peptide_length} < {self.PEPTIDE_MIN_LENGTH}"
            )
            return None
        
        elif peptide_length > self.PEPTIDE_EXTENSION_MAX_LENGTH:
            warnings.append(
                f"Peptide chain {peptide_chain.chain_id_auth} too long: "
                f"{peptide_length} > {self.PEPTIDE_EXTENSION_MAX_LENGTH}"
            )
            return None
        
        # Determine category
        if peptide_length <= self.PEPTIDE_CORE_MAX_LENGTH:
            category = PeptideCategory.CORE
            confidence = 1.0
        elif peptide_length <= self.PEPTIDE_EXTENSION_MAX_LENGTH:
            category = PeptideCategory.EXTENSION
            confidence = 0.8  # Lower confidence for extension set
        else:
            return None
        
        # Create pair
        notes = []
        
        if protein_length < self.PROTEIN_MIN_LENGTH:
            notes.append(f"Protein length {protein_length} < {self.PROTEIN_MIN_LENGTH}")
            confidence *= 0.7
        
        pair = PeptideProteinPair(
            protein_chain=protein_chain,
            peptide_chain=peptide_chain,
            peptide_category=category,
            confidence=confidence,
            notes="; ".join(notes)
        )
        
        return pair
    
    def _find_chain(self, chains: List[ChainInfo], chain_id: str) -> Optional[ChainInfo]:
        """Find chain by ID"""
        for chain in chains:
            if chain.chain_id_auth == chain_id or chain.chain_id_label == chain_id:
                return chain
        return None
    
    def get_statistics(self, pairs: List[PeptideProteinPair]) -> dict:
        """Get pair extraction statistics"""
        if not pairs:
            return {
                "total_pairs": 0,
                "core_pairs": 0,
                "extension_pairs": 0,
                "avg_peptide_length": 0,
                "avg_protein_length": 0,
                "avg_confidence": 0
            }
        
        core_pairs = [p for p in pairs if p.peptide_category == PeptideCategory.CORE]
        extension_pairs = [p for p in pairs if p.peptide_category == PeptideCategory.EXTENSION]
        
        peptide_lengths = [len(p.peptide_chain.residues) for p in pairs]
        protein_lengths = [len(p.protein_chain.residues) for p in pairs]
        confidences = [p.confidence for p in pairs]
        
        return {
            "total_pairs": len(pairs),
            "core_pairs": len(core_pairs),
            "extension_pairs": len(extension_pairs),
            "avg_peptide_length": sum(peptide_lengths) / len(peptide_lengths),
            "avg_protein_length": sum(protein_lengths) / len(protein_lengths),
            "avg_confidence": sum(confidences) / len(confidences),
            "peptide_length_range": (min(peptide_lengths), max(peptide_lengths)),
            "protein_length_range": (min(protein_lengths), max(protein_lengths))
        }


class PairValidator:
    """Validate extracted pairs"""
    
    @staticmethod
    def validate_pair(pair: PeptideProteinPair) -> Tuple[bool, List[str]]:
        """
        Validate a single pair
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check protein
        if not pair.protein_chain.residues:
            errors.append("Protein chain has no residues")
        
        # Check peptide
        if not pair.peptide_chain.residues:
            errors.append("Peptide chain has no residues")
        
        # Check category
        peptide_length = len(pair.peptide_chain.residues)
        
        if pair.peptide_category == PeptideCategory.CORE:
            if not (5 <= peptide_length <= 30):
                errors.append(
                    f"Core peptide length {peptide_length} outside 5-30 range"
                )
        
        elif pair.peptide_category == PeptideCategory.EXTENSION:
            if not (31 <= peptide_length <= 50):
                errors.append(
                    f"Extension peptide length {peptide_length} outside 31-50 range"
                )
        
        # Check confidence
        if not (0 <= pair.confidence <= 1):
            errors.append(f"Invalid confidence: {pair.confidence}")
        
        is_valid = len(errors) == 0
        
        return is_valid, errors
    
    @staticmethod
    def validate_pairs(pairs: List[PeptideProteinPair]) -> Tuple[bool, List[str]]:
        """Validate all pairs"""
        all_errors = []
        
        for i, pair in enumerate(pairs):
            is_valid, errors = PairValidator.validate_pair(pair)
            if not is_valid:
                all_errors.append(f"Pair {i}: {'; '.join(errors)}")
        
        is_valid = len(all_errors) == 0
        
        return is_valid, all_errors


if __name__ == "__main__":
    # Test extractor
    logging.basicConfig(level=logging.INFO)
    
    # Create mock chains
    from .mmcif_parser import ChainInfo, ResidueInfo
    
    protein_chain = ChainInfo(
        chain_id_auth="A",
        chain_id_label="A",
        entity_id="1",
        entity_type="polymer",
        sequence="M" * 100,
        residues=[ResidueInfo(i, i, "ALA", "A", "A", [], None) for i in range(100)]
    )
    
    peptide_chain_core = ChainInfo(
        chain_id_auth="B",
        chain_id_label="B",
        entity_id="2",
        entity_type="polymer",
        sequence="G" * 15,
        residues=[ResidueInfo(i, i, "GLY", "B", "B", [], None) for i in range(15)]
    )
    
    peptide_chain_extension = ChainInfo(
        chain_id_auth="C",
        chain_id_label="C",
        entity_id="3",
        entity_type="polymer",
        sequence="A" * 40,
        residues=[ResidueInfo(i, i, "ALA", "C", "C", [], None) for i in range(40)]
    )
    
    # Test extraction
    extractor = PeptideProteinPairExtractor(allow_extension=True)
    
    chains = [protein_chain, peptide_chain_core, peptide_chain_extension]
    pairs, warnings = extractor.extract_pairs(chains)
    
    print(f"\nExtracted {len(pairs)} pairs")
    for i, pair in enumerate(pairs):
        print(f"\nPair {i+1}:")
        print(f"  Protein: {pair.protein_chain.chain_id_auth} ({len(pair.protein_chain.residues)} aa)")
        print(f"  Peptide: {pair.peptide_chain.chain_id_auth} ({len(pair.peptide_chain.residues)} aa)")
        print(f"  Category: {pair.peptide_category.value}")
        print(f"  Confidence: {pair.confidence:.2f}")
    
    if warnings:
        print(f"\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Statistics
    stats = extractor.get_statistics(pairs)
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
