"""Canonical dataset builder - staging to parquet conversion"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

from .mmcif_parser import MMCIFStructureParser
from .chain_mapper import ChainResidueMapper, MappingStatus
from .pair_extractor import PeptideProteinPairExtractor, PeptideCategory
from .quarantine_manager import QuarantineManager, QuarantineReason
from ..canonical.schema import (
    ComplexRecord, ChainRecord, ResidueRecord, ProvenanceRecord,
    SourceDatabase, SplitTag, QualityFlag, EntityType, CanonicalSchema
)

logger = logging.getLogger(__name__)


class CanonicalBuilder:
    """
    Build canonical dataset from staging data
    
    Pipeline: staging → canonical parquet files
    
    Output:
    - complexes.parquet
    - chains.parquet
    - residues.parquet
    - interactions.parquet (placeholder for now)
    - provenance.parquet
    """
    
    def __init__(
        self,
        staging_dir: str | Path,
        canonical_dir: str | Path,
        chain_id_mode: str = "auth",
        residue_number_mode: str = "auth",
        max_pairs_per_structure: int = 50
    ):
        """
        Initialize builder
        
        Args:
            staging_dir: Staging directory
            canonical_dir: Canonical output directory
            chain_id_mode: Chain ID mode (auth/label)
            residue_number_mode: Residue number mode (auth/label)
            max_pairs_per_structure: Maximum pairs per structure (prevent explosion)
        """
        self.staging_dir = Path(staging_dir)
        self.canonical_dir = Path(canonical_dir)
        self.chain_id_mode = chain_id_mode
        self.residue_number_mode = residue_number_mode
        self.max_pairs_per_structure = max_pairs_per_structure
        
        # Create directories
        self.canonical_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = MMCIFStructureParser(use_auth_ids=(chain_id_mode == "auth"))
        self.pair_extractor = PeptideProteinPairExtractor(allow_extension=True)
        self.quarantine_manager = QuarantineManager(self.staging_dir / "quarantine")
        
        # Records
        self.complex_records: List[ComplexRecord] = []
        self.chain_records: List[ChainRecord] = []
        self.residue_records: List[ResidueRecord] = []
        self.provenance_records: List[ProvenanceRecord] = []
        
        # Track seen complex_key to prevent duplicates
        self.seen_complex_keys: set = set()
    
    def build(
        self,
        source_files: List[Path],
        source_db: SourceDatabase,
        batch_size: int = 100
    ):
        """
        Build canonical dataset from source files
        
        Args:
            source_files: List of mmCIF files
            source_db: Source database
            batch_size: Batch size for processing
        """
        logger.info(f"Building canonical dataset from {len(source_files)} files")
        logger.info(f"Source: {source_db.value}")
        logger.info(f"Chain ID mode: {self.chain_id_mode}")
        logger.info(f"Residue number mode: {self.residue_number_mode}")
        
        processed = 0
        quarantined = 0
        
        for i, cif_file in enumerate(source_files):
            try:
                # Process file
                success = self._process_file(cif_file, source_db)
                
                if success:
                    processed += 1
                else:
                    quarantined += 1
                
                # Progress
                if (i + 1) % batch_size == 0:
                    logger.info(
                        f"Progress: {i+1}/{len(source_files)} "
                        f"(processed: {processed}, quarantined: {quarantined})"
                    )
                
            except Exception as e:
                logger.error(f"Failed to process {cif_file.name}: {e}")
                quarantined += 1
        
        # Save to parquet
        self._save_parquet()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Canonical dataset build complete")
        logger.info(f"  Processed: {processed}")
        logger.info(f"  Quarantined: {quarantined}")
        logger.info(f"  Output: {self.canonical_dir}")
        logger.info(f"{'='*60}")
    
    def _process_file(self, cif_file: Path, source_db: SourceDatabase) -> bool:
        """
        Process single mmCIF file
        
        Returns:
            True if successful, False if quarantined
        """
        file_stem = cif_file.stem
        pdb_id = file_stem.split("_")[0].upper()
        
        structure_format = "mmcif" if cif_file.suffix.lower() == ".cif" else "pdb"

        # Parse structure
        try:
            parsed = self.parser.parse(cif_file)
        except Exception as e:
            self.quarantine_manager.quarantine(
                complex_id=f"{file_stem.upper()}_parse_error",
                pdb_id=pdb_id,
                source_db=source_db.value,
                reason=QuarantineReason.STRUCTURE_PARSE_ERROR,
                details=str(e),
                structure_file=cif_file
            )
            return False
        
        chains = parsed['chains']
        
        if not chains:
            self.quarantine_manager.quarantine(
                complex_id=f"{file_stem.upper()}_no_chains",
                pdb_id=pdb_id,
                source_db=source_db.value,
                reason=QuarantineReason.INVALID_RESIDUES,
                details="No chains found in structure",
                structure_file=cif_file
            )
            return False
        
        # Extract pairs
        pairs, warnings = self.pair_extractor.extract_pairs(chains)
        
        if not pairs:
            reason = QuarantineReason.MISSING_PEPTIDE_CHAIN
            if any("protein" in w.lower() for w in warnings):
                reason = QuarantineReason.MISSING_PROTEIN_CHAIN
            
            self.quarantine_manager.quarantine(
                complex_id=f"{file_stem.upper()}_no_pairs",
                pdb_id=pdb_id,
                source_db=source_db.value,
                reason=reason,
                details="; ".join(warnings),
                structure_file=cif_file
            )
            return False
        
        # Check for pair explosion
        if len(pairs) > self.max_pairs_per_structure:
            logger.warning(
                f"{pdb_id}: {len(pairs)} pairs exceeds limit {self.max_pairs_per_structure}, "
                f"taking top {self.max_pairs_per_structure} by confidence"
            )
            # Sort by confidence and take top N
            pairs = sorted(pairs, key=lambda p: p.confidence, reverse=True)[:self.max_pairs_per_structure]
        
        # Process each pair
        for pair in pairs:
            complex_id = f"{pdb_id}_{pair.protein_chain.chain_id_auth}_{pair.peptide_chain.chain_id_auth}"
            
            # Check for duplicate complex_key
            complex_key = f"{pdb_id}_{pair.protein_chain.chain_id_auth}_{pair.peptide_chain.chain_id_auth}"
            if complex_key in self.seen_complex_keys:
                logger.debug(f"Skipping duplicate complex_key: {complex_key}")
                continue
            
            self.seen_complex_keys.add(complex_key)
            
            # Create complex record
            complex_record = ComplexRecord(
                complex_id=complex_id,
                source_db=source_db,
                pdb_id=pdb_id,
                structure_source="experimental",  # Assume experimental for now
                structure_format=structure_format,
                resolution=None,  # TODO: Extract from mmCIF
                protein_chain_id=pair.protein_chain.chain_id_auth,
                peptide_chain_id=pair.peptide_chain.chain_id_auth,
                chain_id_mode=self.chain_id_mode,
                residue_number_mode=self.residue_number_mode,
                peptide_length=len(pair.peptide_chain.residues),
                protein_length=len(pair.protein_chain.residues),
                split_tag=SplitTag.TRAIN,  # Will be updated in split phase
                quality_flag=QualityFlag.CLEAN if pair.confidence >= 0.9 else QualityFlag.WARNING,
                structure_file=str(cif_file)
            )
            
            # Validate
            is_valid, error_msg = CanonicalSchema.validate_complex(complex_record)
            if not is_valid:
                self.quarantine_manager.quarantine(
                    complex_id=complex_id,
                    pdb_id=pdb_id,
                    source_db=source_db.value,
                    reason=QuarantineReason.OTHER,
                    details=error_msg,
                    structure_file=cif_file
                )
                continue
            
            self.complex_records.append(complex_record)
            
            # Create chain records
            for chain in [pair.protein_chain, pair.peptide_chain]:
                entity_type = EntityType.PROTEIN if chain == pair.protein_chain else EntityType.PEPTIDE
                
                chain_record = ChainRecord(
                    complex_id=complex_id,
                    chain_id_auth=chain.chain_id_auth,
                    chain_id_label=chain.chain_id_label,
                    entity_type=entity_type,
                    sequence=chain.sequence,
                    length=len(chain.residues)
                )
                
                self.chain_records.append(chain_record)
                
                # Create residue records
                for residue in chain.residues:
                    residue_record = ResidueRecord(
                        complex_id=complex_id,
                        chain_id=residue.chain_id_auth,
                        residue_number_auth=residue.residue_number_auth,
                        residue_number_label=residue.residue_number_label,
                        resname=residue.resname,
                        is_interface=False,  # Will be updated in interaction phase
                        is_pocket=False,  # Will be updated in interaction phase
                        x=residue.centroid[0] if residue.centroid else 0.0,
                        y=residue.centroid[1] if residue.centroid else 0.0,
                        z=residue.centroid[2] if residue.centroid else 0.0,
                        secondary_structure=None
                    )
                    
                    self.residue_records.append(residue_record)
            
            # Create provenance record
            provenance_record = ProvenanceRecord(
                complex_id=complex_id,
                original_source_url=(
                    f"https://files.rcsb.org/download/{pdb_id}.cif"
                    if structure_format == "mmcif"
                    else f"https://propedia.russelllab.org/download/{cif_file.name}"
                ),
                download_date=datetime.now().isoformat(),
                parser_version=CanonicalSchema.SCHEMA_VERSION,
                normalization_version=CanonicalSchema.SCHEMA_VERSION,
                notes=f"Parsed with {parsed['parser']}, confidence: {pair.confidence:.2f}"
            )
            
            self.provenance_records.append(provenance_record)
        
        return True
    
    def _save_parquet(self):
        """Save records to parquet files"""
        logger.info("Saving canonical dataset to parquet...")
        
        # Complexes
        if self.complex_records:
            df = pd.DataFrame([
                {
                    'complex_id': r.complex_id,
                    'source_db': r.source_db.value,
                    'pdb_id': r.pdb_id,
                    'structure_source': r.structure_source,
                    'structure_format': r.structure_format,
                    'resolution': r.resolution,
                    'protein_chain_id': r.protein_chain_id,
                    'peptide_chain_id': r.peptide_chain_id,
                    'chain_id_mode': r.chain_id_mode,
                    'residue_number_mode': r.residue_number_mode,
                    'peptide_length': r.peptide_length,
                    'protein_length': r.protein_length,
                    'split_tag': r.split_tag.value,
                    'quality_flag': r.quality_flag.value,
                    'structure_file': r.structure_file
                }
                for r in self.complex_records
            ])
            
            output_file = self.canonical_dir / "complexes.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"✓ Saved {len(df)} complexes to {output_file}")
        
        # Chains
        if self.chain_records:
            df = pd.DataFrame([
                {
                    'complex_id': r.complex_id,
                    'chain_id_auth': r.chain_id_auth,
                    'chain_id_label': r.chain_id_label,
                    'entity_type': r.entity_type.value,
                    'sequence': r.sequence,
                    'length': r.length
                }
                for r in self.chain_records
            ])
            
            output_file = self.canonical_dir / "chains.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"✓ Saved {len(df)} chains to {output_file}")
        
        # Residues
        if self.residue_records:
            df = pd.DataFrame([
                {
                    'complex_id': r.complex_id,
                    'chain_id': r.chain_id,
                    'residue_number_auth': r.residue_number_auth,
                    'residue_number_label': r.residue_number_label,
                    'resname': r.resname,
                    'is_interface': r.is_interface,
                    'is_pocket': r.is_pocket,
                    'x': r.x,
                    'y': r.y,
                    'z': r.z,
                    'secondary_structure': r.secondary_structure
                }
                for r in self.residue_records
            ])
            
            output_file = self.canonical_dir / "residues.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"✓ Saved {len(df)} residues to {output_file}")
        
        # Provenance
        if self.provenance_records:
            df = pd.DataFrame([
                {
                    'complex_id': r.complex_id,
                    'original_source_url': r.original_source_url,
                    'download_date': r.download_date,
                    'parser_version': r.parser_version,
                    'normalization_version': r.normalization_version,
                    'notes': r.notes
                }
                for r in self.provenance_records
            ])
            
            output_file = self.canonical_dir / "provenance.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"✓ Saved {len(df)} provenance records to {output_file}")
        
        # Save version info
        version_file = self.canonical_dir / "VERSION.txt"
        with open(version_file, 'w') as f:
            f.write(f"schema_version: {CanonicalSchema.SCHEMA_VERSION}\n")
            f.write(f"generation_date: {datetime.now().isoformat()}\n")
            f.write(f"chain_id_mode: {self.chain_id_mode}\n")
            f.write(f"residue_number_mode: {self.residue_number_mode}\n")
            f.write(f"peptide_length_range: {CanonicalSchema.PEPTIDE_LENGTH_MIN}-{CanonicalSchema.PEPTIDE_LENGTH_MAX_EXTENSION}\n")
            f.write(f"total_complexes: {len(self.complex_records)}\n")
        
        logger.info(f"✓ Saved version info to {version_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python canonical_builder.py <staging_dir> <canonical_dir>")
        sys.exit(1)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    staging_dir = Path(sys.argv[1])
    canonical_dir = Path(sys.argv[2])
    
    # Find mmCIF files
    cif_files = list(staging_dir.glob("**/*.cif"))
    
    if not cif_files:
        print(f"No CIF files found in {staging_dir}")
        sys.exit(1)
    
    print(f"Found {len(cif_files)} CIF files")
    
    # Build canonical dataset
    builder = CanonicalBuilder(
        staging_dir=staging_dir,
        canonical_dir=canonical_dir,
        chain_id_mode="auth",
        residue_number_mode="auth"
    )
    
    builder.build(
        source_files=cif_files[:100],  # Test with first 100
        source_db=SourceDatabase.PROPEDIA,
        batch_size=10
    )
