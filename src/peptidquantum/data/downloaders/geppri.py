"""GEPPRI dataset downloader (EXTERNAL HOLDOUT - DO NOT USE FOR TRAINING)"""
from __future__ import annotations

import logging
import requests
from pathlib import Path
from typing import Optional
import pandas as pd

from .base import BaseDownloader

logger = logging.getLogger(__name__)


class GEPPRIDownloader(BaseDownloader):
    """
    Download GEPPRI dataset
    
    WARNING: This is the EXTERNAL HOLDOUT dataset.
    DO NOT use for training or validation.
    ONLY use for final benchmark evaluation.
    """
    
    # GEPPRI repository (GitHub or Zenodo)
    BASE_URL = "https://github.com/your-org/GEPPRI"  # Update with actual URL
    
    def __init__(self, output_dir: str | Path, cache_dir: Optional[str | Path] = None):
        super().__init__(output_dir, cache_dir)
        self.metadata["dataset"] = "GEPPRI"
        self.metadata["source_url"] = self.BASE_URL
        self.metadata["WARNING"] = "EXTERNAL HOLDOUT - DO NOT USE FOR TRAINING"
        
        # Create warning file
        warning_file = self.output_dir / "WARNING_EXTERNAL_HOLDOUT.txt"
        warning_file.parent.mkdir(parents=True, exist_ok=True)
        with open(warning_file, 'w') as f:
            f.write("""
╔═══════════════════════════════════════════════════════════════╗
║                         WARNING                               ║
║                                                               ║
║  This is the GEPPRI EXTERNAL HOLDOUT dataset.                ║
║                                                               ║
║  DO NOT USE FOR:                                             ║
║    - Training                                                ║
║    - Validation                                              ║
║    - Hyperparameter tuning                                   ║
║    - Model selection                                         ║
║    - Any development decisions                               ║
║                                                               ║
║  ONLY USE FOR:                                               ║
║    - Final benchmark evaluation                              ║
║    - Publication-ready metrics                               ║
║                                                               ║
║  Touching this data during development will invalidate       ║
║  your external benchmark results.                            ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    def download(
        self,
        download_structures: bool = True,
        download_metadata: bool = True
    ):
        """
        Download GEPPRI dataset
        
        Args:
            download_structures: Download structure files
            download_metadata: Download metadata
        """
        logger.warning("="*70)
        logger.warning("DOWNLOADING EXTERNAL HOLDOUT DATASET (GEPPRI)")
        logger.warning("DO NOT USE THIS DATA FOR TRAINING OR VALIDATION")
        logger.warning("="*70)
        
        # Download metadata
        if download_metadata:
            self._download_metadata()
        
        # Download structures
        if download_structures:
            self._download_structures()
        
        # Save metadata
        self.save_metadata()
        
        logger.info(f"✓ GEPPRI download complete: {self.output_dir}")
        logger.warning("Remember: This is EXTERNAL HOLDOUT - use only for final evaluation")
    
    def _download_metadata(self):
        """Download GEPPRI metadata"""
        logger.info("Downloading GEPPRI metadata...")
        
        # GEPPRI metadata files (update URLs as needed)
        metadata_files = {
            "geppri_complexes.csv": f"{self.BASE_URL}/data/complexes.csv",
            "geppri_interactions.csv": f"{self.BASE_URL}/data/interactions.csv",
            "geppri_splits.csv": f"{self.BASE_URL}/data/splits.csv"
        }
        
        for filename, url in metadata_files.items():
            output_path = self.output_dir / filename
            
            if output_path.exists():
                logger.info(f"File already exists: {filename}")
                continue
            
            try:
                logger.info(f"Downloading {filename}...")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"✓ Downloaded: {filename}")
                self.log_download(output_path, url)
                
            except requests.RequestException as e:
                logger.error(f"Failed to download {filename}: {e}")
                logger.info("You may need to download GEPPRI manually from the repository")
    
    def _download_structures(self):
        """Download GEPPRI structures"""
        logger.info("Downloading GEPPRI structures...")
        
        # Read metadata to get structure list
        metadata_file = self.output_dir / "geppri_complexes.csv"
        
        if not metadata_file.exists():
            logger.warning("Metadata file not found. Cannot download structures.")
            logger.info("Please download metadata first.")
            return
        
        try:
            df = pd.read_csv(metadata_file)
            
            structures_dir = self.output_dir / "structures"
            structures_dir.mkdir(exist_ok=True)
            
            logger.info(f"Found {len(df)} structures to download")
            
            for idx, row in df.iterrows():
                pdb_id = row.get('pdb_id', row.get('PDB_ID', None))
                
                if not pdb_id:
                    continue
                
                self._download_single_structure(pdb_id, structures_dir)
                
                if idx % 10 == 0:
                    logger.info(f"  Progress: {idx}/{len(df)}")
            
            logger.info(f"✓ Downloaded structures for GEPPRI")
            
        except Exception as e:
            logger.error(f"Failed to download structures: {e}")
    
    def _download_single_structure(self, pdb_id: str, structures_dir: Path):
        """Download a single structure"""
        pdb_id = pdb_id.lower()
        output_path = structures_dir / f"{pdb_id}.cif"
        
        if output_path.exists():
            return
        
        # Download from RCSB
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            self.log_download(output_path, url)
            
        except requests.RequestException:
            logger.debug(f"Failed to download {pdb_id}")
    
    def verify_holdout_isolation(self, canonical_dir: Path) -> bool:
        """
        Verify that GEPPRI is not contaminated with training data
        
        Args:
            canonical_dir: Path to canonical dataset directory
            
        Returns:
            True if properly isolated, False if contamination detected
        """
        logger.info("Verifying GEPPRI holdout isolation...")
        
        # Load GEPPRI IDs
        geppri_metadata = self.output_dir / "geppri_complexes.csv"
        if not geppri_metadata.exists():
            logger.warning("GEPPRI metadata not found")
            return False
        
        geppri_df = pd.read_csv(geppri_metadata)
        geppri_ids = set(geppri_df['pdb_id'].str.lower())
        
        # Load canonical dataset
        canonical_complexes = canonical_dir / "complexes.parquet"
        if not canonical_complexes.exists():
            logger.warning("Canonical dataset not found")
            return True  # Can't check yet
        
        canonical_df = pd.read_parquet(canonical_complexes)
        
        # Check for overlap in training/validation sets
        train_val_df = canonical_df[canonical_df['split_tag'].isin(['train', 'val'])]
        train_val_ids = set(train_val_df['pdb_id'].str.lower())
        
        overlap = geppri_ids & train_val_ids
        
        if overlap:
            logger.error("="*70)
            logger.error("CONTAMINATION DETECTED!")
            logger.error(f"Found {len(overlap)} GEPPRI complexes in training/validation data:")
            logger.error(f"  {list(overlap)[:10]}...")
            logger.error("="*70)
            return False
        
        logger.info("✓ GEPPRI holdout properly isolated (no contamination)")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download GEPPRI dataset (EXTERNAL HOLDOUT)",
        epilog="WARNING: This is the external holdout. Do NOT use for training!"
    )
    parser.add_argument("--output", type=Path, default="data/raw/geppri",
                       help="Output directory")
    parser.add_argument("--verify-isolation", type=Path,
                       help="Verify isolation against canonical dataset")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    downloader = GEPPRIDownloader(args.output)
    
    if args.verify_isolation:
        downloader.verify_holdout_isolation(args.verify_isolation)
    else:
        downloader.download()
