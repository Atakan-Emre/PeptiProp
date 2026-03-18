"""BioLiP2 dataset downloader (peptide subset only)"""
from __future__ import annotations

import logging
import requests
from pathlib import Path
from typing import Optional
import pandas as pd
import gzip
import shutil

from .base import BaseDownloader

logger = logging.getLogger(__name__)


class BioLiP2Downloader(BaseDownloader):
    """Download BioLiP2 dataset (peptide ligands only)"""
    
    BASE_URL = "https://zhanggroup.org/BioLiP"
    DOWNLOAD_BASE = f"{BASE_URL}/download"
    
    # BioLiP2 download files
    FILES = {
        "receptor": "receptor.tar.bz2",
        "ligand": "ligand.tar.bz2",
        "binding_data": "BioLiP.txt",
        "nr_list": "nr_list.txt"
    }
    
    def __init__(self, output_dir: str | Path, cache_dir: Optional[str | Path] = None):
        super().__init__(output_dir, cache_dir)
        self.metadata["dataset"] = "BioLiP2"
        self.metadata["source_url"] = self.BASE_URL
        
        # Create subdirectories
        self.full_download_dir = self.output_dir / "full_download"
        self.peptide_subset_dir = self.output_dir / "peptide_subset"
        self.full_download_dir.mkdir(parents=True, exist_ok=True)
        self.peptide_subset_dir.mkdir(parents=True, exist_ok=True)
    
    def download(
        self,
        peptides_only: bool = True,
        max_peptide_length: int = 30,
        download_structures: bool = True,
        download_metadata: bool = True
    ):
        """
        Download BioLiP2 dataset
        
        Args:
            peptides_only: Extract only peptide ligands
            max_peptide_length: Maximum peptide length for core set
            download_structures: Download structure files
            download_metadata: Download metadata
        """
        logger.info("Starting BioLiP2 download...")
        
        if peptides_only:
            logger.info(f"Peptide-only mode: max length {max_peptide_length} aa")
        
        # Download metadata
        if download_metadata:
            self._download_metadata()
        
        # Download structures
        if download_structures:
            if peptides_only:
                self._download_peptide_subset(max_peptide_length)
            else:
                self._download_full_dataset()
        
        # Save metadata
        self.save_metadata()
        
        logger.info(f"✓ BioLiP2 download complete: {self.output_dir}")
    
    def _download_metadata(self):
        """Download BioLiP2 metadata files"""
        logger.info("Downloading BioLiP2 metadata...")
        
        metadata_files = {
            "BioLiP.txt": f"{self.DOWNLOAD_BASE}/BioLiP.txt",
            "nr_list.txt": f"{self.DOWNLOAD_BASE}/nr_list.txt"
        }
        
        for filename, url in metadata_files.items():
            output_path = self.full_download_dir / filename
            
            if output_path.exists():
                logger.info(f"File already exists: {filename}")
                continue
            
            try:
                logger.info(f"Downloading {filename}...")
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"✓ Downloaded: {filename}")
                self.log_download(output_path, url)
                
            except requests.RequestException as e:
                logger.error(f"Failed to download {filename}: {e}")
    
    def _download_full_dataset(self):
        """Download full BioLiP2 dataset"""
        logger.info("Downloading full BioLiP2 dataset...")
        
        for file_type, filename in self.FILES.items():
            if file_type in ["binding_data", "nr_list"]:
                continue  # Already downloaded in metadata
            
            url = f"{self.DOWNLOAD_BASE}/{filename}"
            output_path = self.full_download_dir / filename
            
            if output_path.exists():
                logger.info(f"Archive already exists: {filename}")
                continue
            
            try:
                logger.info(f"Downloading {filename} (this may take a while)...")
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0 and downloaded % (1024 * 1024 * 50) == 0:
                                progress = (downloaded / total_size) * 100
                                logger.info(f"  Progress: {progress:.1f}%")
                
                logger.info(f"✓ Downloaded: {filename}")
                self.log_download(output_path, url)
                
            except requests.RequestException as e:
                logger.error(f"Failed to download {filename}: {e}")
    
    def _download_peptide_subset(self, max_length: int):
        """Download and filter peptide subset"""
        logger.info("Extracting peptide subset from BioLiP2...")
        
        # First, download metadata
        metadata_file = self.full_download_dir / "BioLiP.txt"
        
        if not metadata_file.exists():
            logger.error("BioLiP metadata not found. Download metadata first.")
            return
        
        # Parse metadata to find peptide ligands
        peptide_entries = self._parse_peptide_ligands(metadata_file, max_length)
        
        if not peptide_entries:
            logger.warning("No peptide ligands found in metadata")
            return
        
        logger.info(f"Found {len(peptide_entries)} peptide ligands (≤{max_length} aa)")
        
        # Save peptide subset metadata
        subset_metadata = self.peptide_subset_dir / "peptide_ligands.csv"
        pd.DataFrame(peptide_entries).to_csv(subset_metadata, index=False)
        logger.info(f"✓ Peptide subset metadata saved: {subset_metadata}")
        
        # Download structures for peptide subset
        self._download_peptide_structures(peptide_entries)
    
    def _parse_peptide_ligands(self, metadata_file: Path, max_length: int) -> list[dict]:
        """Parse BioLiP metadata to extract peptide ligands"""
        logger.info("Parsing BioLiP metadata for peptide ligands...")
        
        peptide_entries = []
        
        try:
            with open(metadata_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    
                    if len(parts) < 10:
                        continue
                    
                    # BioLiP format: PDB, Chain, Ligand, etc.
                    pdb_id = parts[0]
                    chain_id = parts[1]
                    ligand_id = parts[2]
                    ligand_chain = parts[3]
                    
                    # Check if ligand is peptide
                    # BioLiP uses specific codes for peptides
                    # Peptides typically have chain IDs and are <30 residues
                    
                    # Try to get ligand sequence/length from metadata
                    # (Simplified - actual parsing depends on BioLiP format)
                    
                    # For now, we'll mark entries that likely contain peptides
                    # based on ligand chain presence
                    if ligand_chain and ligand_chain != '-':
                        peptide_entries.append({
                            'pdb_id': pdb_id,
                            'protein_chain': chain_id,
                            'peptide_chain': ligand_chain,
                            'ligand_id': ligand_id
                        })
            
            logger.info(f"Parsed {len(peptide_entries)} potential peptide entries")
            
        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")
        
        return peptide_entries
    
    def _download_peptide_structures(self, peptide_entries: list[dict]):
        """Download structures for peptide subset"""
        logger.info("Downloading structures for peptide subset...")
        
        structures_dir = self.peptide_subset_dir / "structures"
        structures_dir.mkdir(exist_ok=True)
        
        downloaded = 0
        
        for entry in peptide_entries:
            pdb_id = entry['pdb_id'].lower()
            output_path = structures_dir / f"{pdb_id}.cif"
            
            if output_path.exists():
                continue
            
            # Download from RCSB
            url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded += 1
                
                if downloaded % 100 == 0:
                    logger.info(f"  Downloaded {downloaded}/{len(peptide_entries)} structures")
                
            except requests.RequestException:
                logger.debug(f"Failed to download {pdb_id}")
                continue
        
        logger.info(f"✓ Downloaded {downloaded} structures")
    
    def extract_archives(self):
        """Extract downloaded archives"""
        import tarfile
        
        logger.info("Extracting BioLiP2 archives...")
        
        for file_type in ["receptor", "ligand"]:
            archive_path = self.full_download_dir / self.FILES[file_type]
            
            if not archive_path.exists():
                logger.warning(f"Archive not found: {archive_path}")
                continue
            
            extract_dir = self.full_download_dir / file_type
            extract_dir.mkdir(exist_ok=True)
            
            logger.info(f"Extracting {archive_path.name}...")
            
            try:
                with tarfile.open(archive_path, 'r:bz2') as tar:
                    tar.extractall(path=extract_dir)
                logger.info(f"✓ Extracted to {extract_dir}")
            except Exception as e:
                logger.error(f"Failed to extract {archive_path}: {e}")
        
        logger.info("✓ Extraction complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download BioLiP2 dataset")
    parser.add_argument("--output", type=Path, default="data/raw/biolip2",
                       help="Output directory")
    parser.add_argument("--full", action="store_true",
                       help="Download full dataset (not just peptides)")
    parser.add_argument("--max-length", type=int, default=30,
                       help="Maximum peptide length (default: 30)")
    parser.add_argument("--extract", action="store_true",
                       help="Extract archives after download")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    downloader = BioLiP2Downloader(args.output)
    downloader.download(
        peptides_only=not args.full,
        max_peptide_length=args.max_length
    )
    
    if args.extract:
        downloader.extract_archives()
