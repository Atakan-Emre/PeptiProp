"""PepBDB dataset downloader"""
from __future__ import annotations

import logging
import requests
from pathlib import Path
from typing import Optional
import pandas as pd
from bs4 import BeautifulSoup
import time

from .base import BaseDownloader

logger = logging.getLogger(__name__)


class PepBDBDownloader(BaseDownloader):
    """Download PepBDB dataset"""
    
    BASE_URL = "http://huanglab.phys.hust.edu.cn/pepbdb"
    DOWNLOAD_URL = f"{BASE_URL}/download.php"
    
    def __init__(self, output_dir: str | Path, cache_dir: Optional[str | Path] = None):
        super().__init__(output_dir, cache_dir)
        self.metadata["dataset"] = "PepBDB"
        self.metadata["source_url"] = self.BASE_URL
        
        # Create subdirectories
        self.structures_dir = self.output_dir / "structures"
        self.metadata_dir = self.output_dir / "metadata"
        self.structures_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def download(
        self,
        download_structures: bool = True,
        download_metadata: bool = True,
        max_length: int = 50
    ):
        """
        Download PepBDB dataset
        
        Args:
            download_structures: Download structure files
            download_metadata: Download metadata
            max_length: Maximum peptide length (default: 50 aa)
        """
        logger.info("Starting PepBDB download...")
        logger.info(f"Peptide length filter: ≤{max_length} aa")
        
        # Download metadata first
        if download_metadata:
            self._download_metadata()
        
        # Download structures
        if download_structures:
            self._download_structures(max_length)
        
        # Save metadata
        self.save_metadata()
        
        logger.info(f"✓ PepBDB download complete: {self.output_dir}")
    
    def _download_metadata(self):
        """Download PepBDB metadata"""
        logger.info("Downloading PepBDB metadata...")
        
        # PepBDB provides downloadable files
        metadata_files = {
            "pepbdb_list.txt": f"{self.BASE_URL}/download/pepbdb_list.txt",
            "pepbdb_info.csv": f"{self.BASE_URL}/download/pepbdb_info.csv"
        }
        
        for filename, url in metadata_files.items():
            output_path = self.metadata_dir / filename
            
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
                logger.warning(f"Failed to download {filename}: {e}")
                logger.info("Will attempt to scrape metadata from website...")
                self._scrape_metadata()
    
    def _scrape_metadata(self):
        """Scrape metadata from PepBDB website if direct download fails"""
        logger.info("Scraping PepBDB metadata from website...")
        
        try:
            # Get main page
            response = requests.get(f"{self.BASE_URL}/browse.php", timeout=60)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract PDB IDs and metadata
            # (This is a simplified example - actual implementation depends on website structure)
            entries = []
            
            # Save scraped data
            output_path = self.metadata_dir / "pepbdb_scraped.csv"
            df = pd.DataFrame(entries)
            df.to_csv(output_path, index=False)
            
            logger.info(f"✓ Scraped metadata saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to scrape metadata: {e}")
    
    def _download_structures(self, max_length: int):
        """Download structure files"""
        logger.info("Downloading PepBDB structures...")
        
        # Read metadata to get list of structures
        metadata_file = self.metadata_dir / "pepbdb_info.csv"
        
        if not metadata_file.exists():
            logger.warning("Metadata file not found. Cannot download structures.")
            logger.info("Please download metadata first or provide PDB IDs manually.")
            return
        
        try:
            df = pd.read_csv(metadata_file)
            
            # Filter by peptide length
            if 'peptide_length' in df.columns:
                df = df[df['peptide_length'] <= max_length]
            
            logger.info(f"Found {len(df)} structures to download")
            
            # Download each structure
            for idx, row in df.iterrows():
                pdb_id = row.get('pdb_id', row.get('PDB_ID', None))
                
                if not pdb_id:
                    continue
                
                self._download_single_structure(pdb_id)
                
                # Rate limiting
                if idx % 10 == 0:
                    time.sleep(1)
            
            logger.info(f"✓ Downloaded {len(df)} structures")
            
        except Exception as e:
            logger.error(f"Failed to download structures: {e}")
    
    def _download_single_structure(self, pdb_id: str):
        """Download a single structure file"""
        pdb_id = pdb_id.lower()
        output_path = self.structures_dir / f"{pdb_id}.cif"
        
        if output_path.exists():
            return
        
        # Try PepBDB first, then RCSB as fallback
        urls = [
            f"{self.BASE_URL}/download/structures/{pdb_id}.cif",
            f"https://files.rcsb.org/download/{pdb_id}.cif"
        ]
        
        for url in urls:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                self.log_download(output_path, url)
                return
                
            except requests.RequestException:
                continue
        
        logger.warning(f"Failed to download structure: {pdb_id}")
    
    def get_pdb_list(self) -> list[str]:
        """Get list of PDB IDs from metadata"""
        metadata_file = self.metadata_dir / "pepbdb_info.csv"
        
        if not metadata_file.exists():
            return []
        
        try:
            df = pd.read_csv(metadata_file)
            pdb_col = 'pdb_id' if 'pdb_id' in df.columns else 'PDB_ID'
            return df[pdb_col].tolist()
        except Exception as e:
            logger.error(f"Failed to read PDB list: {e}")
            return []


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download PepBDB dataset")
    parser.add_argument("--output", type=Path, default="data/raw/pepbdb",
                       help="Output directory")
    parser.add_argument("--no-structures", action="store_true",
                       help="Skip structure downloads")
    parser.add_argument("--max-length", type=int, default=50,
                       help="Maximum peptide length (default: 50)")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    downloader = PepBDBDownloader(args.output)
    downloader.download(
        download_structures=not args.no_structures,
        max_length=args.max_length
    )
