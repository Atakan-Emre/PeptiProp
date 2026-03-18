"""PROPEDIA v2.3 dataset downloader"""
from __future__ import annotations

import logging
import requests
from pathlib import Path
from typing import Optional
import time

from .base import BaseDownloader

logger = logging.getLogger(__name__)


class PROPEDIADownloader(BaseDownloader):
    """Download PROPEDIA v2.3 dataset"""
    
    BASE_URL = "https://propedia.russelllab.org/download"
    
    # PROPEDIA file structure
    FILES = {
        "complexes": "propedia_complexes.tar.gz",
        "receptors": "propedia_receptors.tar.gz",
        "peptides": "propedia_peptides.tar.gz",
        "interfaces": "propedia_interfaces.tar.gz",
        "metadata": "propedia_metadata.csv",
        "clusters": "propedia_clusters.csv"
    }
    
    def __init__(self, output_dir: str | Path, cache_dir: Optional[str | Path] = None):
        super().__init__(output_dir, cache_dir)
        self.metadata["dataset"] = "PROPEDIA"
        self.metadata["version"] = "2.3"
        self.metadata["source_url"] = self.BASE_URL
    
    def download(
        self,
        download_structures: bool = True,
        download_metadata: bool = True,
        download_clusters: bool = True
    ):
        """
        Download PROPEDIA dataset
        
        Args:
            download_structures: Download structure files
            download_metadata: Download metadata CSV
            download_clusters: Download cluster information
        """
        logger.info("Starting PROPEDIA v2.3 download...")
        
        # Download metadata
        if download_metadata:
            self._download_file("metadata")
        
        # Download clusters
        if download_clusters:
            self._download_file("clusters")
        
        # Download structures
        if download_structures:
            for file_type in ["complexes", "receptors", "peptides", "interfaces"]:
                self._download_file(file_type)
        
        # Save metadata
        self.save_metadata()
        
        logger.info(f"✓ PROPEDIA download complete: {self.output_dir}")
    
    def _download_file(self, file_type: str):
        """Download a specific file"""
        if file_type not in self.FILES:
            logger.error(f"Unknown file type: {file_type}")
            return
        
        filename = self.FILES[file_type]
        url = f"{self.BASE_URL}/{filename}"
        output_path = self.output_dir / filename
        
        # Check if already downloaded
        if output_path.exists():
            logger.info(f"File already exists: {filename}")
            return
        
        logger.info(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress logging
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024 * 10) == 0:  # Every 10 MB
                                logger.info(f"  Progress: {progress:.1f}%")
            
            logger.info(f"✓ Downloaded: {filename}")
            self.log_download(output_path, url)
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {filename}: {e}")
            if output_path.exists():
                output_path.unlink()
    
    def extract_archives(self):
        """Extract downloaded tar.gz archives"""
        import tarfile
        
        logger.info("Extracting archives...")
        
        for file_type in ["complexes", "receptors", "peptides", "interfaces"]:
            archive_path = self.output_dir / self.FILES[file_type]
            
            if not archive_path.exists():
                logger.warning(f"Archive not found: {archive_path}")
                continue
            
            extract_dir = self.output_dir / file_type
            extract_dir.mkdir(exist_ok=True)
            
            logger.info(f"Extracting {archive_path.name}...")
            
            try:
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(path=extract_dir)
                logger.info(f"✓ Extracted to {extract_dir}")
            except Exception as e:
                logger.error(f"Failed to extract {archive_path}: {e}")
        
        logger.info("✓ Extraction complete")


if __name__ == "__main__":
    # CLI for downloading PROPEDIA
    import argparse
    
    parser = argparse.ArgumentParser(description="Download PROPEDIA v2.3 dataset")
    parser.add_argument("--output", type=Path, default="data/raw/propedia",
                       help="Output directory")
    parser.add_argument("--no-structures", action="store_true",
                       help="Skip structure downloads")
    parser.add_argument("--extract", action="store_true",
                       help="Extract archives after download")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Download
    downloader = PROPEDIADownloader(args.output)
    downloader.download(download_structures=not args.no_structures)
    
    # Extract
    if args.extract:
        downloader.extract_archives()
