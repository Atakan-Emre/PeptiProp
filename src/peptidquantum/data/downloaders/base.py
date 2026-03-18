"""Base downloader class for dataset acquisition"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseDownloader:
    """Base class for dataset downloaders"""
    
    def __init__(self, output_dir: str | Path, cache_dir: Optional[str | Path] = None):
        """
        Initialize downloader
        
        Args:
            output_dir: Output directory for downloaded data
            cache_dir: Optional cache directory
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / "cache"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata
        self.metadata = {
            "downloader": self.__class__.__name__,
            "download_date": datetime.now().isoformat(),
            "files": []
        }
    
    def download(self, **kwargs):
        """Download dataset (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement download()")
    
    def verify_download(self, file_path: Path, expected_hash: Optional[str] = None) -> bool:
        """
        Verify downloaded file
        
        Args:
            file_path: Path to file
            expected_hash: Expected MD5 hash (optional)
            
        Returns:
            True if valid
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        if expected_hash:
            actual_hash = self._compute_hash(file_path)
            if actual_hash != expected_hash:
                logger.error(f"Hash mismatch for {file_path}")
                logger.error(f"  Expected: {expected_hash}")
                logger.error(f"  Actual: {actual_hash}")
                return False
        
        return True
    
    def _compute_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
    
    def save_metadata(self):
        """Save download metadata"""
        metadata_file = self.output_dir / "download_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")
    
    def log_download(self, file_path: Path, source_url: str):
        """Log downloaded file"""
        self.metadata["files"].append({
            "path": str(file_path),
            "source_url": source_url,
            "size_bytes": file_path.stat().st_size,
            "hash": self._compute_hash(file_path),
            "timestamp": datetime.now().isoformat()
        })
