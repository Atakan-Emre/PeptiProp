"""PDB to mmCIF normalization via RCSB backfill"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
import requests
import time

logger = logging.getLogger(__name__)


class PDBToMMCIFConverter:
    """
    Convert PDB files to mmCIF format via RCSB backfill
    
    Strategy:
    1. Extract PDB ID from PDB file
    2. Download mmCIF from RCSB File Download Services
    3. Verify chain mapping consistency
    4. Cache mmCIF for reuse
    
    Rationale:
    - PROPEDIA provides PDB format
    - Canonical format is mmCIF
    - RCSB provides authoritative mmCIF versions
    - Ensures consistency with RCSB Data API
    """
    
    RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
    
    def __init__(self, cache_dir: str | Path):
        """
        Initialize converter
        
        Args:
            cache_dir: Directory to cache downloaded mmCIF files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "cached": 0,
            "downloaded": 0,
            "failed": 0
        }
    
    def convert(
        self,
        pdb_file: str | Path,
        force_download: bool = False
    ) -> Tuple[Optional[Path], str]:
        """
        Convert PDB file to mmCIF
        
        Args:
            pdb_file: Path to PDB file
            force_download: Force re-download even if cached
            
        Returns:
            (mmcif_path, status_message)
        """
        pdb_file = Path(pdb_file)
        
        if not pdb_file.exists():
            return None, f"PDB file not found: {pdb_file}"
        
        # Extract PDB ID from filename or file content
        pdb_id = self._extract_pdb_id(pdb_file)
        
        if not pdb_id:
            return None, "Could not extract PDB ID"
        
        # Check cache
        cached_mmcif = self.cache_dir / f"{pdb_id.lower()}.cif"
        
        if cached_mmcif.exists() and not force_download:
            logger.debug(f"Using cached mmCIF: {pdb_id}")
            self.stats["cached"] += 1
            return cached_mmcif, "cached"
        
        # Download from RCSB
        success, message = self._download_mmcif(pdb_id, cached_mmcif)
        
        if success:
            self.stats["downloaded"] += 1
            return cached_mmcif, "downloaded"
        else:
            self.stats["failed"] += 1
            return None, message
    
    def _extract_pdb_id(self, pdb_file: Path) -> Optional[str]:
        """
        Extract PDB ID from PDB file
        
        Strategy:
        1. Try filename (e.g., 1ABC.pdb or 1ABC_A_B.pdb for PROPEDIA)
        2. Try HEADER line in PDB file
        """
        # Try filename first
        stem = pdb_file.stem.upper()
        
        # Handle PROPEDIA format: {pdb_id}_{chain1}_{chain2}.pdb
        if '_' in stem:
            pdb_id = stem.split('_')[0]
            if len(pdb_id) == 4 and pdb_id.isalnum():
                return pdb_id
        
        # Standard format: {pdb_id}.pdb
        if len(stem) == 4 and stem.isalnum():
            return stem
        
        # Try PDB file content
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('HEADER'):
                        # HEADER line format: HEADER    CLASSIFICATION   DD-MMM-YY   IDCODE
                        parts = line.split()
                        if len(parts) >= 4:
                            pdb_id = parts[-1].strip()
                            if len(pdb_id) == 4 and pdb_id.isalnum():
                                return pdb_id.upper()
                    
                    # Stop after first 100 lines
                    if f.tell() > 10000:
                        break
        except Exception as e:
            logger.warning(f"Failed to read PDB file {pdb_file}: {e}")
        
        return None
    
    def _download_mmcif(self, pdb_id: str, output_path: Path) -> Tuple[bool, str]:
        """
        Download mmCIF from RCSB
        
        Args:
            pdb_id: PDB ID
            output_path: Output file path
            
        Returns:
            (success, message)
        """
        url = self.RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.lower())
        
        try:
            logger.info(f"Downloading mmCIF for {pdb_id} from RCSB...")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✓ Downloaded mmCIF: {pdb_id}")
            return True, "success"
            
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"mmCIF not found on RCSB: {pdb_id}")
                return False, "not_found_on_rcsb"
            else:
                logger.error(f"HTTP error downloading {pdb_id}: {e}")
                return False, f"http_error_{e.response.status_code}"
        
        except requests.RequestException as e:
            logger.error(f"Network error downloading {pdb_id}: {e}")
            return False, "network_error"
        
        except Exception as e:
            logger.error(f"Unexpected error downloading {pdb_id}: {e}")
            return False, "unexpected_error"
    
    def batch_convert(
        self,
        pdb_files: list[Path],
        rate_limit_delay: float = 0.5
    ) -> dict:
        """
        Convert multiple PDB files to mmCIF
        
        Args:
            pdb_files: List of PDB files
            rate_limit_delay: Delay between downloads (seconds)
            
        Returns:
            Conversion results
        """
        results = {
            "success": [],
            "cached": [],
            "failed": []
        }
        
        for i, pdb_file in enumerate(pdb_files):
            mmcif_path, status = self.convert(pdb_file)
            
            if mmcif_path:
                if status == "cached":
                    results["cached"].append((pdb_file, mmcif_path))
                else:
                    results["success"].append((pdb_file, mmcif_path))
                    # Rate limiting
                    if i < len(pdb_files) - 1:
                        time.sleep(rate_limit_delay)
            else:
                results["failed"].append((pdb_file, status))
            
            # Progress
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Progress: {i+1}/{len(pdb_files)} "
                    f"(success: {len(results['success'])}, "
                    f"cached: {len(results['cached'])}, "
                    f"failed: {len(results['failed'])})"
                )
        
        return results
    
    def get_statistics(self) -> dict:
        """Get conversion statistics"""
        total = sum(self.stats.values())
        
        return {
            **self.stats,
            "total": total,
            "success_rate": (self.stats["cached"] + self.stats["downloaded"]) / total if total > 0 else 0
        }
    
    def verify_chain_consistency(
        self,
        pdb_file: Path,
        mmcif_file: Path
    ) -> Tuple[bool, str]:
        """
        Verify that chain IDs are consistent between PDB and mmCIF
        
        Args:
            pdb_file: Original PDB file
            mmcif_file: Converted mmCIF file
            
        Returns:
            (is_consistent, message)
        """
        try:
            # Extract chain IDs from PDB
            pdb_chains = self._extract_pdb_chains(pdb_file)
            
            # Extract chain IDs from mmCIF
            mmcif_chains = self._extract_mmcif_chains(mmcif_file)
            
            # Compare
            if not pdb_chains or not mmcif_chains:
                return False, "Could not extract chains"
            
            pdb_set = set(pdb_chains)
            mmcif_set = set(mmcif_chains)
            
            if pdb_set == mmcif_set:
                return True, "chains_match"
            else:
                missing_in_mmcif = pdb_set - mmcif_set
                extra_in_mmcif = mmcif_set - pdb_set
                
                message = []
                if missing_in_mmcif:
                    message.append(f"missing_in_mmcif: {missing_in_mmcif}")
                if extra_in_mmcif:
                    message.append(f"extra_in_mmcif: {extra_in_mmcif}")
                
                return False, "; ".join(message)
        
        except Exception as e:
            return False, f"verification_error: {e}"
    
    def _extract_pdb_chains(self, pdb_file: Path) -> list[str]:
        """Extract chain IDs from PDB file"""
        chains = set()
        
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        # Chain ID is at position 21 (0-indexed: 21)
                        if len(line) > 21:
                            chain_id = line[21].strip()
                            if chain_id:
                                chains.add(chain_id)
        except Exception as e:
            logger.warning(f"Failed to extract PDB chains: {e}")
        
        return sorted(chains)
    
    def _extract_mmcif_chains(self, mmcif_file: Path) -> list[str]:
        """Extract chain IDs from mmCIF file"""
        chains = set()
        
        try:
            with open(mmcif_file, 'r') as f:
                in_atom_site = False
                chain_col = None
                
                for line in f:
                    line = line.strip()
                    
                    # Find _atom_site section
                    if line.startswith('_atom_site.'):
                        in_atom_site = True
                        if 'auth_asym_id' in line:
                            # This is the auth chain ID column
                            continue
                    
                    elif in_atom_site:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            parts = line.split()
                            # Auth chain ID is typically around column 6-8
                            if len(parts) > 6:
                                chain_id = parts[6].strip()
                                if chain_id and chain_id != '.':
                                    chains.add(chain_id)
                        
                        elif line.startswith('#'):
                            in_atom_site = False
        
        except Exception as e:
            logger.warning(f"Failed to extract mmCIF chains: {e}")
        
        return sorted(chains)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdb_to_mmcif.py <pdb_file_or_directory>")
        sys.exit(1)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    input_path = Path(sys.argv[1])
    cache_dir = Path("data/staging/mmcif_cache")
    
    converter = PDBToMMCIFConverter(cache_dir)
    
    if input_path.is_file():
        # Single file
        mmcif_path, status = converter.convert(input_path)
        
        if mmcif_path:
            print(f"\n✓ Converted: {input_path.name}")
            print(f"  mmCIF: {mmcif_path}")
            print(f"  Status: {status}")
            
            # Verify consistency
            is_consistent, message = converter.verify_chain_consistency(input_path, mmcif_path)
            print(f"  Chain consistency: {message}")
        else:
            print(f"\n✗ Failed: {input_path.name}")
            print(f"  Reason: {status}")
    
    elif input_path.is_dir():
        # Directory of PDB files
        pdb_files = list(input_path.glob("*.pdb")) + list(input_path.glob("*.ent"))
        
        if not pdb_files:
            print(f"No PDB files found in {input_path}")
            sys.exit(1)
        
        print(f"\nFound {len(pdb_files)} PDB files")
        print("Converting to mmCIF...\n")
        
        results = converter.batch_convert(pdb_files)
        
        # Summary
        print("\n" + "="*60)
        print("Conversion Summary")
        print("="*60)
        print(f"Success: {len(results['success'])}")
        print(f"Cached: {len(results['cached'])}")
        print(f"Failed: {len(results['failed'])}")
        
        if results['failed']:
            print("\nFailed conversions:")
            for pdb_file, reason in results['failed'][:10]:
                print(f"  • {pdb_file.name}: {reason}")
        
        stats = converter.get_statistics()
        print(f"\nSuccess rate: {stats['success_rate']:.1%}")
        print("="*60)
    
    else:
        print(f"Invalid path: {input_path}")
        sys.exit(1)
