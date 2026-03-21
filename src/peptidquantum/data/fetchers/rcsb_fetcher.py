"""RCSB PDB data fetcher"""
from __future__ import annotations

import requests
from pathlib import Path
from typing import Optional, List, Dict
import json


class RCSBFetcher:
    """Fetch structures and metadata from RCSB PDB"""
    
    BASE_URL = "https://data.rcsb.org"
    FILES_URL = "https://files.rcsb.org/download"
    SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    def __init__(self, cache_dir: Optional[str | Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/rcsb")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_structure(self, pdb_id: str, format: str = "cif") -> Optional[Path]:
        """
        Download structure file
        
        Args:
            pdb_id: PDB ID (e.g., "1ABC")
            format: "cif" (mmCIF) or "pdb"
            
        Returns:
            Path to downloaded file
        """
        pdb_id = pdb_id.upper()
        
        ext = "cif" if format == "cif" else "pdb"
        cache_file = self.cache_dir / f"{pdb_id}.{ext}"

        if cache_file.exists() and cache_file.stat().st_size > 0:
            return cache_file
        if cache_file.exists():
            cache_file.unlink(missing_ok=True)

        url = f"{self.FILES_URL}/{pdb_id}.{ext}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text = response.text
            if not text or not text.strip():
                print(f"Error downloading {pdb_id}: empty response body")
                return None

            cache_file.write_text(text, encoding="utf-8")
            return cache_file
            
        except requests.RequestException as e:
            print(f"Error downloading {pdb_id}: {e}")
            return None
    
    def fetch_metadata(self, pdb_id: str) -> Optional[Dict]:
        """
        Fetch structure metadata
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            Metadata dictionary
        """
        pdb_id = pdb_id.upper()
        
        # Check cache
        cache_file = self.cache_dir / f"{pdb_id}_metadata.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        
        # Fetch from API
        url = f"{self.BASE_URL}/rest/v1/core/entry/{pdb_id}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            metadata = response.json()
            cache_file.write_text(json.dumps(metadata, indent=2))
            
            return metadata
            
        except requests.RequestException as e:
            print(f"Error fetching metadata for {pdb_id}: {e}")
            return None
    
    def search_peptide_complexes(
        self,
        peptide_sequence: Optional[str] = None,
        min_peptide_length: int = 4,
        max_peptide_length: int = 50,
        max_results: int = 100
    ) -> List[str]:
        """
        Search for peptide-protein complexes
        
        Args:
            peptide_sequence: Specific peptide sequence to search
            min_peptide_length: Minimum peptide chain length
            max_peptide_length: Maximum peptide chain length
            max_results: Maximum number of results
            
        Returns:
            List of PDB IDs
        """
        # Build query
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_entity_polymer_type",
                            "operator": "exact_match",
                            "value": "Protein"
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity.rcsb_polymer_entity_container_identifiers.polymer_entity_type",
                            "operator": "exact_match",
                            "value": "Polypeptide(L)"
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {
                "results_content_type": ["experimental"],
                "return_all_hits": False,
                "pager": {
                    "start": 0,
                    "rows": max_results
                }
            }
        }
        
        # Add sequence search if provided
        if peptide_sequence:
            query["query"]["nodes"].append({
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": 1,
                    "identity_cutoff": 0.9,
                    "sequence_type": "protein",
                    "value": peptide_sequence
                }
            })
        
        try:
            response = requests.post(
                self.SEARCH_URL,
                json=query,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            results = response.json()
            pdb_ids = [hit["identifier"] for hit in results.get("result_set", [])]
            
            return pdb_ids
            
        except requests.RequestException as e:
            print(f"Error searching RCSB: {e}")
            return []
    
    def get_chain_info(self, pdb_id: str) -> Optional[Dict]:
        """
        Get chain information for a structure
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            Dictionary with chain information
        """
        pdb_id = pdb_id.upper()
        
        url = f"{self.BASE_URL}/rest/v1/core/polymer_entity/{pdb_id}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            print(f"Error fetching chain info for {pdb_id}: {e}")
            return None
    
    def is_experimental(self, pdb_id: str) -> bool:
        """Check if structure is experimental"""
        metadata = self.fetch_metadata(pdb_id)
        if not metadata:
            return False
        
        exp_method = metadata.get("exptl", [{}])[0].get("method", "")
        return exp_method.upper() in ["X-RAY DIFFRACTION", "SOLUTION NMR", "ELECTRON MICROSCOPY"]
    
    def get_resolution(self, pdb_id: str) -> Optional[float]:
        """Get structure resolution"""
        metadata = self.fetch_metadata(pdb_id)
        if not metadata:
            return None
        
        refine = metadata.get("refine", [{}])[0]
        return refine.get("ls_d_res_high")
