"""Structure parser with mmCIF-first support and PDB fallback."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

# Biopython for standard parsing
try:
    from Bio.PDB import MMCIFParser as BioPythonMMCIFParser
    from Bio.PDB import PDBParser as BioPythonPDBParser
    from Bio.PDB.Structure import Structure
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logging.warning("Biopython not available")

# Gemmi for robust mmCIF handling and neighbor search
try:
    import gemmi
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False
    logging.warning("Gemmi not available")

logger = logging.getLogger(__name__)


@dataclass
class ChainInfo:
    """Chain information from mmCIF"""
    chain_id_auth: str
    chain_id_label: str
    entity_id: str
    entity_type: str  # polymer, non-polymer, water
    sequence: str
    residues: List[ResidueInfo]


@dataclass
class ResidueInfo:
    """Residue information from mmCIF"""
    residue_number_auth: int
    residue_number_label: int
    resname: str
    chain_id_auth: str
    chain_id_label: str
    atoms: List[AtomInfo]
    centroid: Optional[Tuple[float, float, float]] = None


@dataclass
class AtomInfo:
    """Atom information from mmCIF"""
    atom_id: int
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float


class MMCIFStructureParser:
    """
    Dual-layer structure parser using Biopython and Gemmi.
    
    Strategy:
    - Biopython: Primary parser for auth/label ID handling
    - Gemmi: Backup parser and neighbor search support
    """
    
    def __init__(
        self,
        use_auth_ids: bool = True,
        use_gemmi_fallback: bool = True
    ):
        """
        Initialize parser
        
        Args:
            use_auth_ids: Use author IDs (default: True)
            use_gemmi_fallback: Use Gemmi as fallback (default: True)
        """
        self.use_auth_ids = use_auth_ids
        self.use_gemmi_fallback = use_gemmi_fallback
        
        if not BIOPYTHON_AVAILABLE and not GEMMI_AVAILABLE:
            raise ImportError("Neither Biopython nor Gemmi available. Install at least one.")
    
    def parse(self, structure_file: str | Path) -> Dict:
        """
        Parse structure file.
        
        Args:
            structure_file: Path to mmCIF or PDB file
            
        Returns:
            Parsed structure data
        """
        structure_file = Path(structure_file)
        
        if not structure_file.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_file}")
        
        logger.info(f"Parsing structure: {structure_file.name}")

        suffix = structure_file.suffix.lower()

        if suffix in {".pdb", ".ent"}:
            if not BIOPYTHON_AVAILABLE:
                raise RuntimeError("Biopython is required to parse PDB files")
            return self._parse_pdb_with_biopython(structure_file)
        
        # Try Biopython first
        if BIOPYTHON_AVAILABLE:
            try:
                return self._parse_with_biopython(structure_file)
            except Exception as e:
                logger.warning(f"Biopython parsing failed: {e}")
                if self.use_gemmi_fallback and GEMMI_AVAILABLE:
                    logger.info("Falling back to Gemmi parser...")
                    return self._parse_with_gemmi(structure_file)
                else:
                    raise
        
        # Fallback to Gemmi
        elif GEMMI_AVAILABLE:
            return self._parse_with_gemmi(structure_file)
        
        else:
            raise RuntimeError("No parser available")

    def _parse_pdb_with_biopython(self, pdb_file: Path) -> Dict:
        """Parse PROPEDIA pair PDB files with Biopython."""
        parser = BioPythonPDBParser(QUIET=True)
        structure = parser.get_structure(pdb_file.stem, str(pdb_file))

        chains = []
        for model in structure:
            for chain in model:
                chain_info = self._extract_chain_biopython(chain)
                if chain_info:
                    chains.append(chain_info)

        pdb_id = pdb_file.stem.split("_")[0].upper()
        return {
            "pdb_id": pdb_id,
            "chains": chains,
            "parser": "biopython_pdb",
            "auth_mode": self.use_auth_ids,
        }
    
    def _parse_with_biopython(self, cif_file: Path) -> Dict:
        """Parse with Biopython"""
        parser = BioPythonMMCIFParser(
            auth_chains=self.use_auth_ids,
            auth_residues=self.use_auth_ids
        )
        
        structure = parser.get_structure("structure", str(cif_file))
        
        # Extract chains
        chains = []
        
        for model in structure:
            for chain in model:
                chain_info = self._extract_chain_biopython(chain)
                if chain_info:
                    chains.append(chain_info)
        
        return {
            "pdb_id": structure.id,
            "chains": chains,
            "parser": "biopython",
            "auth_mode": self.use_auth_ids
        }
    
    def _extract_chain_biopython(self, chain) -> Optional[ChainInfo]:
        """Extract chain information from Biopython chain"""
        try:
            chain_id_auth = chain.id
            chain_id_label = chain.id  # Biopython uses auth by default
            
            # Extract residues
            residues = []
            sequence = []
            
            for residue in chain:
                # Skip hetero atoms and water
                if residue.id[0] != ' ':
                    continue
                
                residue_info = self._extract_residue_biopython(residue, chain_id_auth, chain_id_label)
                if residue_info:
                    residues.append(residue_info)
                    sequence.append(residue_info.resname)
            
            if not residues:
                return None
            
            return ChainInfo(
                chain_id_auth=chain_id_auth,
                chain_id_label=chain_id_label,
                entity_id="",  # Not available in Biopython
                entity_type="polymer",  # Assume polymer for now
                sequence="".join(sequence),
                residues=residues
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract chain: {e}")
            return None
    
    def _extract_residue_biopython(
        self,
        residue,
        chain_id_auth: str,
        chain_id_label: str
    ) -> Optional[ResidueInfo]:
        """Extract residue information from Biopython residue"""
        try:
            resname = residue.resname
            residue_number_auth = residue.id[1]
            residue_number_label = residue.id[1]  # Biopython uses auth by default
            
            # Extract atoms
            atoms = []
            coords = []
            
            for atom in residue:
                atom_info = AtomInfo(
                    atom_id=atom.serial_number,
                    atom_name=atom.name,
                    element=atom.element,
                    x=float(atom.coord[0]),
                    y=float(atom.coord[1]),
                    z=float(atom.coord[2]),
                    occupancy=float(atom.occupancy),
                    b_factor=float(atom.bfactor)
                )
                atoms.append(atom_info)
                coords.append(atom.coord)
            
            # Calculate centroid
            if coords:
                centroid = tuple(np.mean(coords, axis=0))
            else:
                centroid = None
            
            return ResidueInfo(
                residue_number_auth=residue_number_auth,
                residue_number_label=residue_number_label,
                resname=resname,
                chain_id_auth=chain_id_auth,
                chain_id_label=chain_id_label,
                atoms=atoms,
                centroid=centroid
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract residue: {e}")
            return None
    
    def _parse_with_gemmi(self, cif_file: Path) -> Dict:
        """Parse with Gemmi"""
        structure = gemmi.read_structure(str(cif_file))
        
        # Extract chains
        chains = []
        
        for model in structure:
            for chain in model:
                chain_info = self._extract_chain_gemmi(chain)
                if chain_info:
                    chains.append(chain_info)
        
        return {
            "pdb_id": structure.name,
            "chains": chains,
            "parser": "gemmi",
            "auth_mode": self.use_auth_ids
        }
    
    def _extract_chain_gemmi(self, chain) -> Optional[ChainInfo]:
        """Extract chain information from Gemmi chain"""
        try:
            chain_id_auth = chain.name
            
            # Extract residues
            residues = []
            sequence = []
            
            for residue in chain:
                # Skip water and hetero atoms
                if residue.is_water() or not residue.is_polymer():
                    continue
                
                residue_info = self._extract_residue_gemmi(residue, chain_id_auth)
                if residue_info:
                    residues.append(residue_info)
                    sequence.append(residue_info.resname)
            
            if not residues:
                return None
            
            return ChainInfo(
                chain_id_auth=chain_id_auth,
                chain_id_label=chain_id_auth,  # Gemmi uses auth by default
                entity_id="",
                entity_type="polymer",
                sequence="".join(sequence),
                residues=residues
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract chain: {e}")
            return None
    
    def _extract_residue_gemmi(
        self,
        residue,
        chain_id_auth: str
    ) -> Optional[ResidueInfo]:
        """Extract residue information from Gemmi residue"""
        try:
            resname = residue.name
            residue_number_auth = residue.seqid.num
            
            # Extract atoms
            atoms = []
            coords = []
            
            for atom in residue:
                atom_info = AtomInfo(
                    atom_id=atom.serial,
                    atom_name=atom.name,
                    element=atom.element.name,
                    x=float(atom.pos.x),
                    y=float(atom.pos.y),
                    z=float(atom.pos.z),
                    occupancy=float(atom.occ),
                    b_factor=float(atom.b_iso)
                )
                atoms.append(atom_info)
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            
            # Calculate centroid
            if coords:
                centroid = tuple(np.mean(coords, axis=0))
            else:
                centroid = None
            
            return ResidueInfo(
                residue_number_auth=residue_number_auth,
                residue_number_label=residue_number_auth,  # Gemmi uses auth
                resname=resname,
                chain_id_auth=chain_id_auth,
                chain_id_label=chain_id_auth,
                atoms=atoms,
                centroid=centroid
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract residue: {e}")
            return None
    
    def find_neighbors(
        self,
        cif_file: str | Path,
        chain_id: str,
        residue_num: int,
        radius: float = 5.0
    ) -> List[Tuple[str, int, float]]:
        """
        Find neighboring residues within radius (Gemmi-based)
        
        Args:
            cif_file: Path to mmCIF file
            chain_id: Chain ID
            residue_num: Residue number
            radius: Search radius in Angstroms
            
        Returns:
            List of (chain_id, residue_num, distance) tuples
        """
        if not GEMMI_AVAILABLE:
            logger.warning("Gemmi not available for neighbor search")
            return []
        
        structure = gemmi.read_structure(str(cif_file))
        
        # Find target residue
        target_residue = None
        for model in structure:
            for chain in model:
                if chain.name == chain_id:
                    for residue in chain:
                        if residue.seqid.num == residue_num:
                            target_residue = residue
                            break
        
        if not target_residue:
            return []
        
        # Get target centroid
        coords = [[atom.pos.x, atom.pos.y, atom.pos.z] for atom in target_residue]
        if not coords:
            return []
        
        target_centroid = np.mean(coords, axis=0)
        
        # Find neighbors
        neighbors = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue == target_residue:
                        continue
                    
                    # Calculate distance
                    res_coords = [[atom.pos.x, atom.pos.y, atom.pos.z] for atom in residue]
                    if not res_coords:
                        continue
                    
                    res_centroid = np.mean(res_coords, axis=0)
                    distance = np.linalg.norm(target_centroid - res_centroid)
                    
                    if distance <= radius:
                        neighbors.append((chain.name, residue.seqid.num, float(distance)))
        
        return sorted(neighbors, key=lambda x: x[2])


if __name__ == "__main__":
    # Test parser
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mmcif_parser.py <cif_file>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    parser = MMCIFStructureParser(use_auth_ids=True)
    result = parser.parse(sys.argv[1])
    
    print(f"\nParsed with: {result['parser']}")
    print(f"PDB ID: {result['pdb_id']}")
    print(f"Chains: {len(result['chains'])}")
    
    for chain in result['chains']:
        print(f"\nChain {chain.chain_id_auth}:")
        print(f"  Residues: {len(chain.residues)}")
        print(f"  Sequence length: {len(chain.sequence)}")
