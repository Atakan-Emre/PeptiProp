"""mmCIF structure parser using Biopython"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Residue import Residue as BioResidue
import numpy as np

from ...data.models import Complex, Chain, Residue, StructureSource, StructureOrigin


class StructureParser:
    """Parse mmCIF files into Complex objects"""
    
    def __init__(self):
        self.parser = MMCIFParser(QUIET=True)
    
    def parse_file(
        self,
        structure_file: str | Path,
        complex_id: Optional[str] = None,
        structure_source: StructureSource = StructureSource.EXPERIMENTAL,
        structure_origin: StructureOrigin = StructureOrigin.RCSB
    ) -> Optional[Complex]:
        """
        Parse mmCIF file into Complex object
        
        Args:
            structure_file: Path to mmCIF file
            complex_id: Complex identifier
            structure_source: Source type
            structure_origin: Origin database
            
        Returns:
            Complex object
        """
        structure_file = Path(structure_file)
        
        if not structure_file.exists():
            print(f"File not found: {structure_file}")
            return None
        
        if complex_id is None:
            complex_id = structure_file.stem
        
        try:
            # Parse structure
            structure = self.parser.get_structure(complex_id, str(structure_file))
            
            # Create complex
            complex_obj = Complex(
                complex_id=complex_id,
                structure_source=structure_source,
                structure_origin=structure_origin,
                structure_file=str(structure_file)
            )
            
            # Extract chains
            model = structure[0]  # Use first model
            
            for bio_chain in model:
                chain = self._parse_chain(bio_chain)
                
                # Classify as protein or peptide based on length
                if len(chain.sequence) <= 50:  # Heuristic: peptides are typically short
                    chain.chain_type = "peptide"
                    complex_obj.peptide_chains.append(chain)
                else:
                    chain.chain_type = "protein"
                    complex_obj.protein_chains.append(chain)
            
            return complex_obj
            
        except Exception as e:
            print(f"Error parsing {structure_file}: {e}")
            return None
    
    def _parse_chain(self, bio_chain: BioChain) -> Chain:
        """Parse Bio.PDB Chain into our Chain object"""
        chain_id = bio_chain.id
        residues = []
        sequence_chars = []
        
        for bio_residue in bio_chain:
            # Skip hetero residues (water, ligands, etc.)
            if bio_residue.id[0] != ' ':
                continue
            
            residue = self._parse_residue(bio_residue, chain_id)
            residues.append(residue)
            
            # Build sequence
            resname = bio_residue.resname.strip()
            aa_code = self._three_to_one(resname)
            sequence_chars.append(aa_code)
        
        sequence = ''.join(sequence_chars)
        
        return Chain(
            chain_id=chain_id,
            chain_type="unknown",  # Will be set by caller
            sequence=sequence,
            residues=residues
        )
    
    def _parse_residue(self, bio_residue: BioResidue, chain_id: str) -> Residue:
        """Parse Bio.PDB Residue into our Residue object"""
        hetflag, resseq, icode = bio_residue.id
        resname = bio_residue.resname.strip()
        
        # Get CA coordinates
        x, y, z = 0.0, 0.0, 0.0
        b_factor = None
        
        if 'CA' in bio_residue:
            ca_atom = bio_residue['CA']
            x, y, z = ca_atom.coord
            b_factor = ca_atom.bfactor
        
        return Residue(
            chain_id=chain_id,
            residue_number=resseq,
            residue_name=resname,
            insertion_code=icode.strip(),
            x=float(x),
            y=float(y),
            z=float(z),
            b_factor=float(b_factor) if b_factor else None
        )
    
    @staticmethod
    def _three_to_one(three_letter: str) -> str:
        """Convert three-letter amino acid code to one-letter"""
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return aa_map.get(three_letter.upper(), 'X')
    
    def extract_pocket(
        self,
        complex_obj: Complex,
        peptide_chain_id: str,
        radius: float = 8.0
    ) -> Complex:
        """
        Extract pocket residues around peptide
        
        Args:
            complex_obj: Input complex
            peptide_chain_id: Peptide chain ID
            radius: Distance cutoff in Angstroms
            
        Returns:
            New complex with only pocket residues
        """
        # Get peptide chain
        peptide_chain = complex_obj.get_peptide_chain(peptide_chain_id)
        if not peptide_chain:
            return complex_obj
        
        # Get peptide coordinates
        peptide_coords = np.array([
            [r.x, r.y, r.z] for r in peptide_chain.residues
        ])
        
        # Find pocket residues
        pocket_complex = Complex(
            complex_id=f"{complex_obj.complex_id}_pocket",
            structure_source=complex_obj.structure_source,
            structure_origin=complex_obj.structure_origin,
            peptide_chains=[peptide_chain]  # Keep full peptide
        )
        
        for protein_chain in complex_obj.protein_chains:
            pocket_residues = []
            
            for residue in protein_chain.residues:
                res_coord = np.array([residue.x, residue.y, residue.z])
                
                # Calculate minimum distance to any peptide residue
                distances = np.linalg.norm(peptide_coords - res_coord, axis=1)
                min_dist = distances.min()
                
                if min_dist <= radius:
                    pocket_residues.append(residue)
            
            if pocket_residues:
                # Create pocket chain
                pocket_chain = Chain(
                    chain_id=protein_chain.chain_id,
                    chain_type="protein",
                    sequence=''.join([self._three_to_one(r.residue_name) 
                                     for r in pocket_residues]),
                    residues=pocket_residues
                )
                pocket_complex.protein_chains.append(pocket_chain)
        
        return pocket_complex
    
    def save_structure(
        self,
        complex_obj: Complex,
        output_file: str | Path,
        format: str = "cif"
    ):
        """
        Save complex to file
        
        Args:
            complex_obj: Complex to save
            output_file: Output file path
            format: "cif" or "pdb"
        """
        # This would require reconstructing Bio.PDB structure
        # For now, just copy the original file if it exists
        if complex_obj.structure_file:
            import shutil
            shutil.copy(complex_obj.structure_file, output_file)
