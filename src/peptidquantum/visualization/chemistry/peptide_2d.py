"""RDKit-based 2D peptide structure rendering"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from PIL import Image
import io


class Peptide2DRenderer:
    """Render 2D chemical structures of peptides"""
    
    def __init__(self, img_size: Tuple[int, int] = (800, 400)):
        """
        Initialize renderer
        
        Args:
            img_size: Image size (width, height)
        """
        self.img_size = img_size
    
    def from_sequence(
        self,
        sequence: str,
        output_png: str | Path,
        highlight_residues: Optional[List[int]] = None,
        title: Optional[str] = None
    ):
        """
        Render peptide from amino acid sequence
        
        Args:
            sequence: Amino acid sequence (one-letter codes)
            output_png: Output PNG file
            highlight_residues: List of residue positions to highlight (1-indexed)
            title: Optional title for the image
        """
        # Convert sequence to SMILES
        mol = self._sequence_to_mol(sequence)
        
        if mol is None:
            print(f"Failed to create molecule from sequence: {sequence}")
            return
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Render
        self._render_mol(
            mol,
            output_png,
            highlight_atoms=self._get_highlight_atoms(mol, highlight_residues, sequence),
            title=title or f"Peptide: {sequence}"
        )
    
    def from_smiles(
        self,
        smiles: str,
        output_png: str | Path,
        title: Optional[str] = None
    ):
        """
        Render peptide from SMILES string
        
        Args:
            smiles: SMILES string
            output_png: Output PNG file
            title: Optional title
        """
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            print(f"Failed to create molecule from SMILES: {smiles}")
            return
        
        AllChem.Compute2DCoords(mol)
        
        self._render_mol(mol, output_png, title=title or "Peptide Structure")
    
    def from_helm(
        self,
        helm: str,
        output_png: str | Path,
        title: Optional[str] = None
    ):
        """
        Render peptide from HELM notation
        
        Args:
            helm: HELM string
            output_png: Output PNG file
            title: Optional title
        """
        # HELM parsing requires specialized library
        # For now, provide placeholder
        print("HELM parsing requires additional dependencies")
        print("Please use sequence or SMILES format")
    
    def highlight_residues(
        self,
        sequence: str,
        hot_positions: List[int],
        output_png: str | Path,
        title: Optional[str] = None
    ):
        """
        Render peptide with highlighted residues
        
        Args:
            sequence: Amino acid sequence
            hot_positions: List of residue positions to highlight (1-indexed)
            output_png: Output PNG file
            title: Optional title
        """
        self.from_sequence(
            sequence,
            output_png,
            highlight_residues=hot_positions,
            title=title or f"Peptide with Hotspots: {sequence}"
        )
    
    def _sequence_to_mol(self, sequence: str) -> Optional[Chem.Mol]:
        """
        Convert amino acid sequence to RDKit molecule
        
        Args:
            sequence: Amino acid sequence (one-letter codes)
            
        Returns:
            RDKit molecule or None
        """
        # Build peptide SMILES from sequence
        # This is a simplified approach - real peptides may have modifications
        
        aa_smiles = {
            'A': 'C[C@H](N)C(=O)',  # Alanine
            'C': 'C(C[C@H](N)C(=O))S',  # Cysteine
            'D': 'C([C@H](N)C(=O))C(=O)O',  # Aspartic acid
            'E': 'C(C[C@H](N)C(=O))C(=O)O',  # Glutamic acid
            'F': 'C1=CC=C(C=C1)C[C@H](N)C(=O)',  # Phenylalanine
            'G': 'C(N)C(=O)',  # Glycine
            'H': 'C1=C(NC=N1)C[C@H](N)C(=O)',  # Histidine
            'I': 'CC[C@H](C)[C@H](N)C(=O)',  # Isoleucine
            'K': 'C(CCN)C[C@H](N)C(=O)',  # Lysine
            'L': 'CC(C)C[C@H](N)C(=O)',  # Leucine
            'M': 'C(CSC)C[C@H](N)C(=O)',  # Methionine
            'N': 'C([C@H](N)C(=O))C(=O)N',  # Asparagine
            'P': 'C1C[C@H](NC1)C(=O)',  # Proline
            'Q': 'C(C[C@H](N)C(=O))C(=O)N',  # Glutamine
            'R': 'C(C[C@H](N)C(=O))CN=C(N)N',  # Arginine
            'S': 'C([C@H](N)C(=O))O',  # Serine
            'T': 'C[C@H]([C@H](N)C(=O))O',  # Threonine
            'V': 'CC(C)[C@H](N)C(=O)',  # Valine
            'W': 'C1=CC=C2C(=C1)C(=CN2)C[C@H](N)C(=O)',  # Tryptophan
            'Y': 'C1=CC(=CC=C1C[C@H](N)C(=O))O',  # Tyrosine
        }
        
        # Build linear peptide SMILES
        smiles_parts = []
        for aa in sequence.upper():
            if aa not in aa_smiles:
                print(f"Unknown amino acid: {aa}")
                return None
            smiles_parts.append(aa_smiles[aa])
        
        # Connect with peptide bonds (simplified)
        # In reality, this needs proper peptide bond formation
        # For visualization purposes, we'll use RDKit's built-in peptide handling
        
        try:
            # Alternative: Use RDKit's sequence to mol
            from rdkit.Chem import rdMolTransforms
            
            # Build from sequence using FASTA-like approach
            mol = Chem.MolFromSequence(sequence)
            return mol
            
        except Exception as e:
            print(f"Error creating molecule from sequence: {e}")
            
            # Fallback: create simple representation
            # Just show individual amino acids
            return None
    
    def _get_highlight_atoms(
        self,
        mol: Chem.Mol,
        highlight_residues: Optional[List[int]],
        sequence: str
    ) -> Optional[List[int]]:
        """
        Get atom indices to highlight based on residue positions
        
        Args:
            mol: RDKit molecule
            highlight_residues: Residue positions (1-indexed)
            sequence: Original sequence
            
        Returns:
            List of atom indices or None
        """
        if not highlight_residues:
            return None
        
        # This is simplified - proper implementation would need
        # residue-to-atom mapping from the molecule
        
        # For now, return None (no highlighting)
        return None
    
    def _render_mol(
        self,
        mol: Chem.Mol,
        output_png: Path,
        highlight_atoms: Optional[List[int]] = None,
        title: Optional[str] = None
    ):
        """
        Render molecule to PNG
        
        Args:
            mol: RDKit molecule
            output_png: Output PNG file
            highlight_atoms: Atom indices to highlight
            title: Image title
        """
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(self.img_size[0], self.img_size[1])
        
        # Set drawing options
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.addStereoAnnotation = True
        opts.bondLineWidth = 2
        
        # Highlight atoms if specified
        if highlight_atoms:
            highlight_atom_colors = {idx: (1, 0.7, 0.7) for idx in highlight_atoms}
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=highlight_atom_colors
            )
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        
        # Save to file
        with open(output_png, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
        print(f"2D structure saved to {output_png}")
    
    def render_with_labels(
        self,
        sequence: str,
        output_png: str | Path,
        residue_labels: Optional[Dict[int, str]] = None
    ):
        """
        Render peptide with custom residue labels
        
        Args:
            sequence: Amino acid sequence
            output_png: Output PNG file
            residue_labels: Dictionary mapping residue position to label text
        """
        mol = self._sequence_to_mol(sequence)
        
        if mol is None:
            print(f"Failed to create molecule from sequence: {sequence}")
            return
        
        AllChem.Compute2DCoords(mol)
        
        # For now, render without custom labels
        # Full implementation would add text annotations
        self._render_mol(mol, output_png, title=f"Peptide: {sequence}")
    
    def compare_peptides(
        self,
        sequences: List[str],
        output_png: str | Path,
        labels: Optional[List[str]] = None
    ):
        """
        Render multiple peptides for comparison
        
        Args:
            sequences: List of amino acid sequences
            output_png: Output PNG file
            labels: Optional labels for each peptide
        """
        mols = []
        for seq in sequences:
            mol = self._sequence_to_mol(seq)
            if mol:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)
        
        if not mols:
            print("No valid molecules to render")
            return
        
        # Create grid image
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=min(3, len(mols)),
            subImgSize=(400, 300),
            legends=labels or [f"Peptide {i+1}" for i in range(len(mols))]
        )
        
        img.save(output_png)
        print(f"Comparison image saved to {output_png}")
    
    def is_available(self) -> bool:
        """Check if RDKit is available"""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            return False
