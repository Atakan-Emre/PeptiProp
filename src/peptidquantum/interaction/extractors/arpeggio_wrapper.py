"""Arpeggio interaction extraction wrapper"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
import json

from ..schema import StandardizedInteraction, InteractionSet, InteractionType
from ...data.models import Complex
from ...structure.parsers.tools_pdb_export import export_single_model_pdb


class ArpeggioWrapper:
    """Wrapper for Arpeggio interaction analysis"""
    
    def __init__(self, arpeggio_path: Optional[str] = None):
        """
        Initialize Arpeggio wrapper
        
        Args:
            arpeggio_path: Path to Arpeggio executable (if None, assumes in PATH)
        """
        self.arpeggio_path = arpeggio_path or "arpeggio"
    
    def extract_interactions(
        self,
        complex_obj: Complex,
        selection: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> InteractionSet:
        """
        Extract interactions using Arpeggio
        
        Args:
            complex_obj: Complex object with structure file
            selection: Chain selection (e.g., "/A/")
            output_dir: Directory for output files
            
        Returns:
            InteractionSet with standardized interactions
        """
        if not complex_obj.structure_file:
            raise ValueError("Complex must have structure_file set")
        
        structure_file = Path(complex_obj.structure_file)
        if not structure_file.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_file}")
        
        # Create temp directory if needed
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="arpeggio_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build selection string for peptide chains
        if selection is None:
            peptide_chains = [c.chain_id for c in complex_obj.peptide_chains]
            if peptide_chains:
                selection = f"/{peptide_chains[0]}//"
        
        # Run Arpeggio (yalnızca düzgün PDB; mmCIF doğrudan verilmez)
        try:
            arpeggio_input = self._structure_path_for_arpeggio(structure_file, output_dir)
            interactions = self._run_arpeggio(
                arpeggio_input,
                selection,
                output_dir
            )
            
            # Standardize interactions
            standardized = self._standardize_interactions(
                interactions,
                complex_obj
            )
            
            return InteractionSet(
                complex_id=complex_obj.complex_id,
                interactions=standardized
            )
            
        except Exception as e:
            print(f"Arpeggio extraction failed: {e}")
            return InteractionSet(complex_id=complex_obj.complex_id, interactions=[])

    def _structure_path_for_arpeggio(self, structure_file: Path, work_dir: Path) -> Path:
        """mmCIF / çok modelli PDB → tek model PDB (BioPython PDBIO)."""
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        ext = structure_file.suffix.lower()
        if ext in {".cif", ".mmcif", ".pdb"}:
            out = work_dir / f"{structure_file.stem}_arpeggio_model0.pdb"
            done = export_single_model_pdb(structure_file, out)
            if done is not None:
                return done
        return structure_file

    def _run_arpeggio(
        self,
        structure_file: Path,
        selection: str,
        output_dir: Path
    ) -> List[dict]:
        """
        Run Arpeggio command
        
        Args:
            structure_file: Path to structure file
            selection: Chain selection string
            output_dir: Output directory
            
        Returns:
            List of interaction dictionaries
        """
        # Arpeggio: çıktılar PDB ile aynı dizine yazılır (.contacts vb.).
        # `-o` kullanma: argparse `-o`yu `-op` (--output-postfix) ile karıştırıp pdb yolunu bozuyor.
        cmd = [
            self.arpeggio_path,
            str(structure_file),
            "-s",
            selection,
        ]
        
        # Run command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Arpeggio failed: {result.stderr}")
        
        # Parse output
        contacts_file = output_dir / "contacts.json"
        if not contacts_file.exists():
            # Try alternative output format
            contacts_file = output_dir / f"{structure_file.stem}.contacts"
        
        if contacts_file.exists():
            return self._parse_arpeggio_output(contacts_file)
        else:
            print(f"Warning: No contacts file found in {output_dir}")
            return []
    
    def _parse_arpeggio_output(self, contacts_file: Path) -> List[dict]:
        """
        Parse Arpeggio contacts file
        
        Args:
            contacts_file: Path to contacts file
            
        Returns:
            List of interaction dictionaries
        """
        interactions = []
        
        # Arpeggio can output JSON or text format
        if contacts_file.suffix == '.json':
            with open(contacts_file) as f:
                data = json.load(f)
                # Parse JSON format (structure depends on Arpeggio version)
                if isinstance(data, list):
                    interactions = data
                elif isinstance(data, dict) and 'contacts' in data:
                    interactions = data['contacts']
        else:
            # Parse text format
            with open(contacts_file) as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        interaction = {
                            'atom1_chain': parts[0],
                            'atom1_res': parts[1],
                            'atom1_resname': parts[2],
                            'atom1_name': parts[3],
                            'atom2_chain': parts[4],
                            'atom2_res': parts[5],
                            'atom2_resname': parts[6],
                            'atom2_name': parts[7],
                            'interaction_type': parts[8] if len(parts) > 8 else 'INTER',
                            'distance': float(parts[9]) if len(parts) > 9 else None
                        }
                        interactions.append(interaction)
        
        return interactions
    
    def _standardize_interactions(
        self,
        arpeggio_interactions: List[dict],
        complex_obj: Complex
    ) -> List[StandardizedInteraction]:
        """
        Convert Arpeggio interactions to standardized format
        
        Args:
            arpeggio_interactions: Raw Arpeggio interactions
            complex_obj: Complex object for chain classification
            
        Returns:
            List of standardized interactions
        """
        standardized = []
        
        # Get peptide chain IDs
        peptide_chain_ids = {c.chain_id for c in complex_obj.peptide_chains}
        protein_chain_ids = {c.chain_id for c in complex_obj.protein_chains}
        
        for interaction in arpeggio_interactions:
            # Determine which is protein and which is peptide
            chain1 = interaction.get('atom1_chain', '')
            chain2 = interaction.get('atom2_chain', '')
            
            # Skip if both are same type
            if (chain1 in peptide_chain_ids and chain2 in peptide_chain_ids):
                continue
            if (chain1 in protein_chain_ids and chain2 in protein_chain_ids):
                continue
            
            # Determine protein and peptide sides
            if chain1 in protein_chain_ids and chain2 in peptide_chain_ids:
                protein_chain = chain1
                protein_res = interaction.get('atom1_res', '')
                protein_resname = interaction.get('atom1_resname', '')
                protein_atom = interaction.get('atom1_name')
                
                peptide_chain = chain2
                peptide_res = interaction.get('atom2_res', '')
                peptide_resname = interaction.get('atom2_resname', '')
                peptide_atom = interaction.get('atom2_name')
            elif chain2 in protein_chain_ids and chain1 in peptide_chain_ids:
                protein_chain = chain2
                protein_res = interaction.get('atom2_res', '')
                protein_resname = interaction.get('atom2_resname', '')
                protein_atom = interaction.get('atom2_name')
                
                peptide_chain = chain1
                peptide_res = interaction.get('atom1_res', '')
                peptide_resname = interaction.get('atom1_resname', '')
                peptide_atom = interaction.get('atom1_name')
            else:
                continue
            
            # Parse residue number
            try:
                protein_res_id = int(protein_res)
                peptide_res_id = int(peptide_res)
            except (ValueError, TypeError):
                continue
            
            # Map interaction type
            raw_type = interaction.get('interaction_type', 'INTER')
            interaction_type = InteractionType.from_arpeggio(raw_type)
            
            if interaction_type is None:
                continue
            
            # Create standardized interaction
            std_interaction = StandardizedInteraction(
                protein_chain=protein_chain,
                protein_residue_id=protein_res_id,
                protein_residue_name=protein_resname,
                protein_atom=protein_atom,
                peptide_chain=peptide_chain,
                peptide_residue_id=peptide_res_id,
                peptide_residue_name=peptide_resname,
                peptide_atom=peptide_atom,
                interaction_type=interaction_type,
                distance=interaction.get('distance'),
                source_tool='arpeggio',
                raw_type=raw_type
            )
            
            standardized.append(std_interaction)
        
        return standardized
    
    def is_available(self) -> bool:
        """Check if Arpeggio is available"""
        try:
            result = subprocess.run(
                [self.arpeggio_path, "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
