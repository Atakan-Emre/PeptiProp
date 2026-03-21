"""PLIP 2025 interaction extraction wrapper"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET

from ..schema import StandardizedInteraction, InteractionSet, InteractionType
from ...data.models import Complex
from ...structure.parsers.tools_pdb_export import export_single_model_pdb


class PLIPWrapper:
    """Wrapper for PLIP 2025 interaction analysis"""
    
    def __init__(self, plip_path: Optional[str] = None):
        """
        Initialize PLIP wrapper
        
        Args:
            plip_path: Path to PLIP executable (if None, assumes in PATH)
        """
        self.plip_path = plip_path or "plip"

    @staticmethod
    def _obabel_executable() -> str:
        found = shutil.which("obabel")
        if found:
            return found
        for candidate in (Path("/opt/homebrew/bin/obabel"), Path("/usr/local/bin/obabel")):
            if candidate.is_file():
                return str(candidate)
        return "obabel"

    def extract_interactions(
        self,
        complex_obj: Complex,
        output_dir: Optional[Path] = None
    ) -> InteractionSet:
        """
        Extract interactions using PLIP
        
        Args:
            complex_obj: Complex object with structure file
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
            output_dir = Path(tempfile.mkdtemp(prefix="plip_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run PLIP
        try:
            plip_input = self._ensure_pdb_for_plip(structure_file, output_dir)
            interactions = self._run_plip(plip_input, output_dir)
            
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
            print(f"PLIP extraction failed: {e}")
            return InteractionSet(complex_id=complex_obj.complex_id, interactions=[])
    
    def _ensure_pdb_for_plip(self, structure_file: Path, work_dir: Path) -> Path:
        """Önce tek-model PDB (BioPython); mmCIF için Open Babel yedek."""
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        ext = structure_file.suffix.lower()

        if ext in {".cif", ".mmcif", ".pdb"}:
            out_bio = work_dir / f"{structure_file.stem}_plip_model0.pdb"
            bio = export_single_model_pdb(structure_file, out_bio)
            if bio is not None:
                return bio

        if ext in {".cif", ".mmcif"}:
            out_ob = work_dir / f"{structure_file.stem}_plip_obabel.pdb"
            result = subprocess.run(
                [self._obabel_executable(), "-icif", str(structure_file), "-opdb", "-O", str(out_ob)],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode == 0 and out_ob.exists() and out_ob.stat().st_size > 0:
                return out_ob

        return structure_file

    def _run_plip(
        self,
        structure_file: Path,
        output_dir: Path
    ) -> List[dict]:
        """
        Run PLIP command
        
        Args:
            structure_file: Path to structure file
            output_dir: Output directory
            
        Returns:
            List of interaction dictionaries
        """
        # PLIP command - output XML format
        cmd = [
            self.plip_path,
            "-f", str(structure_file),
            "-o", str(output_dir),
            "-x",  # XML output
            "-t"   # Include all interaction types
        ]
        
        # Run command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"PLIP failed: {result.stderr}")
        
        # Parse output - PLIP creates XML files
        xml_files = list(output_dir.glob("*.xml"))
        
        interactions = []
        for xml_file in xml_files:
            interactions.extend(self._parse_plip_xml(xml_file))
        
        return interactions
    
    def _parse_plip_xml(self, xml_file: Path) -> List[dict]:
        """
        Parse PLIP XML output
        
        Args:
            xml_file: Path to PLIP XML file
            
        Returns:
            List of interaction dictionaries
        """
        interactions = []
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Parse different interaction types
            for binding_site in root.findall('.//bindingsite'):
                # Hydrogen bonds
                for hbond in binding_site.findall('.//hydrogen_bond'):
                    interactions.append(self._parse_hbond(hbond))
                
                # Salt bridges
                for saltbr in binding_site.findall('.//salt_bridge'):
                    interactions.append(self._parse_saltbridge(saltbr))
                
                # Hydrophobic contacts
                for hydrophobic in binding_site.findall('.//hydrophobic_interaction'):
                    interactions.append(self._parse_hydrophobic(hydrophobic))
                
                # Pi-stacking
                for pistack in binding_site.findall('.//pi_stack'):
                    interactions.append(self._parse_pistack(pistack))
                
                # Pi-cation
                for pication in binding_site.findall('.//pi_cation'):
                    interactions.append(self._parse_pication(pication))
                
                # Halogen bonds
                for halogen in binding_site.findall('.//halogen_bond'):
                    interactions.append(self._parse_halogen(halogen))
                
                # Water bridges
                for water in binding_site.findall('.//water_bridge'):
                    interactions.append(self._parse_waterbridge(water))
                
                # Metal complexes
                for metal in binding_site.findall('.//metal_complex'):
                    interactions.append(self._parse_metal(metal))
        
        except ET.ParseError as e:
            print(f"Error parsing PLIP XML {xml_file}: {e}")
        
        return interactions
    
    def _parse_hbond(self, element: ET.Element) -> dict:
        """Parse hydrogen bond element"""
        return {
            'interaction_type': 'hbond',
            'donor_chain': element.findtext('.//donorchain', ''),
            'donor_res': element.findtext('.//donorresnr', ''),
            'donor_resname': element.findtext('.//donorrestype', ''),
            'donor_atom': element.findtext('.//donoratom', ''),
            'acceptor_chain': element.findtext('.//acceptorchain', ''),
            'acceptor_res': element.findtext('.//acceptorresnr', ''),
            'acceptor_resname': element.findtext('.//acceptorrestype', ''),
            'acceptor_atom': element.findtext('.//acceptoratom', ''),
            'distance': float(element.findtext('.//distance_ah', '0') or 0),
            'angle': float(element.findtext('.//angle', '0') or 0)
        }
    
    def _parse_saltbridge(self, element: ET.Element) -> dict:
        """Parse salt bridge element"""
        return {
            'interaction_type': 'saltbridge',
            'positive_chain': element.findtext('.//reschain_pos', ''),
            'positive_res': element.findtext('.//resnr_pos', ''),
            'positive_resname': element.findtext('.//restype_pos', ''),
            'negative_chain': element.findtext('.//reschain_neg', ''),
            'negative_res': element.findtext('.//resnr_neg', ''),
            'negative_resname': element.findtext('.//restype_neg', ''),
            'distance': float(element.findtext('.//distance', '0') or 0)
        }
    
    def _parse_hydrophobic(self, element: ET.Element) -> dict:
        """Parse hydrophobic contact element"""
        return {
            'interaction_type': 'hydrophobic',
            'chain1': element.findtext('.//reschain', ''),
            'res1': element.findtext('.//resnr', ''),
            'resname1': element.findtext('.//restype', ''),
            'atom1': element.findtext('.//resatom', ''),
            'chain2': element.findtext('.//ligchain', ''),
            'res2': element.findtext('.//lignr', ''),
            'resname2': element.findtext('.//ligtype', ''),
            'atom2': element.findtext('.//ligatom', ''),
            'distance': float(element.findtext('.//distance', '0') or 0)
        }
    
    def _parse_pistack(self, element: ET.Element) -> dict:
        """Parse pi-stacking element"""
        return {
            'interaction_type': 'pistacking',
            'chain1': element.findtext('.//reschain', ''),
            'res1': element.findtext('.//resnr', ''),
            'resname1': element.findtext('.//restype', ''),
            'chain2': element.findtext('.//ligchain', ''),
            'res2': element.findtext('.//lignr', ''),
            'resname2': element.findtext('.//ligtype', ''),
            'distance': float(element.findtext('.//distance', '0') or 0),
            'angle': float(element.findtext('.//angle', '0') or 0)
        }
    
    def _parse_pication(self, element: ET.Element) -> dict:
        """Parse pi-cation element"""
        return {
            'interaction_type': 'pication',
            'chain1': element.findtext('.//reschain', ''),
            'res1': element.findtext('.//resnr', ''),
            'resname1': element.findtext('.//restype', ''),
            'chain2': element.findtext('.//ligchain', ''),
            'res2': element.findtext('.//lignr', ''),
            'resname2': element.findtext('.//ligtype', ''),
            'distance': float(element.findtext('.//distance', '0') or 0)
        }
    
    def _parse_halogen(self, element: ET.Element) -> dict:
        """Parse halogen bond element"""
        return {
            'interaction_type': 'halogen',
            'donor_chain': element.findtext('.//donorchain', ''),
            'donor_res': element.findtext('.//donorresnr', ''),
            'donor_resname': element.findtext('.//donorrestype', ''),
            'acceptor_chain': element.findtext('.//acceptorchain', ''),
            'acceptor_res': element.findtext('.//acceptorresnr', ''),
            'acceptor_resname': element.findtext('.//acceptorrestype', ''),
            'distance': float(element.findtext('.//distance', '0') or 0)
        }
    
    def _parse_waterbridge(self, element: ET.Element) -> dict:
        """Parse water bridge element"""
        return {
            'interaction_type': 'waterbridge',
            'donor_chain': element.findtext('.//donorchain', ''),
            'donor_res': element.findtext('.//donorresnr', ''),
            'donor_resname': element.findtext('.//donorrestype', ''),
            'acceptor_chain': element.findtext('.//acceptorchain', ''),
            'acceptor_res': element.findtext('.//acceptorresnr', ''),
            'acceptor_resname': element.findtext('.//acceptorrestype', ''),
            'distance': float(element.findtext('.//distance_aw', '0') or 0)
        }
    
    def _parse_metal(self, element: ET.Element) -> dict:
        """Parse metal complex element"""
        return {
            'interaction_type': 'metal',
            'metal_chain': element.findtext('.//metalchain', ''),
            'metal_res': element.findtext('.//metalnr', ''),
            'target_chain': element.findtext('.//targetchain', ''),
            'target_res': element.findtext('.//targetnr', ''),
            'target_resname': element.findtext('.//targetrestype', ''),
            'distance': float(element.findtext('.//distance', '0') or 0)
        }
    
    def _standardize_interactions(
        self,
        plip_interactions: List[dict],
        complex_obj: Complex
    ) -> List[StandardizedInteraction]:
        """
        Convert PLIP interactions to standardized format
        
        Args:
            plip_interactions: Raw PLIP interactions
            complex_obj: Complex object for chain classification
            
        Returns:
            List of standardized interactions
        """
        standardized = []
        
        peptide_chain_ids = {c.chain_id for c in complex_obj.peptide_chains}
        protein_chain_ids = {c.chain_id for c in complex_obj.protein_chains}
        
        for interaction in plip_interactions:
            itype = interaction['interaction_type']
            
            # Extract chain and residue info based on interaction type
            if itype in ['hbond', 'halogen', 'waterbridge']:
                chain1 = interaction.get('donor_chain', '')
                res1 = interaction.get('donor_res', '')
                resname1 = interaction.get('donor_resname', '')
                atom1 = interaction.get('donor_atom')
                
                chain2 = interaction.get('acceptor_chain', '')
                res2 = interaction.get('acceptor_res', '')
                resname2 = interaction.get('acceptor_resname', '')
                atom2 = interaction.get('acceptor_atom')
                
            elif itype == 'saltbridge':
                chain1 = interaction.get('positive_chain', '')
                res1 = interaction.get('positive_res', '')
                resname1 = interaction.get('positive_resname', '')
                atom1 = None
                
                chain2 = interaction.get('negative_chain', '')
                res2 = interaction.get('negative_res', '')
                resname2 = interaction.get('negative_resname', '')
                atom2 = None
                
            elif itype == 'metal':
                chain1 = interaction.get('metal_chain', '')
                res1 = interaction.get('metal_res', '')
                resname1 = 'MET'
                atom1 = None
                
                chain2 = interaction.get('target_chain', '')
                res2 = interaction.get('target_res', '')
                resname2 = interaction.get('target_resname', '')
                atom2 = None
                
            else:  # hydrophobic, pistacking, pication
                chain1 = interaction.get('chain1', '')
                res1 = interaction.get('res1', '')
                resname1 = interaction.get('resname1', '')
                atom1 = interaction.get('atom1')
                
                chain2 = interaction.get('chain2', '')
                res2 = interaction.get('res2', '')
                resname2 = interaction.get('resname2', '')
                atom2 = interaction.get('atom2')
            
            # Determine protein and peptide sides
            if chain1 in protein_chain_ids and chain2 in peptide_chain_ids:
                protein_chain, protein_res, protein_resname, protein_atom = chain1, res1, resname1, atom1
                peptide_chain, peptide_res, peptide_resname, peptide_atom = chain2, res2, resname2, atom2
            elif chain2 in protein_chain_ids and chain1 in peptide_chain_ids:
                protein_chain, protein_res, protein_resname, protein_atom = chain2, res2, resname2, atom2
                peptide_chain, peptide_res, peptide_resname, peptide_atom = chain1, res1, resname1, atom1
            else:
                continue
            
            # Parse residue numbers
            try:
                protein_res_id = int(protein_res)
                peptide_res_id = int(peptide_res)
            except (ValueError, TypeError):
                continue
            
            # Map interaction type
            interaction_type = InteractionType.from_plip(itype)
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
                angle=interaction.get('angle'),
                source_tool='plip',
                raw_type=itype
            )
            
            standardized.append(std_interaction)
        
        return standardized
    
    def is_available(self) -> bool:
        """Check if PLIP is available"""
        try:
            result = subprocess.run(
                [self.plip_path, "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
