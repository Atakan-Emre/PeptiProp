"""PyMOL renderer for publication-quality figures"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import subprocess
import tempfile

from ...data.models import Complex
from ...interaction import InteractionSet, InteractionType


class PyMOLRenderer:
    """Generate publication-quality figures using PyMOL"""
    
    def __init__(self, pymol_path: Optional[str] = None):
        """
        Initialize PyMOL renderer
        
        Args:
            pymol_path: Path to PyMOL executable (if None, assumes in PATH)
        """
        self.pymol_path = pymol_path or "pymol"
    
    def render_overview(
        self,
        complex_cif: str | Path,
        peptide_chain: str,
        output_png: str | Path,
        width: int = 1200,
        height: int = 900,
        ray_trace: bool = True
    ):
        """
        Render complex overview
        
        Args:
            complex_cif: Path to complex CIF file
            peptide_chain: Peptide chain ID
            output_png: Output PNG file
            width: Image width
            height: Image height
            ray_trace: Use ray tracing for high quality
        """
        script = self._generate_overview_script(
            complex_cif, peptide_chain, output_png, width, height, ray_trace
        )
        self._run_pymol_script(script)
    
    def render_pocket(
        self,
        complex_cif: str | Path,
        peptide_chain: str,
        pocket_residues: List[Tuple[str, int]],
        output_png: str | Path,
        interactions: Optional[InteractionSet] = None,
        width: int = 1200,
        height: int = 900
    ):
        """
        Render binding pocket zoom
        
        Args:
            complex_cif: Path to complex CIF file
            peptide_chain: Peptide chain ID
            pocket_residues: List of (chain_id, residue_id) tuples
            output_png: Output PNG file
            interactions: Optional interactions to show
            width: Image width
            height: Image height
        """
        script = self._generate_pocket_script(
            complex_cif, peptide_chain, pocket_residues, 
            output_png, interactions, width, height
        )
        self._run_pymol_script(script)
    
    def render_interactions(
        self,
        complex_cif: str | Path,
        interaction_set: InteractionSet,
        protein_chain: str,
        peptide_chain: str,
        output_png: str | Path,
        width: int = 1200,
        height: int = 900
    ):
        """
        Render interactions overlay
        
        Args:
            complex_cif: Path to complex CIF file
            interaction_set: Set of interactions
            protein_chain: Protein chain ID
            peptide_chain: Peptide chain ID
            output_png: Output PNG file
            width: Image width
            height: Image height
        """
        script = self._generate_interaction_script(
            complex_cif, interaction_set, protein_chain, 
            peptide_chain, output_png, width, height
        )
        self._run_pymol_script(script)
    
    def render_importance(
        self,
        complex_cif: str | Path,
        residue_scores: Dict[Tuple[str, int], float],
        peptide_chain: str,
        output_png: str | Path,
        width: int = 1200,
        height: int = 900
    ):
        """
        Render residue importance coloring
        
        Args:
            complex_cif: Path to complex CIF file
            residue_scores: Dictionary mapping (chain, residue_id) to importance score
            peptide_chain: Peptide chain ID
            output_png: Output PNG file
            width: Image width
            height: Image height
        """
        script = self._generate_importance_script(
            complex_cif, residue_scores, peptide_chain, 
            output_png, width, height
        )
        self._run_pymol_script(script)
    
    def _generate_overview_script(
        self,
        complex_cif: Path,
        peptide_chain: str,
        output_png: Path,
        width: int,
        height: int,
        ray_trace: bool
    ) -> str:
        """Generate PyMOL script for overview"""
        return f"""
# Load structure
load {complex_cif}, complex

# Hide everything
hide everything

# Protein: surface + cartoon
select protein, not chain {peptide_chain}
show surface, protein
show cartoon, protein
color gray80, protein
set transparency, 0.3, protein

# Peptide: sticks
select peptide, chain {peptide_chain}
show sticks, peptide
color marine, peptide
set stick_radius, 0.3, peptide

# Styling
set ray_shadows, 0
set ray_trace_mode, 1
set antialias, 2
set orthoscopic, on

# View
zoom complex
orient

# Render
viewport {width}, {height}
{'ray' if ray_trace else 'draw'}
png {output_png}, dpi=300

quit
"""
    
    def _generate_pocket_script(
        self,
        complex_cif: Path,
        peptide_chain: str,
        pocket_residues: List[Tuple[str, int]],
        output_png: Path,
        interactions: Optional[InteractionSet],
        width: int,
        height: int
    ) -> str:
        """Generate PyMOL script for pocket view"""
        # Build pocket selection
        pocket_sel = " or ".join([
            f"(chain {chain} and resi {res})" 
            for chain, res in pocket_residues
        ])
        
        script = f"""
# Load structure
load {complex_cif}, complex

# Hide everything
hide everything

# Pocket residues: sticks + surface
select pocket, {pocket_sel}
show sticks, pocket
show surface, pocket
color wheat, pocket
set transparency, 0.5, pocket
set stick_radius, 0.25, pocket

# Peptide: sticks
select peptide, chain {peptide_chain}
show sticks, peptide
color marine, peptide
set stick_radius, 0.3, peptide

# Labels for key residues
label pocket and name CA, "%s%s" % (resn, resi)
set label_size, 14
set label_color, black
"""
        
        # Add interaction lines if provided
        if interactions:
            script += "\n# Interaction lines\n"
            
            for interaction in interactions.interactions:
                if interaction.peptide_chain == peptide_chain:
                    color = self._get_interaction_color(interaction.interaction_type)
                    
                    script += f"""
distance int_{interaction.protein_residue_id}_{interaction.peptide_residue_id}, \
(chain {interaction.protein_chain} and resi {interaction.protein_residue_id} and name CA), \
(chain {interaction.peptide_chain} and resi {interaction.peptide_residue_id} and name CA)
color {color}, int_{interaction.protein_residue_id}_{interaction.peptide_residue_id}
hide labels, int_{interaction.protein_residue_id}_{interaction.peptide_residue_id}
"""
        
        script += f"""
# Styling
set dash_gap, 0.3
set dash_radius, 0.15
set ray_shadows, 0
set antialias, 2
set orthoscopic, on

# View
zoom peptide, 8
orient peptide

# Render
viewport {width}, {height}
ray
png {output_png}, dpi=300

quit
"""
        return script
    
    def _generate_interaction_script(
        self,
        complex_cif: Path,
        interaction_set: InteractionSet,
        protein_chain: str,
        peptide_chain: str,
        output_png: Path,
        width: int,
        height: int
    ) -> str:
        """Generate PyMOL script for interaction overlay"""
        script = f"""
# Load structure
load {complex_cif}, complex

# Hide everything
hide everything

# Protein: cartoon
select protein, chain {protein_chain}
show cartoon, protein
color gray70, protein

# Peptide: sticks
select peptide, chain {peptide_chain}
show sticks, peptide
color marine, peptide
set stick_radius, 0.3, peptide

# Interacting residues: sticks
"""
        
        # Collect interacting protein residues
        prot_residues = set()
        for interaction in interaction_set.interactions:
            if interaction.protein_chain == protein_chain:
                prot_residues.add(interaction.protein_residue_id)
        
        if prot_residues:
            prot_sel = " or ".join([f"resi {r}" for r in prot_residues])
            script += f"""
select prot_interact, chain {protein_chain} and ({prot_sel})
show sticks, prot_interact
color wheat, prot_interact
set stick_radius, 0.25, prot_interact
"""
        
        # Add interactions by type
        for itype in [InteractionType.HBOND, InteractionType.SALT_BRIDGE, 
                     InteractionType.HYDROPHOBIC, InteractionType.PI_STACKING]:
            typed_interactions = interaction_set.filter_by_type(itype)
            
            if typed_interactions:
                color = self._get_interaction_color(itype)
                script += f"\n# {itype.value}\n"
                
                for interaction in typed_interactions:
                    if (interaction.protein_chain == protein_chain and 
                        interaction.peptide_chain == peptide_chain):
                        script += f"""
distance {itype.value}_{interaction.protein_residue_id}_{interaction.peptide_residue_id}, \
(chain {protein_chain} and resi {interaction.protein_residue_id} and name CA), \
(chain {peptide_chain} and resi {interaction.peptide_residue_id} and name CA)
color {color}, {itype.value}_{interaction.protein_residue_id}_{interaction.peptide_residue_id}
hide labels, {itype.value}_{interaction.protein_residue_id}_{interaction.peptide_residue_id}
"""
        
        script += f"""
# Styling
set dash_gap, 0.3
set dash_radius, 0.15
set ray_shadows, 0
set antialias, 2

# View
zoom peptide, 10
orient peptide

# Render
viewport {width}, {height}
ray
png {output_png}, dpi=300

quit
"""
        return script
    
    def _generate_importance_script(
        self,
        complex_cif: Path,
        residue_scores: Dict[Tuple[str, int], float],
        peptide_chain: str,
        output_png: Path,
        width: int,
        height: int
    ) -> str:
        """Generate PyMOL script for importance coloring"""
        script = f"""
# Load structure
load {complex_cif}, complex

# Hide everything
hide everything

# Protein: cartoon
show cartoon, complex
color gray70, complex

# Peptide: sticks
select peptide, chain {peptide_chain}
show sticks, peptide
set stick_radius, 0.3, peptide

# Color by importance (blue to red gradient)
"""
        
        # Normalize scores to 0-1
        if residue_scores:
            max_score = max(residue_scores.values())
            min_score = min(residue_scores.values())
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for (chain, res_id), score in residue_scores.items():
                # Normalize score
                norm_score = (score - min_score) / score_range
                
                # Map to color (blue=low, red=high)
                if norm_score < 0.33:
                    color = "blue"
                elif norm_score < 0.67:
                    color = "yellow"
                else:
                    color = "red"
                
                script += f"""
select res_{chain}_{res_id}, chain {chain} and resi {res_id}
show sticks, res_{chain}_{res_id}
color {color}, res_{chain}_{res_id}
set stick_radius, 0.4, res_{chain}_{res_id}
"""
        
        script += f"""
# Styling
set ray_shadows, 0
set antialias, 2
set orthoscopic, on

# View
zoom complex
orient

# Render
viewport {width}, {height}
ray
png {output_png}, dpi=300

quit
"""
        return script
    
    def _run_pymol_script(self, script: str):
        """Execute PyMOL script"""
        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pml', delete=False) as f:
            f.write(script)
            script_file = f.name
        
        try:
            # Run PyMOL in command-line mode
            cmd = [self.pymol_path, "-c", "-q", script_file]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"PyMOL error: {result.stderr}")
            
        finally:
            # Clean up temp file
            Path(script_file).unlink(missing_ok=True)
    
    @staticmethod
    def _get_interaction_color(itype: InteractionType) -> str:
        """Map interaction type to PyMOL color"""
        color_map = {
            InteractionType.HBOND: "blue",
            InteractionType.SALT_BRIDGE: "magenta",
            InteractionType.HYDROPHOBIC: "orange",
            InteractionType.PI_STACKING: "red",
            InteractionType.CATION_PI: "green",
            InteractionType.VDW: "gray",
            InteractionType.HALOGEN: "purple",
            InteractionType.METAL: "gray50",
        }
        return color_map.get(itype, "gray")
    
    def is_available(self) -> bool:
        """Check if PyMOL is available"""
        try:
            result = subprocess.run(
                [self.pymol_path, "-h"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
