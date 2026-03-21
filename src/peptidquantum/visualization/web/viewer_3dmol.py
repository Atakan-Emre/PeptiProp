"""3Dmol.js viewer integration for interactive web visualization"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from ...data.models import Complex
from ...interaction import InteractionSet, InteractionType


class Viewer3DMol:
    """Generate 3Dmol.js viewer configurations"""
    
    THREEDMOL_CDN = "https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"

    def __init__(self):
        self.viewer_template = self._get_viewer_template()

    @staticmethod
    def infer_structure_format(structure_file: Optional[str | Path]) -> str:
        """3Dmol addModel format: pdb vs cif/mmcif."""
        if not structure_file:
            return "cif"
        ext = Path(structure_file).suffix.lower()
        if ext == ".pdb":
            return "pdb"
        return "cif"
    
    def create_viewer(
        self,
        complex_obj: Complex,
        interaction_set: Optional[InteractionSet] = None,
        output_html: Optional[str | Path] = None,
        output_json: Optional[str | Path] = None
    ) -> Dict:
        """
        Create 3Dmol.js viewer configuration
        
        Args:
            complex_obj: Complex object
            interaction_set: Optional interaction set
            output_html: Optional HTML output file
            output_json: Optional JSON state file
            
        Returns:
            Viewer configuration dictionary
        """
        sf = Path(complex_obj.structure_file) if complex_obj.structure_file else None
        structure_format = self.infer_structure_format(sf)
        viewer_state = {
            'complex_id': complex_obj.complex_id,
            'structure_file': str(complex_obj.structure_file) if complex_obj.structure_file else None,
            'structure_basename': sf.name if sf else None,
            'structure_format': structure_format,
            'chains': self._get_chain_config(complex_obj),
            'interactions': self._get_interaction_config(interaction_set) if interaction_set else [],
            'view_config': self._get_default_view_config(),
        }
        
        # Save JSON if requested
        if output_json:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(viewer_state, f, indent=2)
        
        # Generate HTML if requested
        if output_html:
            self._generate_html(viewer_state, output_html, complex_obj.structure_file)
        
        return viewer_state
    
    def _get_chain_config(self, complex_obj: Complex) -> List[Dict]:
        """Get chain visualization configuration"""
        chains = []
        
        # Protein chains
        for chain in complex_obj.protein_chains:
            chains.append({
                'chain_id': chain.chain_id,
                'type': 'protein',
                'representation': 'cartoon',
                'color': 'spectrum',
                'surface': {
                    'enabled': True,
                    'opacity': 0.3,
                    'color': 'white'
                }
            })
        
        # Peptide chains
        for chain in complex_obj.peptide_chains:
            chains.append({
                'chain_id': chain.chain_id,
                'type': 'peptide',
                'representation': 'stick',
                'color': 'marine',
                'surface': {
                    'enabled': False
                }
            })
        
        return chains
    
    def _get_interaction_config(self, interaction_set: InteractionSet) -> List[Dict]:
        """Get interaction visualization configuration"""
        interactions = []
        
        # Group by type for better visualization
        type_colors = {
            InteractionType.HBOND: '#2E86AB',
            InteractionType.SALT_BRIDGE: '#A23B72',
            InteractionType.HYDROPHOBIC: '#F18F01',
            InteractionType.PI_STACKING: '#C73E1D',
            InteractionType.CATION_PI: '#6A994E',
            InteractionType.VDW: '#CCCCCC',
        }
        
        for interaction in interaction_set.interactions:
            color = type_colors.get(interaction.interaction_type, '#888888')
            
            interactions.append({
                'protein_chain': interaction.protein_chain,
                'protein_residue': interaction.protein_residue_id,
                'protein_resname': interaction.protein_residue_name,
                'peptide_chain': interaction.peptide_chain,
                'peptide_residue': interaction.peptide_residue_id,
                'peptide_resname': interaction.peptide_residue_name,
                'type': interaction.interaction_type.value,
                'color': color,
                'distance': interaction.distance,
                'dashed': True,
                'radius': 0.15
            })
        
        return interactions
    
    def _get_default_view_config(self) -> Dict:
        """Get default view configuration"""
        return {
            'backgroundColor': 'white',
            'width': '100%',
            'height': '600px',
            'style': {
                'cartoon': {
                    'color': 'spectrum'
                },
                'stick': {
                    'radius': 0.3
                }
            },
            'zoom': {
                'enabled': True,
                'factor': 1.0
            },
            'rotate': {
                'enabled': True
            },
            'labels': {
                'enabled': True,
                'fontSize': 12
            }
        }
    
    def _generate_html(
        self,
        viewer_state: Dict,
        output_html: Path,
        structure_file: Optional[Path]
    ):
        """Generate standalone HTML viewer"""
        
        # Read structure file if available
        structure_data = ""
        if structure_file and Path(structure_file).exists():
            with open(structure_file, 'r', encoding='utf-8') as f:
                structure_data = f.read()
        
        fmt = viewer_state.get("structure_format") or self.infer_structure_format(structure_file)
        js_code = self._generate_viewer_js(
            viewer_state, structure_data, fmt, dom_element_id="viewer"
        )
        protein_chains = [
            str(chain.get("chain_id"))
            for chain in viewer_state.get("chains", [])
            if chain.get("type") == "protein"
        ]
        peptide_chains = [
            str(chain.get("chain_id"))
            for chain in viewer_state.get("chains", [])
            if chain.get("type") == "peptide"
        ]
        protein_chain_text = ", ".join(sorted(set(protein_chains))) if protein_chains else "N/A"
        peptide_chain_text = ", ".join(sorted(set(peptide_chains))) if peptide_chains else "N/A"

        # Create HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Viewer - {viewer_state['complex_id']}</title>
    <script src="{self.THREEDMOL_CDN}"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-top: 0;
        }}
        .meta {{
            margin: 10px 0 18px 0;
            padding: 10px 12px;
            border: 1px solid #e5e5e5;
            border-radius: 6px;
            background: #fafafa;
            line-height: 1.5;
        }}
        #viewer {{
            width: 100%;
            height: 600px;
            position: relative;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .controls {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .control-group {{
            margin-bottom: 10px;
        }}
        label {{
            display: inline-block;
            width: 150px;
            font-weight: bold;
        }}
        button {{
            padding: 8px 16px;
            margin: 5px;
            background-color: #2E86AB;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background-color: #1a5f7a;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
            border: 1px solid #ccc;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive 3D Viewer: {viewer_state['complex_id']}</h1>
        <div class="meta">
            <div><strong>Complex:</strong> {viewer_state['complex_id']}</div>
            <div><strong>Protein Chains:</strong> {protein_chain_text}</div>
            <div><strong>Peptide Chains:</strong> {peptide_chain_text}</div>
        </div>
        
        <div id="viewer"></div>
        
        <div class="controls">
            <h3>Controls</h3>
            <div class="control-group">
                <button onclick="resetView()">Reset View</button>
                <button onclick="toggleSurface()">Toggle Surface</button>
                <button onclick="toggleInteractions()">Toggle Interactions</button>
            </div>
            <div class="control-group">
                <label>Protein Style:</label>
                <button onclick="setProteinStyle('cartoon')">Cartoon</button>
                <button onclick="setProteinStyle('stick')">Stick</button>
                <button onclick="setProteinStyle('sphere')">Sphere</button>
            </div>
            <div class="control-group">
                <label>Background:</label>
                <button onclick="setBackground('white')">White</button>
                <button onclick="setBackground('black')">Black</button>
                <button onclick="setBackground('#f0f0f0')">Gray</button>
            </div>
        </div>
        
        <div class="legend">
            <h3>Interaction Types</h3>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #2E86AB;"></span>
                Hydrogen Bond
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #A23B72;"></span>
                Salt Bridge
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #F18F01;"></span>
                Hydrophobic
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #C73E1D;"></span>
                Pi-Stacking
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #6A994E;"></span>
                Cation-Pi
            </div>
        </div>
    </div>
    
    <script>
{js_code}
    </script>
</body>
</html>
"""
        
        # Write HTML file
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"3Dmol.js viewer saved to {output_html}")
    
    def _generate_viewer_js(
        self,
        viewer_state: Dict,
        structure_data: str,
        structure_format: str,
        dom_element_id: str = "viewer",
    ) -> str:
        """Generate JavaScript code for viewer (safe string embedding, correct PDB/CIF format)."""
        interactions_json = json.dumps(viewer_state.get("interactions", []))
        chains_json = json.dumps(viewer_state.get("chains", []))
        structure_literal = json.dumps(structure_data)
        fmt_lit = json.dumps(structure_format)
        elem_lit = json.dumps(dom_element_id)

        return f"""
// Initialize viewer
let viewer = null;
let surfaceVisible = true;
let interactionsVisible = true;

function initViewer() {{
    const element = document.getElementById({elem_lit});
    if (!element) {{
        console.error('3D viewer: missing element', {elem_lit});
        return;
    }}
    const config = {{ backgroundColor: 'white' }};
    viewer = $3Dmol.createViewer(element, config);
    
    const structureData = {structure_literal};
    viewer.addModel(structureData, {fmt_lit});
    
    applyStyles();
    addInteractions();
    
    viewer.zoomTo();
    viewer.render();
}}

function applyStyles() {{
    const chains = {chains_json};
    
    chains.forEach(chain => {{
        const sel = {{chain: chain.chain_id}};
        
        if (chain.type === 'protein') {{
            viewer.setStyle(sel, {{cartoon: {{color: 'spectrum'}}}});
            if (chain.surface.enabled) {{
                viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity: chain.surface.opacity,
                    color: chain.surface.color
                }}, sel);
            }}
        }} else if (chain.type === 'peptide') {{
            viewer.setStyle(sel, {{stick: {{radius: 0.3, color: chain.color}}}});
        }}
    }});
}}

function addInteractions() {{
    const interactions = {interactions_json};
    const model = viewer.getModel();
    if (!model) return;
    
    interactions.forEach(int => {{
        const start = {{
            chain: int.protein_chain,
            resi: int.protein_residue
        }};
        const end = {{
            chain: int.peptide_chain,
            resi: int.peptide_residue
        }};
        const sa = model.selectedAtoms(start);
        const ea = model.selectedAtoms(end);
        if (!sa || !sa.length || !ea || !ea.length) {{
            return;
        }}
        viewer.addCylinder({{
            start: sa[0],
            end: ea[0],
            radius: int.radius,
            color: int.color,
            dashed: int.dashed,
            fromCap: 1,
            toCap: 1
        }});
    }});
}}

function resetView() {{
    viewer.zoomTo();
    viewer.render();
}}

function toggleSurface() {{
    surfaceVisible = !surfaceVisible;
    const chains = {chains_json};
    const prot = chains.find(c => c.type === 'protein');
    if (surfaceVisible && prot) {{
        viewer.addSurface($3Dmol.SurfaceType.VDW, {{
            opacity: 0.3,
            color: 'white'
        }}, {{chain: prot.chain_id}});
    }} else {{
        viewer.removeAllSurfaces();
    }}
    viewer.render();
}}

function toggleInteractions() {{
    interactionsVisible = !interactionsVisible;
    if (!interactionsVisible) {{
        viewer.removeAllShapes();
    }} else {{
        addInteractions();
    }}
    viewer.render();
}}

function setProteinStyle(style) {{
    const chains = {chains_json};
    const proteinChains = chains.filter(c => c.type === 'protein');
    
    proteinChains.forEach(chain => {{
        const sel = {{chain: chain.chain_id}};
        const styleObj = {{}};
        styleObj[style] = {{color: 'spectrum'}};
        viewer.setStyle(sel, styleObj);
    }});
    
    viewer.render();
}}

function setBackground(color) {{
    viewer.setBackgroundColor(color);
    viewer.render();
}}

// Initialize on load
window.addEventListener('load', initViewer);
"""
    
    def _get_viewer_template(self) -> str:
        """Get base viewer HTML template"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <script src="{self.THREEDMOL_CDN}"></script>
</head>
<body>
    <div id="viewer" style="width: 100%; height: 600px;"></div>
    <script>
        // Viewer code will be inserted here
    </script>
</body>
</html>
"""
    
    def create_embedded_viewer(
        self,
        complex_obj: Complex,
        interaction_set: Optional[InteractionSet] = None,
        div_id: str = "viewer"
    ) -> Tuple[str, str]:
        """
        Create embeddable viewer HTML and JavaScript
        
        Args:
            complex_obj: Complex object
            interaction_set: Optional interaction set
            div_id: HTML div ID for viewer
            
        Returns:
            Tuple of (html_div, javascript_code)
        """
        sf = Path(complex_obj.structure_file) if complex_obj.structure_file else None
        structure_format = self.infer_structure_format(sf)
        viewer_state = {
            'complex_id': complex_obj.complex_id,
            'chains': self._get_chain_config(complex_obj),
            'interactions': self._get_interaction_config(interaction_set) if interaction_set else [],
            'structure_format': structure_format,
        }
        
        structure_data = ""
        if sf and sf.exists():
            with open(sf, 'r', encoding='utf-8') as f:
                structure_data = f.read()
        
        html_div = f'<div id="{div_id}" style="width: 100%; height: 600px; border: 1px solid #ddd;"></div>'
        js_code = self._generate_viewer_js(
            viewer_state, structure_data, structure_format, dom_element_id=div_id
        )
        
        return html_div, js_code
