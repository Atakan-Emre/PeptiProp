"""HTML report generator for protein-peptide complexes"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, List
import base64
from datetime import datetime

from ...data.models import Complex
from ...interaction import InteractionSet
from .viewer_3dmol import Viewer3DMol


class ReportBuilder:
    """Build comprehensive HTML reports"""
    
    def __init__(self):
        self.viewer_builder = Viewer3DMol()
    
    def build(
        self,
        complex_obj: Complex,
        interaction_set: InteractionSet,
        assets_dir: str | Path,
        output_html: str | Path,
        include_viewer: bool = True,
        interaction_provenance: Optional[Dict] = None,
    ):
        """
        Build complete HTML report
        
        Args:
            complex_obj: Complex object
            interaction_set: Interaction set
            assets_dir: Directory containing figure assets
            output_html: Output HTML file
            include_viewer: Include 3Dmol.js viewer
        """
        assets_dir = Path(assets_dir)
        
        # Collect assets
        assets = self._collect_assets(assets_dir)
        
        # Generate sections
        sections = []
        
        # 1. Header
        sections.append(self._generate_header(complex_obj))
        
        # 2. Overview section
        sections.append(
            self._generate_overview_section(complex_obj, interaction_set, interaction_provenance)
        )
        
        # 3. Interactive viewer
        if include_viewer:
            sections.append(self._generate_viewer_section(complex_obj, interaction_set))
        
        # 4. Structure figures
        sections.append(self._generate_structure_section(assets))
        
        # 5. Contact map
        sections.append(self._generate_contact_map_section(assets))
        
        # 6. Peptide chemistry
        sections.append(self._generate_chemistry_section(assets))
        
        # 7. Interaction analysis
        sections.append(self._generate_interaction_section(interaction_set, assets))
        
        # 8. Data tables
        sections.append(self._generate_data_section(assets_dir))
        
        # 9. Provenance
        sections.append(self._generate_provenance_section(complex_obj))
        
        # Build complete HTML
        html = self._build_html(complex_obj.complex_id, sections)
        
        # Write file
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"HTML report saved to {output_html}")
    
    def _collect_assets(self, assets_dir: Path) -> Dict[str, Path]:
        """Collect all asset files"""
        assets = {}
        
        # Expected assets
        asset_names = [
            'complex_overview.png',
            'pocket_zoom.png',
            'interaction_overlay.png',
            'contact_map.png',
            'contact_map_by_type.png',
            'interaction_summary.png',
            'peptide_2d.png'
        ]
        
        for name in asset_names:
            asset_path = assets_dir / name
            if asset_path.exists():
                assets[name] = asset_path
        
        return assets
    
    def _generate_header(self, complex_obj: Complex) -> str:
        """Generate report header"""
        return f"""
        <div class="header">
            <h1>Protein-Peptide Interaction Report</h1>
            <h2>{complex_obj.complex_id}</h2>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
    
    def _generate_overview_section(
        self,
        complex_obj: Complex,
        interaction_set: InteractionSet,
        interaction_provenance: Optional[Dict] = None,
    ) -> str:
        """Generate overview section"""
        # Count chains
        n_protein = len(complex_obj.protein_chains)
        n_peptide = len(complex_obj.peptide_chains)
        protein_chain_ids = ", ".join(chain.chain_id for chain in complex_obj.protein_chains) if complex_obj.protein_chains else "N/A"
        peptide_chain_ids = ", ".join(chain.chain_id for chain in complex_obj.peptide_chains) if complex_obj.peptide_chains else "N/A"
        n_interactions = len(interaction_set.interactions)
        
        # Get interaction types
        interaction_types = interaction_set.get_interaction_types()
        type_counts = interaction_set.count_by_type()
        
        type_list = "<br>".join([
            f"&bull; {itype.value.replace('_', ' ').title()}: {count}"
            for itype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        ])

        prov_html = ""
        if interaction_provenance:
            mode = interaction_provenance.get("extraction_mode", "unknown")
            tbf = interaction_provenance.get("tool_based_interaction_fraction")
            fbf = interaction_provenance.get("fallback_interaction_fraction")
            fr = interaction_provenance.get("per_interaction_source_fraction") or {}
            frac_lines = "<br>".join(
                f"&bull; {k}: {100.0 * float(v):.1f}%"
                for k, v in sorted(fr.items(), key=lambda x: -x[1])
            )
            prov_html = f"""
                <div class="info-card">
                    <h3>Interaction source</h3>
                    <p><strong>Extraction mode:</strong> {mode}</p>
                    <p><strong>Tool-based (PLIP/Arpeggio) fraction:</strong> {100.0 * float(tbf or 0):.1f}%</p>
                    <p><strong>Geometric fallback fraction:</strong> {100.0 * float(fbf or 0):.1f}%</p>
                    <p><strong>Per-record source mix:</strong></p>
                    <p style="font-size: 0.9em; line-height: 1.6;">{frac_lines or "N/A"}</p>
                </div>
            """
        
        return f"""
        <div class="section">
            <h2>Overview</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h3>Complex Information</h3>
                    <p><strong>Complex ID:</strong> {complex_obj.complex_id}</p>
                    <p><strong>Protein Chains:</strong> {n_protein}</p>
                    <p><strong>Protein Chain IDs:</strong> {protein_chain_ids}</p>
                    <p><strong>Peptide Chains:</strong> {n_peptide}</p>
                    <p><strong>Peptide Chain IDs:</strong> {peptide_chain_ids}</p>
                    <p><strong>Source:</strong> {complex_obj.structure_source.value}</p>
                    <p><strong>Origin:</strong> {complex_obj.structure_origin.value}</p>
                </div>
                <div class="info-card">
                    <h3>Interaction Summary</h3>
                    <p><strong>Total Interactions:</strong> {n_interactions}</p>
                    <p><strong>Unique Types:</strong> {len(interaction_types)}</p>
                    <p><strong>Breakdown:</strong></p>
                    <p style="font-size: 0.9em; line-height: 1.6;">{type_list}</p>
                </div>
                {prov_html}
            </div>
        </div>
        """
    
    def _generate_viewer_section(self, complex_obj: Complex, interaction_set: InteractionSet) -> str:
        """Generate interactive viewer section"""
        # Create embedded viewer
        html_div, js_code = self.viewer_builder.create_embedded_viewer(
            complex_obj,
            interaction_set,
            div_id="main_viewer"
        )
        
        return f"""
        <div class="section">
            <h2>Interactive 3D Viewer</h2>
            <p>Use mouse to rotate, zoom, and explore the structure. Click on residues for details.</p>
            {html_div}
            <script>
            {js_code}
            </script>
        </div>
        """
    
    def _generate_structure_section(self, assets: Dict[str, Path]) -> str:
        """Generate structure figures section"""
        figures = []
        
        if 'complex_overview.png' in assets:
            figures.append(self._create_figure_html(
                assets['complex_overview.png'],
                "Complex Overview",
                "Overall structure showing protein (surface) and peptide (sticks)"
            ))
        
        if 'pocket_zoom.png' in assets:
            figures.append(self._create_figure_html(
                assets['pocket_zoom.png'],
                "Binding Pocket",
                "Zoomed view of the binding pocket with key interactions"
            ))
        
        if 'interaction_overlay.png' in assets:
            figures.append(self._create_figure_html(
                assets['interaction_overlay.png'],
                "Interaction Overlay",
                "Detailed view of molecular interactions"
            ))
        
        if not figures:
            return ""
        
        return f"""
        <div class="section">
            <h2>Structure Visualization</h2>
            <div class="figure-grid">
                {"".join(figures)}
            </div>
        </div>
        """
    
    def _generate_contact_map_section(self, assets: Dict[str, Path]) -> str:
        """Generate contact map section"""
        figures = []
        
        if 'contact_map.png' in assets:
            figures.append(self._create_figure_html(
                assets['contact_map.png'],
                "Contact Map",
                "Residue-residue contact matrix"
            ))
        
        if 'contact_map_by_type.png' in assets:
            figures.append(self._create_figure_html(
                assets['contact_map_by_type.png'],
                "Contact Map by Type",
                "Interaction type-specific contact matrices"
            ))
        
        if not figures:
            return ""
        
        return f"""
        <div class="section">
            <h2>Contact Analysis</h2>
            <div class="figure-grid">
                {"".join(figures)}
            </div>
        </div>
        """
    
    def _generate_chemistry_section(self, assets: Dict[str, Path]) -> str:
        """Generate peptide chemistry section"""
        if 'peptide_2d.png' not in assets:
            return ""
        
        figure = self._create_figure_html(
            assets['peptide_2d.png'],
            "Peptide 2D Structure",
            "Chemical structure of the peptide"
        )
        
        return f"""
        <div class="section">
            <h2>Peptide Chemistry</h2>
            {figure}
        </div>
        """
    
    def _generate_interaction_section(self, interaction_set: InteractionSet, assets: Dict[str, Path]) -> str:
        """Generate interaction analysis section"""
        figures = []
        
        if 'interaction_summary.png' in assets:
            figures.append(self._create_figure_html(
                assets['interaction_summary.png'],
                "Interaction Statistics",
                "Statistical analysis of interactions"
            ))
        
        # Interaction fingerprint
        fingerprint_html = self._create_fingerprint_card(interaction_set)
        
        return f"""
        <div class="section">
            <h2>Interaction Analysis</h2>
            {fingerprint_html}
            <div class="figure-grid">
                {"".join(figures)}
            </div>
        </div>
        """
    
    def _generate_data_section(self, assets_dir: Path) -> str:
        """Generate data tables section"""
        contacts_file = assets_dir / "contacts.tsv"
        fingerprint_file = assets_dir / "interaction_fingerprint.json"
        
        sections = []
        
        # Contacts table preview
        if contacts_file.exists():
            with open(contacts_file, 'r') as f:
                lines = f.readlines()[:11]  # Header + 10 rows
            
            table_html = self._create_table_from_tsv(lines)
            sections.append(f"""
            <div class="data-card">
                <h3>Contacts Table (Preview)</h3>
                <p>Showing first 10 interactions. <a href="{contacts_file.name}">Download full table</a></p>
                {table_html}
            </div>
            """)
        
        # Fingerprint JSON
        if fingerprint_file.exists():
            with open(fingerprint_file, 'r') as f:
                fingerprint = json.load(f)
            
            stats = fingerprint.get('statistics', {})
            stats_html = "<br>".join([
                f"<strong>{k.replace('_', ' ').title()}:</strong> {v:.2f}" if isinstance(v, float) else f"<strong>{k.replace('_', ' ').title()}:</strong> {v}"
                for k, v in stats.items()
            ])
            
            sections.append(f"""
            <div class="data-card">
                <h3>Interaction Fingerprint</h3>
                <p>{stats_html}</p>
                <p><a href="{fingerprint_file.name}">Download JSON</a></p>
            </div>
            """)
        
        if not sections:
            return ""
        
        return f"""
        <div class="section">
            <h2>Data Tables</h2>
            <div class="data-grid">
                {"".join(sections)}
            </div>
        </div>
        """
    
    def _generate_provenance_section(self, complex_obj: Complex) -> str:
        """Generate provenance and metadata section"""
        confidence_html = ""
        if complex_obj.confidence is not None:
            confidence_html = f"<p><strong>Model Confidence:</strong> {complex_obj.confidence:.3f}</p>"
        
        if complex_obj.interface_confidence is not None:
            confidence_html += f"<p><strong>Interface Confidence:</strong> {complex_obj.interface_confidence:.3f}</p>"
        
        return f"""
        <div class="section">
            <h2>Provenance & Metadata</h2>
            <div class="info-card">
                <p><strong>Structure Source:</strong> {complex_obj.structure_source.value}</p>
                <p><strong>Structure Origin:</strong> {complex_obj.structure_origin.value}</p>
                <p><strong>Assembly:</strong> {complex_obj.assembly_used}</p>
                {confidence_html}
                <p><strong>Structure File:</strong> {complex_obj.structure_file or 'N/A'}</p>
            </div>
        </div>
        """
    
    def _create_figure_html(self, image_path: Path, title: str, caption: str) -> str:
        """Create figure HTML with embedded image"""
        # Embed image as base64
        with open(image_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        
        return f"""
        <div class="figure">
            <img src="data:image/png;base64,{img_data}" alt="{title}">
            <p class="figure-caption"><strong>{title}:</strong> {caption}</p>
        </div>
        """
    
    def _create_fingerprint_card(self, interaction_set: InteractionSet) -> str:
        """Create interaction fingerprint summary card"""
        type_counts = interaction_set.count_by_type()
        
        rows = []
        for itype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(interaction_set.interactions)) * 100
            rows.append(f"""
            <tr>
                <td>{itype.value.replace('_', ' ').title()}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
            """)
        
        return f"""
        <div class="fingerprint-card">
            <h3>Interaction Fingerprint</h3>
            <table>
                <thead>
                    <tr>
                        <th>Interaction Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """
    
    def _create_table_from_tsv(self, lines: List[str]) -> str:
        """Create HTML table from TSV lines"""
        if not lines:
            return ""
        
        # Parse header
        header = lines[0].strip().split('\t')
        
        # Parse rows
        rows = []
        for line in lines[1:]:
            cells = line.strip().split('\t')
            rows.append(cells)
        
        # Build HTML
        header_html = "".join([f"<th>{h}</th>" for h in header])
        rows_html = []
        for row in rows:
            cells_html = "".join([f"<td>{cell}</td>" for cell in row])
            rows_html.append(f"<tr>{cells_html}</tr>")
        
        return f"""
        <table class="data-table">
            <thead>
                <tr>{header_html}</tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
        """
    
    def _build_html(self, complex_id: str, sections: List[str]) -> str:
        """Build complete HTML document"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report - {complex_id}</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        {"".join(sections)}
        
        <div class="footer">
            <p>Generated by PeptidQuantum - Protein-Peptide Interaction Visualization System</p>
            <p>Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
    
    def _get_css(self) -> str:
        """Get CSS styles"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #2E86AB 0%, #1a5f7a 100%);
            color: white;
            padding: 40px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header h2 {
            font-size: 1.8em;
            font-weight: normal;
            opacity: 0.9;
        }
        
        .timestamp {
            margin-top: 10px;
            opacity: 0.8;
            font-size: 0.9em;
        }
        
        .section {
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            color: #2E86AB;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .info-card {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #2E86AB;
        }
        
        .info-card h3 {
            color: #2E86AB;
            margin-bottom: 15px;
        }
        
        .info-card p {
            margin-bottom: 8px;
        }
        
        .figure-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }
        
        .figure {
            text-align: center;
        }
        
        .figure img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .figure-caption {
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
            text-align: left;
        }
        
        .fingerprint-card {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 20px;
        }
        
        .fingerprint-card h3 {
            color: #2E86AB;
            margin-bottom: 15px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background-color: #2E86AB;
            color: white;
            font-weight: bold;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .data-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .data-card {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
        }
        
        .data-card h3 {
            color: #2E86AB;
            margin-bottom: 15px;
        }
        
        .data-table {
            font-size: 0.85em;
            max-height: 400px;
            overflow-y: auto;
            display: block;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        a {
            color: #2E86AB;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        """
