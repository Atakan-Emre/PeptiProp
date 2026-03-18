"""Dataset quality control dashboard generator"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json

sns.set_style("whitegrid")


class DatasetQCDashboard:
    """Generate quality control dashboard for dataset"""
    
    def __init__(self, canonical_dir: str | Path, splits_dir: str | Path):
        """
        Initialize QC dashboard
        
        Args:
            canonical_dir: Directory containing canonical parquet files
            splits_dir: Directory containing split files
        """
        self.canonical_dir = Path(canonical_dir)
        self.splits_dir = Path(splits_dir)
        
        # Load data
        self.complexes = pd.read_parquet(self.canonical_dir / "complexes.parquet")
        self.chains = pd.read_parquet(self.canonical_dir / "chains.parquet")
        self.residues = pd.read_parquet(self.canonical_dir / "residues.parquet")
        self.interactions = pd.read_parquet(self.canonical_dir / "interactions.parquet")
    
    def generate_dashboard(self, output_file: str | Path):
        """
        Generate complete QC dashboard
        
        Args:
            output_file: Output HTML file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        plots = []
        
        plots.append(self._plot_peptide_length_distribution())
        plots.append(self._plot_resolution_distribution())
        plots.append(self._plot_interaction_type_distribution())
        plots.append(self._plot_source_database_breakdown())
        plots.append(self._plot_split_distribution())
        plots.append(self._plot_quality_flags())
        plots.append(self._plot_pocket_size_distribution())
        plots.append(self._plot_interaction_count_per_complex())
        
        # Generate statistics
        stats = self._compute_statistics()
        
        # Build HTML
        html = self._build_html(plots, stats)
        
        # Save
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"QC dashboard saved to {output_file}")
    
    def _plot_peptide_length_distribution(self) -> str:
        """Plot peptide length histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self.complexes['peptide_length'], bins=50, 
               color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.axvline(5, color='red', linestyle='--', label='Min (5 aa)')
        ax.axvline(30, color='orange', linestyle='--', label='Core max (30 aa)')
        ax.axvline(50, color='green', linestyle='--', label='Extension max (50 aa)')
        
        ax.set_xlabel('Peptide Length (aa)', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Peptide Length Distribution', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _plot_resolution_distribution(self) -> str:
        """Plot resolution distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter experimental structures with resolution
        exp_data = self.complexes[
            (self.complexes['structure_source'] == 'experimental') & 
            (self.complexes['resolution'].notna())
        ]
        
        if len(exp_data) > 0:
            ax.hist(exp_data['resolution'], bins=50, 
                   color='#F18F01', alpha=0.7, edgecolor='black')
            ax.axvline(exp_data['resolution'].median(), color='red', 
                      linestyle='--', label=f'Median: {exp_data["resolution"].median():.2f} Å')
            
            ax.set_xlabel('Resolution (Å)', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title('Resolution Distribution (Experimental Structures)', 
                        fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No resolution data available', 
                   ha='center', va='center', fontsize=14)
        
        return self._fig_to_base64(fig)
    
    def _plot_interaction_type_distribution(self) -> str:
        """Plot interaction type distribution"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if len(self.interactions) > 0:
            type_counts = self.interactions['interaction_type'].value_counts()
            
            colors = plt.cm.Set3(range(len(type_counts)))
            type_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
            
            ax.set_xlabel('Interaction Type', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title('Interaction Type Distribution', fontweight='bold', fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No interaction data available', 
                   ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_source_database_breakdown(self) -> str:
        """Plot source database breakdown"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        source_counts = self.complexes['source_db'].value_counts()
        
        colors = ['#2E86AB', '#F18F01', '#6A994E', '#A23B72']
        ax.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%',
              colors=colors, startangle=90)
        ax.set_title('Source Database Distribution', fontweight='bold', fontsize=14)
        
        return self._fig_to_base64(fig)
    
    def _plot_split_distribution(self) -> str:
        """Plot train/val/test split distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        split_counts = self.complexes['split_tag'].value_counts()
        
        colors = {'train': '#2E86AB', 'val': '#F18F01', 
                 'test': '#6A994E', 'external': '#A23B72'}
        bar_colors = [colors.get(s, '#888888') for s in split_counts.index]
        
        split_counts.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black')
        
        ax.set_xlabel('Split', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Dataset Split Distribution', fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(alpha=0.3, axis='y')
        
        # Add percentages
        total = split_counts.sum()
        for i, (split, count) in enumerate(split_counts.items()):
            pct = (count / total) * 100
            ax.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_quality_flags(self) -> str:
        """Plot quality flag distribution"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        quality_counts = self.complexes['quality_flag'].value_counts()
        
        colors = {'clean': '#6A994E', 'warning': '#F18F01', 'quarantine': '#C73E1D'}
        bar_colors = [colors.get(q, '#888888') for q in quality_counts.index]
        
        quality_counts.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black')
        
        ax.set_xlabel('Quality Flag', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Quality Flag Distribution', fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_pocket_size_distribution(self) -> str:
        """Plot pocket size distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Count pocket residues per complex
        pocket_sizes = self.residues[self.residues['is_pocket']].groupby('complex_id').size()
        
        if len(pocket_sizes) > 0:
            ax.hist(pocket_sizes, bins=50, color='#6A994E', alpha=0.7, edgecolor='black')
            ax.axvline(pocket_sizes.median(), color='red', linestyle='--',
                      label=f'Median: {pocket_sizes.median():.0f} residues')
            
            ax.set_xlabel('Pocket Size (residues)', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title('Binding Pocket Size Distribution', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No pocket data available', 
                   ha='center', va='center', fontsize=14)
        
        return self._fig_to_base64(fig)
    
    def _plot_interaction_count_per_complex(self) -> str:
        """Plot interaction count per complex"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(self.interactions) > 0:
            interaction_counts = self.interactions.groupby('complex_id').size()
            
            ax.hist(interaction_counts, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
            ax.axvline(interaction_counts.median(), color='red', linestyle='--',
                      label=f'Median: {interaction_counts.median():.0f} interactions')
            
            ax.set_xlabel('Interactions per Complex', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title('Interaction Count Distribution', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No interaction data available', 
                   ha='center', va='center', fontsize=14)
        
        return self._fig_to_base64(fig)
    
    def _compute_statistics(self) -> Dict:
        """Compute dataset statistics"""
        stats = {
            "total_complexes": len(self.complexes),
            "total_chains": len(self.chains),
            "total_residues": len(self.residues),
            "total_interactions": len(self.interactions),
            "peptide_length": {
                "min": int(self.complexes['peptide_length'].min()),
                "max": int(self.complexes['peptide_length'].max()),
                "mean": float(self.complexes['peptide_length'].mean()),
                "median": float(self.complexes['peptide_length'].median())
            },
            "split_counts": self.complexes['split_tag'].value_counts().to_dict(),
            "source_counts": self.complexes['source_db'].value_counts().to_dict(),
            "quality_counts": self.complexes['quality_flag'].value_counts().to_dict()
        }
        
        # Resolution stats (if available)
        exp_data = self.complexes[
            (self.complexes['structure_source'] == 'experimental') & 
            (self.complexes['resolution'].notna())
        ]
        if len(exp_data) > 0:
            stats["resolution"] = {
                "min": float(exp_data['resolution'].min()),
                "max": float(exp_data['resolution'].max()),
                "mean": float(exp_data['resolution'].mean()),
                "median": float(exp_data['resolution'].median())
            }
        
        return stats
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        import io
        import base64
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return f'data:image/png;base64,{img_base64}'
    
    def _build_html(self, plots: List[str], stats: Dict) -> str:
        """Build HTML dashboard"""
        stats_json = json.dumps(stats, indent=2)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Quality Control Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2E86AB;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #333;
            margin-top: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #2E86AB;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2E86AB;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dataset Quality Control Dashboard</h1>
        
        <h2>Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['total_complexes']:,}</div>
                <div class="stat-label">Total Complexes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['total_interactions']:,}</div>
                <div class="stat-label">Total Interactions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['peptide_length']['mean']:.1f}</div>
                <div class="stat-label">Mean Peptide Length (aa)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('resolution', {}).get('mean', 'N/A')}</div>
                <div class="stat-label">Mean Resolution (Å)</div>
            </div>
        </div>
        
        <h2>Visualizations</h2>
        
        <div class="plot">
            <img src="{plots[0]}" alt="Peptide Length Distribution">
        </div>
        
        <div class="plot">
            <img src="{plots[1]}" alt="Resolution Distribution">
        </div>
        
        <div class="plot">
            <img src="{plots[2]}" alt="Interaction Type Distribution">
        </div>
        
        <div class="plot">
            <img src="{plots[3]}" alt="Source Database Breakdown">
        </div>
        
        <div class="plot">
            <img src="{plots[4]}" alt="Split Distribution">
        </div>
        
        <div class="plot">
            <img src="{plots[5]}" alt="Quality Flags">
        </div>
        
        <div class="plot">
            <img src="{plots[6]}" alt="Pocket Size Distribution">
        </div>
        
        <div class="plot">
            <img src="{plots[7]}" alt="Interaction Count Distribution">
        </div>
        
        <h2>Detailed Statistics</h2>
        <pre>{stats_json}</pre>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dataset QC dashboard")
    parser.add_argument("--canonical", type=Path, required=True,
                       help="Canonical data directory")
    parser.add_argument("--splits", type=Path, required=True,
                       help="Splits directory")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output HTML file")
    
    args = parser.parse_args()
    
    dashboard = DatasetQCDashboard(args.canonical, args.splits)
    dashboard.generate_dashboard(args.output)
