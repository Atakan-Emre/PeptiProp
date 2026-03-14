"""Contact map visualization"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from ...interaction import InteractionSet, InteractionType, ContactMatrixGenerator


class ContactMapPlotter:
    """Generate contact map visualizations"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        
        # Color scheme for interaction types
        self.type_colors = {
            InteractionType.HBOND: '#2E86AB',           # Blue
            InteractionType.SALT_BRIDGE: '#A23B72',     # Purple
            InteractionType.HYDROPHOBIC: '#F18F01',     # Orange
            InteractionType.PI_STACKING: '#C73E1D',     # Red
            InteractionType.CATION_PI: '#6A994E',       # Green
            InteractionType.VDW: '#CCCCCC',             # Gray
            InteractionType.HALOGEN: '#BC4B51',         # Dark red
            InteractionType.METAL: '#8B8C89',           # Dark gray
            InteractionType.WATER_BRIDGE: '#4ECDC4',    # Cyan
            InteractionType.WEAK_HBOND: '#95B8D1',      # Light blue
        }
    
    def plot_contact_map(
        self,
        interaction_set: InteractionSet,
        protein_chain: str,
        peptide_chain: str,
        output_file: str | Path,
        aggregation: str = "count",
        show_labels: bool = True,
        annotate_hotspots: bool = True
    ):
        """
        Plot basic contact map
        
        Args:
            interaction_set: Set of interactions
            protein_chain: Protein chain ID
            peptide_chain: Peptide chain ID
            output_file: Output PNG file
            aggregation: "count", "binary", or "distance"
            show_labels: Show residue labels
            annotate_hotspots: Annotate hotspot residues
        """
        # Generate matrix
        generator = ContactMatrixGenerator()
        matrix, protein_res, peptide_res = generator.generate_matrix(
            interaction_set,
            protein_chain,
            peptide_chain,
            aggregation=aggregation
        )
        
        if matrix.size == 0:
            print("No interactions found for contact map")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot heatmap
        if aggregation == "distance":
            cmap = "viridis_r"  # Reverse for distance (shorter = better)
            cbar_label = "Distance (Å)"
        else:
            cmap = "YlOrRd"
            cbar_label = "Contact Count" if aggregation == "count" else "Contact"
        
        sns.heatmap(
            matrix,
            cmap=cmap,
            cbar_kws={'label': cbar_label},
            linewidths=0.5,
            linecolor='white',
            square=False,
            ax=ax
        )
        
        # Set labels
        if show_labels:
            ax.set_xticks(np.arange(len(protein_res)) + 0.5)
            ax.set_yticks(np.arange(len(peptide_res)) + 0.5)
            ax.set_xticklabels([f"{r}" for r in protein_res], rotation=90, fontsize=8)
            ax.set_yticklabels([f"{r}" for r in peptide_res], rotation=0, fontsize=8)
        
        ax.set_xlabel(f"Protein Chain {protein_chain} Residues", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"Peptide Chain {peptide_chain} Residues", fontsize=12, fontweight='bold')
        ax.set_title(
            f"Protein-Peptide Contact Map\n{interaction_set.complex_id}",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Annotate hotspots
        if annotate_hotspots:
            hotspots = generator.get_hotspot_residues(
                matrix, protein_res, peptide_res, top_n=3
            )
            
            # Mark protein hotspots
            for res_id, count in hotspots['protein'][:3]:
                idx = protein_res.index(res_id)
                ax.axvline(idx + 0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
            
            # Mark peptide hotspots
            for res_id, count in hotspots['peptide'][:3]:
                idx = peptide_res.index(res_id)
                ax.axhline(idx + 0.5, color='blue', linestyle='--', alpha=0.5, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Contact map saved to {output_file}")
    
    def plot_contact_map_by_type(
        self,
        interaction_set: InteractionSet,
        protein_chain: str,
        peptide_chain: str,
        output_file: str | Path,
        interaction_types: Optional[List[InteractionType]] = None
    ):
        """
        Plot separate contact maps for each interaction type
        
        Args:
            interaction_set: Set of interactions
            protein_chain: Protein chain ID
            peptide_chain: Peptide chain ID
            output_file: Output PNG file
            interaction_types: Specific types to plot (None = all)
        """
        generator = ContactMatrixGenerator()
        
        # Generate typed matrices
        typed_matrices = generator.generate_typed_matrices(
            interaction_set,
            protein_chain,
            peptide_chain
        )
        
        if not typed_matrices:
            print("No interactions found for type-specific maps")
            return
        
        # Filter types if specified
        if interaction_types:
            typed_matrices = {
                k: v for k, v in typed_matrices.items()
                if k in interaction_types
            }
        
        # Create subplots
        n_types = len(typed_matrices)
        ncols = min(3, n_types)
        nrows = (n_types + ncols - 1) // ncols
        
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 5, nrows * 4),
            dpi=self.dpi
        )
        
        if n_types == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 else axes
        
        # Plot each type
        for idx, (itype, matrix) in enumerate(sorted(typed_matrices.items(), key=lambda x: x[0].value)):
            ax = axes[idx]
            
            color = self.type_colors.get(itype, '#888888')
            
            # Create custom colormap
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list(
                'custom',
                ['white', color],
                N=256
            )
            
            sns.heatmap(
                matrix,
                cmap=cmap,
                cbar_kws={'label': 'Contact'},
                linewidths=0.5,
                linecolor='lightgray',
                square=False,
                ax=ax,
                vmin=0,
                vmax=1
            )
            
            ax.set_title(
                itype.value.replace('_', ' ').title(),
                fontsize=11,
                fontweight='bold',
                color=color
            )
            ax.set_xlabel("Protein Residues", fontsize=9)
            ax.set_ylabel("Peptide Residues", fontsize=9)
            ax.tick_params(labelsize=7)
        
        # Hide unused subplots
        for idx in range(n_types, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(
            f"Interaction Type-Specific Contact Maps\n{interaction_set.complex_id}",
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Type-specific contact maps saved to {output_file}")
    
    def plot_interaction_summary(
        self,
        interaction_set: InteractionSet,
        output_file: str | Path
    ):
        """
        Plot interaction summary statistics
        
        Args:
            interaction_set: Set of interactions
            output_file: Output PNG file
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        
        # 1. Interaction type distribution
        ax = axes[0, 0]
        type_counts = interaction_set.count_by_type()
        
        types = [t.value.replace('_', ' ').title() for t in type_counts.keys()]
        counts = list(type_counts.values())
        colors = [self.type_colors.get(t, '#888888') for t in type_counts.keys()]
        
        ax.barh(types, counts, color=colors)
        ax.set_xlabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Interaction Type Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 2. Distance distribution
        ax = axes[0, 1]
        distances = [i.distance for i in interaction_set.interactions if i.distance is not None]
        
        if distances:
            ax.hist(distances, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(distances), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(distances):.2f} Å', linewidth=2)
            ax.set_xlabel('Distance (Å)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Interaction Distance Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 3. Residue participation
        ax = axes[1, 0]
        
        protein_residues = {}
        peptide_residues = {}
        
        for interaction in interaction_set.interactions:
            prot_key = f"{interaction.protein_chain}:{interaction.protein_residue_id}"
            pep_key = f"{interaction.peptide_chain}:{interaction.peptide_residue_id}"
            
            protein_residues[prot_key] = protein_residues.get(prot_key, 0) + 1
            peptide_residues[pep_key] = peptide_residues.get(pep_key, 0) + 1
        
        top_protein = sorted(protein_residues.items(), key=lambda x: x[1], reverse=True)[:10]
        top_peptide = sorted(peptide_residues.items(), key=lambda x: x[1], reverse=True)[:10]
        
        y_pos = np.arange(len(top_protein))
        ax.barh(y_pos, [c for _, c in top_protein], color='#F18F01', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r for r, _ in top_protein], fontsize=9)
        ax.set_xlabel('Interaction Count', fontsize=11, fontweight='bold')
        ax.set_title('Top Protein Residues', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 4. Peptide residue participation
        ax = axes[1, 1]
        y_pos = np.arange(len(top_peptide))
        ax.barh(y_pos, [c for _, c in top_peptide], color='#6A994E', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r for r, _ in top_peptide], fontsize=9)
        ax.set_xlabel('Interaction Count', fontsize=11, fontweight='bold')
        ax.set_title('Top Peptide Residues', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle(
            f"Interaction Analysis Summary\n{interaction_set.complex_id}",
            fontsize=14,
            fontweight='bold',
            y=1.00
        )
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Interaction summary saved to {output_file}")
    
    def plot_chord_diagram(
        self,
        interaction_set: InteractionSet,
        protein_chain: str,
        peptide_chain: str,
        output_file: str | Path,
        min_interactions: int = 2
    ):
        """
        Plot chord diagram of interactions
        
        Args:
            interaction_set: Set of interactions
            protein_chain: Protein chain ID
            peptide_chain: Peptide chain ID
            output_file: Output PNG file
            min_interactions: Minimum interactions to show
        """
        # This would require a specialized library like circos or holoviews
        # For now, create a simplified circular plot
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi, subplot_kw=dict(projection='polar'))
        
        # Get residue pairs with counts
        from collections import defaultdict
        pair_counts = defaultdict(int)
        
        for interaction in interaction_set.interactions:
            if interaction.protein_chain == protein_chain and interaction.peptide_chain == peptide_chain:
                key = (interaction.protein_residue_id, interaction.peptide_residue_id)
                pair_counts[key] += 1
        
        # Filter by minimum
        pair_counts = {k: v for k, v in pair_counts.items() if v >= min_interactions}
        
        if not pair_counts:
            print("No residue pairs meet minimum interaction threshold")
            plt.close()
            return
        
        # Plot connections
        protein_residues = sorted(set(p for p, _ in pair_counts.keys()))
        peptide_residues = sorted(set(p for _, p in pair_counts.keys()))
        
        n_prot = len(protein_residues)
        n_pep = len(peptide_residues)
        
        # Assign angles
        prot_angles = np.linspace(0, np.pi, n_prot)
        pep_angles = np.linspace(np.pi, 2*np.pi, n_pep)
        
        prot_angle_map = {res: angle for res, angle in zip(protein_residues, prot_angles)}
        pep_angle_map = {res: angle for res, angle in zip(peptide_residues, pep_angles)}
        
        # Draw connections
        for (prot_res, pep_res), count in pair_counts.items():
            theta1 = prot_angle_map[prot_res]
            theta2 = pep_angle_map[pep_res]
            
            # Draw arc
            theta = np.linspace(theta1, theta2, 50)
            r = np.sin((theta - theta1) / (theta2 - theta1) * np.pi) * 0.8
            
            alpha = min(1.0, count / max(pair_counts.values()))
            ax.plot(theta, r, alpha=alpha, linewidth=count, color='#2E86AB')
        
        # Draw residue markers
        for res, angle in prot_angle_map.items():
            ax.plot([angle], [1.0], 'o', markersize=8, color='#F18F01', zorder=10)
            ax.text(angle, 1.1, str(res), ha='center', va='center', fontsize=8)
        
        for res, angle in pep_angle_map.items():
            ax.plot([angle], [1.0], 'o', markersize=8, color='#6A994E', zorder=10)
            ax.text(angle, 1.1, str(res), ha='center', va='center', fontsize=8)
        
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
        
        plt.title(
            f"Residue Interaction Network\n{interaction_set.complex_id}",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Chord diagram saved to {output_file}")
