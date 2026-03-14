"""Contact matrix generation and analysis"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..schema import InteractionSet, InteractionType, StandardizedInteraction


class ContactMatrixGenerator:
    """Generate residue-residue contact matrices"""
    
    def __init__(self):
        pass
    
    def generate_matrix(
        self,
        interaction_set: InteractionSet,
        protein_chain: str,
        peptide_chain: str,
        aggregation: str = "count"
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Generate contact matrix
        
        Args:
            interaction_set: Set of interactions
            protein_chain: Protein chain ID
            peptide_chain: Peptide chain ID
            aggregation: "count", "binary", "distance", or "type"
            
        Returns:
            Tuple of (matrix, protein_residue_ids, peptide_residue_ids)
        """
        # Filter interactions for these chains
        interactions = [
            i for i in interaction_set.interactions
            if i.protein_chain == protein_chain and i.peptide_chain == peptide_chain
        ]
        
        if not interactions:
            return np.array([]), [], []
        
        # Get unique residue IDs
        protein_residues = sorted(set(i.protein_residue_id for i in interactions))
        peptide_residues = sorted(set(i.peptide_residue_id for i in interactions))
        
        # Create matrix
        n_protein = len(protein_residues)
        n_peptide = len(peptide_residues)
        matrix = np.zeros((n_peptide, n_protein))
        
        # Map residue IDs to indices
        protein_idx = {res_id: idx for idx, res_id in enumerate(protein_residues)}
        peptide_idx = {res_id: idx for idx, res_id in enumerate(peptide_residues)}
        
        # Fill matrix
        for interaction in interactions:
            p_idx = protein_idx[interaction.protein_residue_id]
            pep_idx = peptide_idx[interaction.peptide_residue_id]
            
            if aggregation == "count":
                matrix[pep_idx, p_idx] += 1
            elif aggregation == "binary":
                matrix[pep_idx, p_idx] = 1
            elif aggregation == "distance":
                if interaction.distance is not None:
                    # Store minimum distance
                    if matrix[pep_idx, p_idx] == 0:
                        matrix[pep_idx, p_idx] = interaction.distance
                    else:
                        matrix[pep_idx, p_idx] = min(matrix[pep_idx, p_idx], interaction.distance)
            elif aggregation == "type":
                # Encode interaction type as number
                type_value = self._interaction_type_to_value(interaction.interaction_type)
                matrix[pep_idx, p_idx] = max(matrix[pep_idx, p_idx], type_value)
        
        return matrix, protein_residues, peptide_residues
    
    def generate_typed_matrices(
        self,
        interaction_set: InteractionSet,
        protein_chain: str,
        peptide_chain: str
    ) -> Dict[InteractionType, np.ndarray]:
        """
        Generate separate matrices for each interaction type
        
        Args:
            interaction_set: Set of interactions
            protein_chain: Protein chain ID
            peptide_chain: Peptide chain ID
            
        Returns:
            Dictionary mapping interaction types to matrices
        """
        # Get all interaction types
        interaction_types = interaction_set.get_interaction_types()
        
        matrices = {}
        for itype in interaction_types:
            # Filter by type
            typed_interactions = interaction_set.filter_by_type(itype)
            typed_set = InteractionSet(
                complex_id=interaction_set.complex_id,
                interactions=typed_interactions
            )
            
            # Generate matrix
            matrix, protein_res, peptide_res = self.generate_matrix(
                typed_set,
                protein_chain,
                peptide_chain,
                aggregation="binary"
            )
            
            if matrix.size > 0:
                matrices[itype] = matrix
        
        return matrices
    
    def to_dataframe(
        self,
        matrix: np.ndarray,
        protein_residues: List[int],
        peptide_residues: List[int]
    ) -> pd.DataFrame:
        """
        Convert matrix to DataFrame
        
        Args:
            matrix: Contact matrix
            protein_residues: Protein residue IDs (columns)
            peptide_residues: Peptide residue IDs (rows)
            
        Returns:
            DataFrame with labeled rows and columns
        """
        return pd.DataFrame(
            matrix,
            index=[f"P{r}" for r in peptide_residues],
            columns=[f"R{r}" for r in protein_residues]
        )
    
    def save_csv(
        self,
        matrix: np.ndarray,
        protein_residues: List[int],
        peptide_residues: List[int],
        output_file: str | Path
    ):
        """Save matrix as CSV"""
        df = self.to_dataframe(matrix, protein_residues, peptide_residues)
        df.to_csv(output_file)
    
    def calculate_statistics(
        self,
        matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate matrix statistics
        
        Args:
            matrix: Contact matrix
            
        Returns:
            Dictionary of statistics
        """
        if matrix.size == 0:
            return {}
        
        return {
            'total_contacts': float(np.sum(matrix > 0)),
            'mean_contacts_per_peptide_residue': float(np.mean(np.sum(matrix > 0, axis=1))),
            'mean_contacts_per_protein_residue': float(np.mean(np.sum(matrix > 0, axis=0))),
            'max_contacts': float(np.max(matrix)),
            'density': float(np.sum(matrix > 0) / matrix.size),
            'peptide_residues_with_contacts': float(np.sum(np.any(matrix > 0, axis=1))),
            'protein_residues_with_contacts': float(np.sum(np.any(matrix > 0, axis=0)))
        }
    
    def get_hotspot_residues(
        self,
        matrix: np.ndarray,
        protein_residues: List[int],
        peptide_residues: List[int],
        top_n: int = 5
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Identify hotspot residues with most contacts
        
        Args:
            matrix: Contact matrix
            protein_residues: Protein residue IDs
            peptide_residues: Peptide residue IDs
            top_n: Number of top residues to return
            
        Returns:
            Dictionary with 'protein' and 'peptide' hotspots
        """
        # Sum contacts per residue
        protein_contacts = np.sum(matrix, axis=0)
        peptide_contacts = np.sum(matrix, axis=1)
        
        # Get top N
        protein_top_idx = np.argsort(protein_contacts)[-top_n:][::-1]
        peptide_top_idx = np.argsort(peptide_contacts)[-top_n:][::-1]
        
        protein_hotspots = [
            (protein_residues[idx], float(protein_contacts[idx]))
            for idx in protein_top_idx
            if protein_contacts[idx] > 0
        ]
        
        peptide_hotspots = [
            (peptide_residues[idx], float(peptide_contacts[idx]))
            for idx in peptide_top_idx
            if peptide_contacts[idx] > 0
        ]
        
        return {
            'protein': protein_hotspots,
            'peptide': peptide_hotspots
        }
    
    @staticmethod
    def _interaction_type_to_value(itype: InteractionType) -> float:
        """Map interaction type to numeric value for encoding"""
        mapping = {
            InteractionType.HBOND: 10.0,
            InteractionType.SALT_BRIDGE: 9.0,
            InteractionType.PI_STACKING: 8.0,
            InteractionType.CATION_PI: 7.0,
            InteractionType.HYDROPHOBIC: 6.0,
            InteractionType.HALOGEN: 5.0,
            InteractionType.METAL: 4.0,
            InteractionType.VDW: 3.0,
            InteractionType.WEAK_HBOND: 2.0,
            InteractionType.WATER_BRIDGE: 1.0,
        }
        return mapping.get(itype, 1.0)
