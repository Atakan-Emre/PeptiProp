"""Interaction fingerprint generation"""
from __future__ import annotations

import json
from typing import Dict, List, Optional
from pathlib import Path
from collections import Counter

from ..schema import InteractionSet, InteractionType, StandardizedInteraction


class InteractionFingerprintBuilder:
    """Build interaction fingerprints for complexes"""
    
    def __init__(self):
        pass
    
    def build_fingerprint(
        self,
        interaction_set: InteractionSet,
        level: str = "residue"
    ) -> Dict:
        """
        Build interaction fingerprint
        
        Args:
            interaction_set: Set of interactions
            level: "residue" or "atom"
            
        Returns:
            Fingerprint dictionary
        """
        fingerprint = {
            'complex_id': interaction_set.complex_id,
            'total_interactions': len(interaction_set.interactions),
            'interaction_types': self._count_by_type(interaction_set),
            'residue_pairs': self._get_residue_pairs(interaction_set),
            'protein_residues': self._get_protein_residues(interaction_set),
            'peptide_residues': self._get_peptide_residues(interaction_set),
            'interaction_network': self._build_network(interaction_set),
            'statistics': self._calculate_statistics(interaction_set)
        }
        
        if level == "atom":
            fingerprint['atom_pairs'] = self._get_atom_pairs(interaction_set)
        
        return fingerprint
    
    def _count_by_type(self, interaction_set: InteractionSet) -> Dict[str, int]:
        """Count interactions by type"""
        counts = interaction_set.count_by_type()
        return {itype.value: count for itype, count in counts.items()}
    
    def _get_residue_pairs(
        self,
        interaction_set: InteractionSet
    ) -> List[Dict]:
        """Get all residue pairs with their interactions"""
        pairs = []
        
        # Group by residue pair
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for interaction in interaction_set.interactions:
            key = (
                interaction.protein_chain,
                interaction.protein_residue_id,
                interaction.peptide_chain,
                interaction.peptide_residue_id
            )
            grouped[key].append(interaction)
        
        # Build pair info
        for (prot_chain, prot_res, pep_chain, pep_res), interactions in grouped.items():
            pair_info = {
                'protein_chain': prot_chain,
                'protein_residue': prot_res,
                'protein_resname': interactions[0].protein_residue_name,
                'peptide_chain': pep_chain,
                'peptide_residue': pep_res,
                'peptide_resname': interactions[0].peptide_residue_name,
                'num_interactions': len(interactions),
                'interaction_types': [i.interaction_type.value for i in interactions],
                'min_distance': min(
                    (i.distance for i in interactions if i.distance is not None),
                    default=None
                )
            }
            pairs.append(pair_info)
        
        return pairs
    
    def _get_protein_residues(
        self,
        interaction_set: InteractionSet
    ) -> List[Dict]:
        """Get protein residues involved in interactions"""
        from collections import defaultdict
        residue_info = defaultdict(lambda: {
            'interaction_count': 0,
            'interaction_types': [],
            'partner_residues': []
        })
        
        for interaction in interaction_set.interactions:
            key = (interaction.protein_chain, interaction.protein_residue_id)
            info = residue_info[key]
            
            info['chain'] = interaction.protein_chain
            info['residue_id'] = interaction.protein_residue_id
            info['residue_name'] = interaction.protein_residue_name
            info['interaction_count'] += 1
            info['interaction_types'].append(interaction.interaction_type.value)
            info['partner_residues'].append({
                'chain': interaction.peptide_chain,
                'residue_id': interaction.peptide_residue_id,
                'residue_name': interaction.peptide_residue_name
            })
        
        # Convert to list and add unique interaction types
        result = []
        for info in residue_info.values():
            info['unique_interaction_types'] = list(set(info['interaction_types']))
            result.append(info)
        
        return sorted(result, key=lambda x: x['interaction_count'], reverse=True)
    
    def _get_peptide_residues(
        self,
        interaction_set: InteractionSet
    ) -> List[Dict]:
        """Get peptide residues involved in interactions"""
        from collections import defaultdict
        residue_info = defaultdict(lambda: {
            'interaction_count': 0,
            'interaction_types': [],
            'partner_residues': []
        })
        
        for interaction in interaction_set.interactions:
            key = (interaction.peptide_chain, interaction.peptide_residue_id)
            info = residue_info[key]
            
            info['chain'] = interaction.peptide_chain
            info['residue_id'] = interaction.peptide_residue_id
            info['residue_name'] = interaction.peptide_residue_name
            info['interaction_count'] += 1
            info['interaction_types'].append(interaction.interaction_type.value)
            info['partner_residues'].append({
                'chain': interaction.protein_chain,
                'residue_id': interaction.protein_residue_id,
                'residue_name': interaction.protein_residue_name
            })
        
        # Convert to list and add unique interaction types
        result = []
        for info in residue_info.values():
            info['unique_interaction_types'] = list(set(info['interaction_types']))
            result.append(info)
        
        return sorted(result, key=lambda x: x['interaction_count'], reverse=True)
    
    def _get_atom_pairs(
        self,
        interaction_set: InteractionSet
    ) -> List[Dict]:
        """Get atom-level interaction pairs"""
        pairs = []
        
        for interaction in interaction_set.interactions:
            if interaction.protein_atom and interaction.peptide_atom:
                pair_info = {
                    'protein_chain': interaction.protein_chain,
                    'protein_residue': interaction.protein_residue_id,
                    'protein_atom': interaction.protein_atom,
                    'peptide_chain': interaction.peptide_chain,
                    'peptide_residue': interaction.peptide_residue_id,
                    'peptide_atom': interaction.peptide_atom,
                    'interaction_type': interaction.interaction_type.value,
                    'distance': interaction.distance,
                    'angle': interaction.angle
                }
                pairs.append(pair_info)
        
        return pairs
    
    def _build_network(
        self,
        interaction_set: InteractionSet
    ) -> Dict:
        """Build interaction network representation"""
        nodes = {
            'protein': [],
            'peptide': []
        }
        edges = []
        
        # Collect unique residues
        protein_residues = set()
        peptide_residues = set()
        
        for interaction in interaction_set.interactions:
            protein_residues.add((
                interaction.protein_chain,
                interaction.protein_residue_id,
                interaction.protein_residue_name
            ))
            peptide_residues.add((
                interaction.peptide_chain,
                interaction.peptide_residue_id,
                interaction.peptide_residue_name
            ))
        
        # Add nodes
        for chain, res_id, res_name in sorted(protein_residues):
            nodes['protein'].append({
                'id': f"{chain}:{res_id}",
                'chain': chain,
                'residue_id': res_id,
                'residue_name': res_name
            })
        
        for chain, res_id, res_name in sorted(peptide_residues):
            nodes['peptide'].append({
                'id': f"{chain}:{res_id}",
                'chain': chain,
                'residue_id': res_id,
                'residue_name': res_name
            })
        
        # Add edges
        for interaction in interaction_set.interactions:
            edge = {
                'source': f"{interaction.protein_chain}:{interaction.protein_residue_id}",
                'target': f"{interaction.peptide_chain}:{interaction.peptide_residue_id}",
                'interaction_type': interaction.interaction_type.value,
                'distance': interaction.distance,
                'source_tool': interaction.source_tool
            }
            edges.append(edge)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'num_protein_nodes': len(nodes['protein']),
            'num_peptide_nodes': len(nodes['peptide']),
            'num_edges': len(edges)
        }
    
    def _calculate_statistics(
        self,
        interaction_set: InteractionSet
    ) -> Dict:
        """Calculate fingerprint statistics"""
        if not interaction_set.interactions:
            return {}
        
        distances = [i.distance for i in interaction_set.interactions if i.distance is not None]
        
        stats = {
            'num_unique_residue_pairs': len(interaction_set.get_unique_residue_pairs()),
            'num_interaction_types': len(interaction_set.get_interaction_types()),
            'interactions_per_residue_pair': (
                len(interaction_set.interactions) / 
                len(interaction_set.get_unique_residue_pairs())
                if interaction_set.get_unique_residue_pairs() else 0
            )
        }
        
        if distances:
            stats.update({
                'mean_distance': float(sum(distances) / len(distances)),
                'min_distance': float(min(distances)),
                'max_distance': float(max(distances))
            })
        
        return stats
    
    def save_json(
        self,
        fingerprint: Dict,
        output_file: str | Path
    ):
        """Save fingerprint to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(fingerprint, f, indent=2)
    
    def compare_fingerprints(
        self,
        fingerprint1: Dict,
        fingerprint2: Dict
    ) -> Dict:
        """
        Compare two fingerprints
        
        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            'complex1': fingerprint1['complex_id'],
            'complex2': fingerprint2['complex_id'],
            'interaction_count_diff': (
                fingerprint1['total_interactions'] - 
                fingerprint2['total_interactions']
            ),
            'common_interaction_types': list(
                set(fingerprint1['interaction_types'].keys()) &
                set(fingerprint2['interaction_types'].keys())
            ),
            'unique_to_complex1': list(
                set(fingerprint1['interaction_types'].keys()) -
                set(fingerprint2['interaction_types'].keys())
            ),
            'unique_to_complex2': list(
                set(fingerprint2['interaction_types'].keys()) -
                set(fingerprint1['interaction_types'].keys())
            )
        }
        
        return comparison
