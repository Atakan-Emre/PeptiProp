"""Merge interactions from multiple tools"""
from __future__ import annotations

from typing import List, Dict
from collections import defaultdict

from ..schema import StandardizedInteraction, InteractionSet, InteractionType


class InteractionMerger:
    """Merge and deduplicate interactions from multiple sources"""
    
    def __init__(self, confidence_weights: Optional[Dict[str, float]] = None):
        """
        Initialize merger
        
        Args:
            confidence_weights: Weights for different tools (default: equal)
        """
        self.confidence_weights = confidence_weights or {
            'arpeggio': 1.0,
            'plip': 1.0
        }
    
    def merge(
        self,
        *interaction_sets: InteractionSet,
        strategy: str = "union"
    ) -> InteractionSet:
        """
        Merge multiple interaction sets
        
        Args:
            *interaction_sets: Variable number of InteractionSet objects
            strategy: "union" (all), "intersection" (common), or "consensus" (majority)
            
        Returns:
            Merged InteractionSet
        """
        if not interaction_sets:
            return InteractionSet(complex_id="merged", interactions=[])
        
        complex_id = interaction_sets[0].complex_id
        
        if strategy == "union":
            merged = self._merge_union(interaction_sets)
        elif strategy == "intersection":
            merged = self._merge_intersection(interaction_sets)
        elif strategy == "consensus":
            merged = self._merge_consensus(interaction_sets)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return InteractionSet(complex_id=complex_id, interactions=merged)
    
    def _merge_union(
        self,
        interaction_sets: tuple[InteractionSet, ...]
    ) -> List[StandardizedInteraction]:
        """
        Union: Include all interactions from all sources
        Deduplicate based on residue pairs
        """
        # Group by residue pair key
        grouped = defaultdict(list)
        
        for iset in interaction_sets:
            for interaction in iset.interactions:
                key = interaction.residue_pair_key
                grouped[key].append(interaction)
        
        # For each group, select best representative
        merged = []
        for key, interactions in grouped.items():
            if len(interactions) == 1:
                merged.append(interactions[0])
            else:
                # Multiple sources agree - merge them
                best = self._select_best_interaction(interactions)
                merged.append(best)
        
        return merged
    
    def _merge_intersection(
        self,
        interaction_sets: tuple[InteractionSet, ...]
    ) -> List[StandardizedInteraction]:
        """
        Intersection: Only include interactions found by all sources
        """
        if len(interaction_sets) < 2:
            return interaction_sets[0].interactions if interaction_sets else []
        
        # Group by residue pair key for each set
        sets_by_key = []
        for iset in interaction_sets:
            keys = {i.residue_pair_key for i in iset.interactions}
            sets_by_key.append(keys)
        
        # Find common keys
        common_keys = set.intersection(*sets_by_key)
        
        # Collect interactions with common keys
        grouped = defaultdict(list)
        for iset in interaction_sets:
            for interaction in iset.interactions:
                if interaction.residue_pair_key in common_keys:
                    grouped[interaction.residue_pair_key].append(interaction)
        
        # Select best from each group
        merged = []
        for key, interactions in grouped.items():
            best = self._select_best_interaction(interactions)
            merged.append(best)
        
        return merged
    
    def _merge_consensus(
        self,
        interaction_sets: tuple[InteractionSet, ...],
        threshold: float = 0.5
    ) -> List[StandardizedInteraction]:
        """
        Consensus: Include interactions found by majority of sources
        """
        # Group by residue pair key
        grouped = defaultdict(list)
        
        for iset in interaction_sets:
            for interaction in iset.interactions:
                key = interaction.residue_pair_key
                grouped[key].append(interaction)
        
        # Filter by consensus threshold
        min_sources = max(1, int(len(interaction_sets) * threshold))
        
        merged = []
        for key, interactions in grouped.items():
            if len(set(i.source_tool for i in interactions)) >= min_sources:
                best = self._select_best_interaction(interactions)
                merged.append(best)
        
        return merged
    
    def _select_best_interaction(
        self,
        interactions: List[StandardizedInteraction]
    ) -> StandardizedInteraction:
        """
        Select best interaction from duplicates
        
        Priority:
        1. Shortest distance (if available)
        2. Higher confidence
        3. Weighted by tool
        """
        if len(interactions) == 1:
            return interactions[0]
        
        # Score each interaction
        scored = []
        for interaction in interactions:
            score = 0.0
            
            # Distance score (shorter is better)
            if interaction.distance is not None:
                score += 1.0 / (1.0 + interaction.distance)
            
            # Confidence score
            score += interaction.confidence
            
            # Tool weight
            tool_weight = self.confidence_weights.get(interaction.source_tool, 1.0)
            score *= tool_weight
            
            scored.append((score, interaction))
        
        # Return highest scoring
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        
        # If multiple sources, increase confidence
        if len(interactions) > 1:
            best.confidence = min(1.0, best.confidence * 1.2)
        
        return best
    
    def deduplicate(
        self,
        interactions: List[StandardizedInteraction],
        level: str = "residue"
    ) -> List[StandardizedInteraction]:
        """
        Remove duplicate interactions
        
        Args:
            interactions: List of interactions
            level: "residue" or "atom"
            
        Returns:
            Deduplicated list
        """
        seen = set()
        unique = []
        
        for interaction in interactions:
            if level == "residue":
                key = interaction.residue_pair_key
            else:
                key = interaction.atom_pair_key
            
            if key not in seen:
                seen.add(key)
                unique.append(interaction)
        
        return unique
