"""Interaction extraction and analysis module"""

from .schema import (
    InteractionType,
    StandardizedInteraction,
    InteractionSet
)

from .extractors.arpeggio_wrapper import ArpeggioWrapper
from .extractors.plip_wrapper import PLIPWrapper
from .extractors.merger import InteractionMerger

from .analysis.contact_matrix import ContactMatrixGenerator
from .analysis.fingerprint import InteractionFingerprintBuilder

__all__ = [
    # Schema
    "InteractionType",
    "StandardizedInteraction",
    "InteractionSet",
    
    # Extractors
    "ArpeggioWrapper",
    "PLIPWrapper",
    "InteractionMerger",
    
    # Analysis
    "ContactMatrixGenerator",
    "InteractionFingerprintBuilder"
]
