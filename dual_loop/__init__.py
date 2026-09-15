"""
Dual-Loop Cognitive Controller v2.0
===================================
A hardware-aligned, manifold-preserving latent reasoning framework
for Transformer architectures.
"""

from .memory import CognitiveWorkingMemory
from .halting import EntropyHaltingUnit
from .controller import RecurrentLatentController, TopKCapacityCrossAttention
from .decoder import DualLoopTransformer
from .adapters.latent_adapter import LatentDeliberationAdapter

__version__ = "2.0.0a1"
__all__ = [
    "CognitiveWorkingMemory",
    "EntropyHaltingUnit",
    "RecurrentLatentController",
    "TopKCapacityCrossAttention",
    "DualLoopTransformer",
    "LatentDeliberationAdapter",
]
