"""
Dual-Loop Cognitive Controller v2.0
===================================
A hardware-aligned, manifold-preserving latent reasoning framework
for Transformer architectures.
"""

from .memory import CognitiveWorkingMemory
from .halting import EntropyHaltingUnit, LearnedHaltingGate, DriftDiffusionHalting
from .controller import RecurrentLatentController, TopKCapacityCrossAttention, LatentCritiqueRefinementUnit
from .plasticity import PlasticFastWeightUnit
from .evidential import EvidentialEpistemicGate
from .open_concept import OpenConceptSynthesizer
from .verification import (
    HypothesisVerificationGate,
    UncertaintySurpriseGate,
    ContrastiveEvidenceAccumulator,
    DirectionalSafetyProjection,
    AdaptiveSurpriseThreshold
)
from .matrix_helper import CognitiveMatrixHelper
from .decoder import DualLoopTransformer
from .adapters.latent_adapter import LatentDeliberationAdapter
from .adapters.qwen_adapter import (
    DualLoopQwenModel,
    DualLoopTransformerModel,
    attach_dual_loop_to_qwen,
    attach_dual_loop,
    attach_dual_loop_to_model
)

import os
import torch
from typing import Optional

__version__ = "2.2.3"

def get_default_checkpoint_path() -> Optional[str]:
    """
    Resolves the pre-trained reference checkpoint path by checking:
    1. Local working directory: './checkpoint_trained_dualloop.pt'
    2. Bundled package data: 'dual_loop/checkpoints/checkpoint_trained_dualloop.pt'
    """
    local_path = "checkpoint_trained_dualloop.pt"
    if os.path.exists(local_path):
        return os.path.abspath(local_path)
    
    pkg_path = os.path.join(os.path.dirname(__file__), "checkpoints", "checkpoint_trained_dualloop.pt")
    if os.path.exists(pkg_path):
        return os.path.abspath(pkg_path)

    return None

def load_trained_checkpoint(model: Optional[DualLoopTransformer] = None, checkpoint_path: Optional[str] = None):
    """
    Loads pre-trained weights into the DualLoopTransformer model, automatically falling back
    to the bundled package checkpoint if no path is provided.
    """
    resolved_path = checkpoint_path or get_default_checkpoint_path()
    if resolved_path is None or not os.path.exists(resolved_path):
        raise FileNotFoundError(
            "Pretrained checkpoint not found in local directory or bundled package data.\n"
            "To train the model from scratch, execute:\n"
            "  python train.py --epochs 35 --hops 3 --k_steps 3\n"
            "or run:\n"
            "  python evaluate_real_behavior.py"
        )

    state_dict = torch.load(resolved_path, map_location="cpu", weights_only=True)
    if model is not None:
        model.load_state_dict(state_dict, strict=False)
        return model, resolved_path
    return state_dict

__all__ = [
    "CognitiveWorkingMemory",
    "EntropyHaltingUnit",
    "LearnedHaltingGate",
    "DriftDiffusionHalting",
    "RecurrentLatentController",
    "TopKCapacityCrossAttention",
    "LatentCritiqueRefinementUnit",
    "PlasticFastWeightUnit",
    "EvidentialEpistemicGate",
    "OpenConceptSynthesizer",
    "HypothesisVerificationGate",
    "UncertaintySurpriseGate",
    "ContrastiveEvidenceAccumulator",
    "DirectionalSafetyProjection",
    "AdaptiveSurpriseThreshold",
    "CognitiveMatrixHelper",
    "DualLoopTransformer",
    "LatentDeliberationAdapter",
    "DualLoopQwenModel",
    "DualLoopTransformerModel",
    "attach_dual_loop_to_qwen",
    "attach_dual_loop",
    "attach_dual_loop_to_model",
    "get_default_checkpoint_path",
    "load_trained_checkpoint",
]
