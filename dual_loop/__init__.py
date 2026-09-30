"""
Dual-Loop Cognitive Controller v2.0 / v3.1
==========================================
A hardware-aligned, manifold-preserving latent reasoning framework
for Transformer architectures.
"""

__version__ = "3.1.1"

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    from .memory import CognitiveWorkingMemory, EpisodicMemoryBuffer
    from .halting import EntropyHaltingUnit, LearnedHaltingGate, DriftDiffusionHalting
    from .controller import RecurrentLatentController, TopKCapacityCrossAttention, LatentCritiqueRefinementUnit
    from .plasticity import PlasticFastWeightUnit
    from .evidential import EvidentialEpistemicGate
    from .open_concept import OpenConceptSynthesizer
    from .homeostasis import HomeostaticDriveEngine, ActiveInferencePolicyRouter
    from .nullspace_engine import OrthogonalNullspaceProjector
    from .allostasis import AllostaticEnergyModulator
    from .curiosity_daemon import (
        EpistemicHumilityModule,
        IntrinsicCuriosityModule,
        PopperianSelfPlayEngine,
        AutonomousDaemonController
    )
    from .functorial_engine import FunctorialCrossDomainMapper
    from .mdl_selector import NeuroSymbolicMDLSelector
    from .verification import (
        HypothesisVerificationGate,
        UncertaintySurpriseGate,
        ContrastiveEvidenceAccumulator,
        DirectionalSafetyProjection,
        AdaptiveSurpriseThreshold
    )
    from .matrix_helper import CognitiveMatrixHelper
    from .cognitive_judge import ProbabilisticCognitiveJudge
    from .directional_reservoir import ContextDirectionalRouter, CompactCommonSenseReservoir
    from .decoder import DualLoopTransformer
    from .multimodal_transport import ProcrustesOptimalManifoldTransport
    from .topological_cwm import SpatioTemporalEntropicCWM
    from .plasticity import HeteroAssociativePlasticMemory
    from .adapters.latent_adapter import LatentDeliberationAdapter
    from .adapters.qwen_adapter import (
        DualLoopQwenModel,
        DualLoopTransformerModel,
        attach_dual_loop_to_qwen,
        attach_dual_loop,
        attach_dual_loop_to_model,
        attach_dual_loop_to_glm4,
        attach_dual_loop_to_glm
    )
    from .adapters.universal_adapter import (
        UniversalDualLoopAdapter,
        UniversalDualLoopModelWrapper,
        DynamicGraphIntrospector,
        attach_universal_dual_loop,
        CANONICAL_DIM
    )
    from .firewall import (
        SheafInvariantFirewall,
        BoundedNormInvariant,
        DirectionalStabilityInvariant,
        DirichletVacuityInvariant,
        CodeExecutionIntegrityInvariant
    )
    from .sleep_consolidation import SleepPhaseConsolidationEngine, SleepMemoryEpisode
    from .cognitive_os import (
        UnifiedCognitiveOS,
        CognitiveConflictEvaluator,
        VitalStateVector
    )
    from .hologram import (
        FISTALatentRecovery,
        LatentReconstructiveHologram,
        SkeletonQuantizer
    )
    from .adapters.qwen3_8_adapter import (
        attach_dual_loop_to_qwen3_8,
        Qwen3_8HologramModel,
        calculate_qwen3_8_vram_budget,
        QWEN3_8_27B_D_NATIVE,
        QWEN3_8_27B_TARGET_LAYER
    )

    # Convenient shorthand alias for users: default to Universal Adapter in v3.0!
    attach = attach_universal_dual_loop

    import os
    from typing import Optional

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

    def auto_attach_hadl(*args, **kwargs):
        from .runtime.detector import auto_attach_hadl as _fn
        return _fn(*args, **kwargs)

    __all__ = [
        "CognitiveWorkingMemory",
        "EpisodicMemoryBuffer",
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
        "ProcrustesOptimalManifoldTransport",
        "SpatioTemporalEntropicCWM",
        "HeteroAssociativePlasticMemory",
        "AllostaticEnergyModulator",
        "EpistemicHumilityModule",
        "IntrinsicCuriosityModule",
        "PopperianSelfPlayEngine",
        "AutonomousDaemonController",
        "DualLoopQwenModel",
        "UniversalDualLoopAdapter",
        "UniversalDualLoopModelWrapper",
        "DynamicGraphIntrospector",
        "attach_universal_dual_loop",
        "CANONICAL_DIM",
        "SheafInvariantFirewall",
        "BoundedNormInvariant",
        "DirectionalStabilityInvariant",
        "DirichletVacuityInvariant",
        "CodeExecutionIntegrityInvariant",
        "SleepPhaseConsolidationEngine",
        "SleepMemoryEpisode",
        "UnifiedCognitiveOS",
        "CognitiveConflictEvaluator",
        "VitalStateVector",
        "FISTALatentRecovery",
        "LatentReconstructiveHologram",
        "SkeletonQuantizer",
        "attach_dual_loop_to_qwen3_8",
        "Qwen3_8HologramModel",
        "calculate_qwen3_8_vram_budget",
        "QWEN3_8_27B_D_NATIVE",
        "QWEN3_8_27B_TARGET_LAYER",
        "attach_dual_loop_to_qwen",
        "attach_dual_loop",
        "attach",
        "attach_dual_loop_to_model",
        "attach_dual_loop_to_glm4",
        "attach_dual_loop_to_glm",
        "get_default_checkpoint_path",
        "load_trained_checkpoint",
        "auto_attach_hadl"
    ]

else:
    def _require_torch(attr_name: str = ""):
        msg = (
            "\n"
            + "=" * 80 + "\n"
            + "[!] PyTorch is required to run 'dual-loop-controller' but is not currently installed.\n"
            + "=" * 80 + "\n"
            + "[*] To install with NVIDIA CUDA GPU acceleration (Recommended for NVIDIA GPUs):\n"
            + "    pip install torch --index-url https://download.pytorch.org/whl/cu124\n\n"
            + "[*] To install PyTorch CPU only:\n"
            + "    pip install torch\n\n"
            + "[*] Windows Note:\n"
            + "    If pip fails with '[Errno 2] No such file or directory' due to the 260-char MAX_PATH limit,\n"
            + "    either:\n"
            + "      1. Run 'fix_windows_longpaths.bat' as Administrator, or run in Admin CMD:\n"
            + "         reg add \"HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\" /v \"LongPathsEnabled\" /t REG_DWORD /d 1 /f\n"
            + "      2. Or use a virtual environment (recommended):\n"
            + "         python -m venv .venv\n"
            + "         .venv\\Scripts\\activate\n"
            + "         pip install torch dual-loop-controller\n"
            + "=" * 80 + "\n"
        )
        raise ImportError(msg)

    def __getattr__(name: str):
        _require_torch(name)

    __all__ = []
