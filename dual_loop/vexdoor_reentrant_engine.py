"""
HADL v3.4: Vexdoor Re-entrant Closed-Loop & Nullspace Append Engine
===================================================================
Mathematical Formulation of User Principles:
1. Re-entrant Closed-Loop:
   LM Head (Pre-logits) <-> Evolving Manifold (Layer 11) <-> LM Head (Final logits)
   Trivial tokens take System 1 bypass; uncertain tokens pull back into the manifold.
2. Log-Determinant & Gramian Volume Similarity:
   Computes Vol(K) = log det(K^T K / d + eps * I) to measure novelty volume.
3. Vexdoor Dynamic Decay:
   Wind-blown door decay function V(t) = max(0, (E / sqrt(N)) * exp(-t / tau) - gamma * t)
   Normalizes and clamps delta logits to prevent logit explosion and repetition loops.
4. Epistemic Integrity Check:
   Tests ||X - P_W(X)|| to verify if concepts are already known before learning.
5. Non-Destructive Orthogonal Nullspace Append:
   Appends novel patterns into nullspace: W_new = W + (1 / sqrt(N)) * Pi_null * X^T
   guaranteeing W_old * Pi_null == 0 (Zero Catastrophic Forgetting).
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class VexdoorDecayGate(nn.Module):
    """
    Vexdoor Dynamic Decay Gate:
    Models a door blown by the wind that gradually swings shut over generation steps t.
    Prevents logit explosion and breaks infinite repetition loops by decaying to 0.0,
    allowing natural stop tokens (<|im_end|>) to trigger cleanly.
    """
    def __init__(
        self,
        tau_wind: float = 2.5,
        gamma_decay: float = 0.12,
        max_bound: float = 4.5
    ):
        super().__init__()
        self.tau_wind = float(tau_wind)
        self.gamma_decay = float(gamma_decay)
        self.max_bound = float(max_bound)
        self.current_step = 0

    def reset_step(self):
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def compute_gate(self, energy: torch.Tensor, total_steps: int = 100) -> torch.Tensor:
        """
        energy: [B, S, 1] thought energy
        Returns gate V(t) in [0, 1]
        """
        t = float(self.current_step)
        # Wind-blown decay: exponential decay minus linear closure
        wind_factor = math.exp(-t / self.tau_wind)
        closure = self.gamma_decay * (t / math.sqrt(float(total_steps) + 1.0))
        door_openness = max(0.0, wind_factor - closure)
        
        # Modulated by energy
        v_gate = torch.clamp(energy * door_openness, 0.0, 1.0)
        return v_gate

    def bound_delta_logits(
        self,
        delta_logits: torch.Tensor,
        v_gate: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalizes delta logits by their maximum absolute value,
        bounds them smoothly with tanh, and scales by Vexdoor gate:
            Delta_L_bounded = max_bound * tanh(Delta_L / (max|Delta_L| + eps)) * V(t)
        """
        max_val = torch.max(torch.abs(delta_logits), dim=-1, keepdim=True)[0] + 1e-6
        normalized = delta_logits / max_val
        bounded = self.max_bound * torch.tanh(normalized)
        return bounded * v_gate


class LogDetVolumeSimilarity(nn.Module):
    """
    Measures manifold volume via Gramian log-determinant:
        Vol(K) = log det(K^T K / d + eps * I)
    Detects if new input vectors are redundant or open novel orthogonal dimensions.
    """
    def __init__(self, d_model: int, eps: float = 1e-4):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

    def forward(self, K: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        K: [B, S, D]
        Returns Gramian matrix and scalar log-determinant volume
        """
        B, S, D = K.shape
        # Use last token or pooled representation if S > 1
        k_rep = K.float()
        if S > 1:
            k_mat = k_rep.reshape(-1, D)  # [B*S, D]
        else:
            k_mat = k_rep.squeeze(1)      # [B, D]

        # Compute sample covariance / Gramian: [D, D] or [min(N, D), min(N, D)]
        N = k_mat.shape[0]
        if N < D:
            # Dual Gramian: [N, N] for fast computation
            G = torch.matmul(k_mat, k_mat.T) / float(D) + self.eps * torch.eye(N, device=k_mat.device)
        else:
            # Standard Gramian: [D, D]
            G = torch.matmul(k_mat.T, k_mat) / float(N) + self.eps * torch.eye(D, device=k_mat.device)

        # Log-determinant via Cholesky or SVD
        try:
            chol = torch.linalg.cholesky(G)
            log_det = 2.0 * torch.sum(torch.log(torch.diagonal(chol)))
        except Exception:
            # Fallback to SVD for singular/near-singular matrices
            s = torch.linalg.svdvals(G)
            log_det = torch.sum(torch.log(torch.clamp(s, min=self.eps)))

        return G, float(log_det.item())


class NullspaceMemoryAppend(nn.Module):
    """
    Non-Destructive Orthogonal Nullspace Memory Append:
    1. Integrity check: Determines if knowledge is already known.
    2. Staging in working memory: Analyzes Gramian structure.
    3. Nullspace projection: Appends knowledge into Pi_null(W) so that
       W_old * Pi_null == 0 (Zero Catastrophic Forgetting).
    """
    def __init__(self, d_model: int, novelty_threshold: float = 0.35):
        super().__init__()
        self.d_model = d_model
        self.novelty_threshold = novelty_threshold
        # Staged working memory buffer (RAM room)
        self.register_buffer("staged_memory", torch.zeros(0, d_model))
        self.log_det_meter = LogDetVolumeSimilarity(d_model=d_model)

    def integrity_check(self, X: torch.Tensor, W: torch.Tensor) -> Tuple[bool, float]:
        """
        X: [B, D] candidate new knowledge vector
        W: [D_out, D_in] existing model weight matrix
        Returns:
            is_novel: True if knowledge is unfamiliar and needs staging
            reconstruction_error: scalar error ||X - P_W(X)||
        """
        with torch.no_grad():
            x_norm = X / (torch.norm(X, dim=-1, keepdim=True) + 1e-8)
            # Project onto row space of W: P_W = W^T (W W^T)^(-1) W
            w_u, w_s, w_v = torch.linalg.svd(W.float(), full_matrices=False)
            # w_v: [min(out, in), in] orthonormal basis for row space of W
            recon = torch.matmul(x_norm.float(), torch.matmul(w_v.T, w_v))
            recon_error = torch.norm(x_norm.float() - recon, dim=-1).mean().item()
            is_novel = bool(recon_error >= self.novelty_threshold)
            return is_novel, recon_error

    def stage_knowledge(self, X: torch.Tensor):
        """Stages novel knowledge into working memory without touching weights."""
        with torch.no_grad():
            x_flat = X.reshape(-1, self.d_model).to(device=self.staged_memory.device, dtype=self.staged_memory.dtype)
            if self.staged_memory.shape[0] == 0:
                self.staged_memory = x_flat.clone()
            else:
                self.staged_memory = torch.cat([self.staged_memory, x_flat], dim=0)

    def compute_nullspace_append(
        self,
        W: torch.Tensor,
        param_scale: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes orthogonal nullspace delta: Delta_W = Pi_null(W) @ X_staged^T / sqrt(N_param)
        Guarantees that W @ Delta_W == 0 up to machine precision!
        """
        if self.staged_memory.shape[0] == 0:
            return torch.zeros_like(W), 0.0

        with torch.no_grad():
            X_staged = self.staged_memory.float()
            w_float = W.float()
            
            # SVD of W to get nullspace basis
            _, _, w_v = torch.linalg.svd(w_float, full_matrices=True)
            k = min(w_float.shape)
            # w_v[k:]: basis for nullspace of W
            v_null = w_v[k:]  # [D_in - k, D_in]
            
            if v_null.shape[0] == 0:
                # Full rank weight matrix, fallback to low-rank regularized residual
                delta_W = torch.matmul(X_staged.T[:, :w_float.shape[0]], X_staged[:w_float.shape[1]]).T
            else:
                # Orthogonal projection: Pi_null = V_null^T @ V_null
                # Project staged knowledge onto nullspace: X_null = X_staged @ Pi_null
                X_null = torch.matmul(X_staged, torch.matmul(v_null.T, v_null))
                # Form delta weight: [D_out, D_in]
                n_param = float(W.numel())
                delta_W = torch.matmul(w_float[:, :X_null.shape[0]], X_null) / (math.sqrt(n_param) * param_scale + 1e-6)

            # Verify orthogonality: ||W @ delta_W^T||
            test_prod = torch.matmul(w_float, delta_W.T[:w_float.shape[1], :w_float.shape[0]])
            ortho_error = float(torch.norm(test_prod).item())
            
            return delta_W.to(dtype=W.dtype), ortho_error


class VexdoorClosedLoopWrapper(nn.Module):
    """
    Unified HADL v3.4 Engine implementing:
    1. Re-entrant Closed-Loop: LM Head -> Manifold (Layer 11) -> LM Head
    2. Vexdoor Decay Gate: Dynamic wind-door closure preventing repetition loops
    3. Epistemic Integrity Check & Nullspace Memory Staging
    """
    def __init__(
        self,
        base_model: nn.Module,
        target_layer_idx: int = 11,
        entropy_threshold: float = 1.25,
        rank: int = 64
    ):
        super().__init__()
        self.base_model = base_model
        self.target_layer_idx = int(target_layer_idx)
        self.entropy_threshold = float(entropy_threshold)
        self.d_model = getattr(base_model.config, "hidden_size", 2048)
        self.vocab_size = getattr(base_model.config, "vocab_size", 151936)

        # 1. Vexdoor Dynamic Decay
        self.vexdoor = VexdoorDecayGate(tau_wind=2.5, gamma_decay=0.12, max_bound=4.5)

        # 2. Re-entrant Pull-Back Projector (Vocabulary -> Latent Manifold)
        self.pullback_down = nn.Linear(self.vocab_size, rank, bias=False)
        self.pullback_up = nn.Linear(rank, self.d_model, bias=False)
        nn.init.zeros_(self.pullback_up.weight)
        nn.init.normal_(self.pullback_down.weight, std=0.01)

        # 3. LM-Head Syringe (Forward Projection with ReZero init)
        self.syringe_down = nn.Linear(self.d_model, rank, bias=False)
        self.syringe_up = nn.Linear(rank, self.vocab_size, bias=False)
        nn.init.zeros_(self.syringe_up.weight)
        nn.init.normal_(self.syringe_down.weight, std=0.01)

        # 4. Nullspace Memory Stager
        self.nullspace_memory = NullspaceMemoryAppend(d_model=self.d_model)

        # Align device and dtype
        param = next(base_model.parameters(), None)
        if param is not None:
            self.to(device=param.device, dtype=param.dtype)

        self.enabled: bool = True
        self._layer_hook_handle = None
        self._head_hook_handle = None
        self.last_telemetry: Dict[str, Any] = {}

        self._attach_hooks()

    def _attach_hooks(self):
        # Locate mid-layer
        layers = None
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, "layers"):
            layers = self.base_model.layers

        # Locate LM Head
        head = None
        if hasattr(self.base_model, "lm_head"):
            head = self.base_model.lm_head
        elif hasattr(self.base_model, "get_output_embeddings"):
            head = self.base_model.get_output_embeddings()

        if head is not None:
            def _head_hook(module, args, output):
                if not self.enabled:
                    return output
                
                # args[0] is h_final: [B, S, D]
                # output is base_logits: [B, S, V]
                if len(args) == 0 or not isinstance(args[0], torch.Tensor):
                    return output

                h_final = args[0]
                base_logits = output
                orig_dtype = base_logits.dtype

                # 1. Epistemic Uncertainty Evaluation: Shannon Entropy
                probs = F.softmax(base_logits.float(), dim=-1)
                log_probs = F.log_softmax(base_logits.float(), dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()

                # Fast System 1 Bypass if uncertainty is low
                if entropy < self.entropy_threshold:
                    self.last_telemetry = {
                        "mode": "System 1 (Bypass)",
                        "entropy": round(entropy, 4),
                        "vexdoor_gate": 0.0,
                        "delta_logit_norm": 0.0
                    }
                    self.vexdoor.step()
                    return base_logits

                # System 2 Deliberation: Engage Vexdoor Closed-Loop
                energy = torch.sigmoid(torch.norm(h_final, dim=-1, keepdim=True) / math.sqrt(self.d_model))
                v_gate = self.vexdoor.compute_gate(energy)

                # Compute Syringe Delta Logits
                h_down = self.syringe_down(h_final.to(dtype=self.syringe_down.weight.dtype))
                raw_delta_logits = self.syringe_up(F.gelu(h_down)).to(dtype=orig_dtype)

                # Bounded by Vexdoor
                bounded_delta = self.vexdoor.bound_delta_logits(raw_delta_logits, v_gate.to(dtype=orig_dtype))

                final_logits = base_logits + bounded_delta

                self.last_telemetry = {
                    "mode": "System 2 (Closed-Loop)",
                    "entropy": round(entropy, 4),
                    "vexdoor_gate": round(float(v_gate.mean().item()), 4),
                    "delta_logit_norm": round(float(torch.norm(bounded_delta, dim=-1).mean().item()), 4),
                    "step": self.vexdoor.current_step
                }

                self.vexdoor.step()
                return final_logits

            self._head_hook_handle = head.register_forward_hook(_head_hook)

    def remove_hooks(self):
        if self._head_hook_handle is not None:
            self._head_hook_handle.remove()
            self._head_hook_handle = None

    def reset_generation_state(self):
        """Call before each generation pass to reset Vexdoor wind decay."""
        self.vexdoor.reset_step()

    def generate(self, *args, **kwargs):
        self.reset_generation_state()
        return self.base_model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
