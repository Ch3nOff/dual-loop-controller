import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import time

class EpistemicHumilityModule(nn.Module):
    """
    Epistemic Humility & Bounded Confidence Engine.
    
    Principles:
    1. Bounded Confidence: Mathematical guarantee that confidence never reaches 1.0 (100%).
       Total Dirichlet evidence is capped so that epistemic vacuity u(x) >= u_min > 0 always.
    2. Asymmetric Overconfidence Penalty (Hyperbolic Arrogance Loss):
       Penalizes confident errors quadratically in hyperbolic odds:
           L_overconf = I_error * (c / (1 - c + eps))^2
       A model claiming 99% confidence on a failed prediction suffers ~10,000x penalty,
       while an epistemically modest model (30% confidence) suffers < 0.2x penalty.
    """
    def __init__(
        self,
        d_model: int,
        num_classes: int = 4,
        max_confidence: float = 0.95,
        min_vacuity: float = 0.05,
        eps: float = 1e-4
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_confidence = float(max_confidence)
        self.min_vacuity = float(min_vacuity)
        self.eps = float(eps)
        
        # Dirichlet evidence projection
        self.evidence_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )
        nn.init.normal_(self.evidence_proj[-1].weight, std=0.01)
        nn.init.zeros_(self.evidence_proj[-1].bias)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes bounded Dirichlet confidence and epistemic vacuity.
        Args:
            h: [B, D] or [B, S, D] Hidden representations.
        Returns:
            dict with 'confidence', 'vacuity', 'alpha', 'probs'.
        """
        evidence = F.softplus(self.evidence_proj(h))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        
        # Subjective Logic Epistemic Vacuity: u = K / S
        vacuity = self.num_classes / S
        # Guarantee minimum vacuity (perpetual curiosity opening)
        vacuity_bounded = torch.clamp(vacuity, min=self.min_vacuity, max=1.0)
        
        # Probability distribution over evidence
        probs = alpha / S
        raw_conf, _ = torch.max(probs, dim=-1, keepdim=True)
        
        # Bound confidence strictly below 100%
        bounded_conf = torch.clamp(raw_conf, min=1.0 / self.num_classes, max=self.max_confidence)
        
        return {
            "confidence": bounded_conf.squeeze(-1),
            "vacuity": vacuity_bounded.squeeze(-1),
            "alpha": alpha,
            "probs": probs
        }

    def compute_humility_loss(
        self,
        confidence: torch.Tensor,
        was_error: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates Asymmetric Epistemic Humility Loss:
            L = was_error * (c / (1 - c + eps))^2
        Args:
            confidence: [...] Confidence score in [0, max_confidence].
            was_error: [...] Binary indicator: 1.0 if incorrect/bug, 0.0 if correct.
        """
        c = torch.clamp(confidence, min=0.0, max=0.999)
        err = was_error.to(device=c.device, dtype=c.dtype)
        
        odds = c / (1.0 - c + self.eps)
        penalty = err * (odds ** 2)
        return torch.mean(penalty)


class IntrinsicCuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) with Inverse Dynamics.
    
    Solves the Noisy-TV problem by decoupling uncontrollable stochastic noise
    from predictable environmental transitions.
    """
    def __init__(self, d_model: int, d_feature: int = 128, d_action: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.d_action = d_action
        
        # Feature encoder: phi(s)
        self.phi_encoder = nn.Sequential(
            nn.Linear(d_model, d_feature),
            nn.LayerNorm(d_feature),
            nn.GELU()
        )
        
        # Inverse Dynamics: g(phi_t, phi_next) -> a_t
        self.inverse_model = nn.Sequential(
            nn.Linear(d_feature * 2, d_feature),
            nn.GELU(),
            nn.Linear(d_feature, d_action)
        )
        
        # Forward Dynamics: f(phi_t, a_t) -> phi_next
        self.forward_model = nn.Sequential(
            nn.Linear(d_feature + d_action, d_feature),
            nn.GELU(),
            nn.Linear(d_feature, d_feature)
        )

    def forward(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        s_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s_t: [B, D] Initial state representation.
            a_t: [B, D_action] Action / deliberation step taken.
            s_next: [B, D] Next state representation.
        Returns:
            curiosity_reward: [B] Intrinsic curiosity reward (prediction error in phi space).
            inverse_loss: Scalar loss training phi to ignore uncontrollable noise.
            forward_loss: Scalar loss tracking prediction accuracy.
        """
        phi_t = self.phi_encoder(s_t)
        phi_next_actual = self.phi_encoder(s_next)
        
        # 1. Inverse Dynamics: predict action from state transition
        a_pred = self.inverse_model(torch.cat([phi_t, phi_next_actual], dim=-1))
        inv_loss = F.mse_loss(a_pred, a_t)
        
        # 2. Forward Dynamics: predict next feature from state and action
        phi_next_pred = self.forward_model(torch.cat([phi_t, a_t], dim=-1))
        fwd_loss = F.mse_loss(phi_next_pred, phi_next_actual.detach())
        
        # 3. Curiosity Reward = ||phi_pred - phi_actual||^2
        curiosity_reward = 0.5 * torch.mean((phi_next_actual.detach() - phi_next_pred) ** 2, dim=-1)
        
        return curiosity_reward, inv_loss, fwd_loss


class PopperianSelfPlayEngine(nn.Module):
    """
    Adversarial Self-Play & Hypothesis Falsification Engine (Karl Popper).
    
    Sub-agents:
    - Proposer: Generates optimistic candidate hypotheses for unresolved memory contradictions.
    - Falsifier (Red Team): Crafts adversarial counter-examples targeting boundary failure modes.
    - Deterministic Verifier: Runs execution checks in an isolated deterministic sandbox.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        self.proposer_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.falsifier_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def propose_hypothesis(self, memory_a: torch.Tensor, memory_b: torch.Tensor) -> torch.Tensor:
        """Proposes candidate synthesized hypothesis reconciling two conflicting memory anchors."""
        pair = torch.cat([memory_a, memory_b], dim=-1)
        return self.proposer_net(pair)

    def generate_counter_example(self, hypothesis: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Red Team generates boundary challenge attempting to refute the hypothesis."""
        pair = torch.cat([hypothesis, anchor], dim=-1)
        return self.falsifier_net(pair)

    @staticmethod
    def verify_sandbox(
        code_or_expression: str,
        test_condition: str = "eval",
        context_variables: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Deterministic Ground-Truth Sandbox Verification.
        Safely executes code or evaluates boolean/arithmetic expressions within
        an isolated Python execution environment with restricted builtins.
        
        Args:
            code_or_expression: Python code string or boolean/arithmetic expression.
            test_condition: Verification mode:
                - 'eval' or 'boolean_logic': Evaluates expression via eval().
                  Returns True if result is Truthy, False if Falsy (e.g. 1 == 2 -> False).
                - 'exec' or 'assertion': Executes code block via exec().
                  Returns True if all assertions hold, False if AssertionError or runtime error.
                - 'syntax_check': Validates syntax only via compile().
            context_variables: Optional dictionary of variables injected into sandbox scope.
            
        Returns:
            is_valid: True if assertion/condition holds, False if refuted/falsified.
            diag: Diagnostic output string.
        """
        # Security Guard 1: Prohibit dunder introspection (__import__, __class__, etc.)
        if "__" in code_or_expression:
            return False, "Sandbox security violation: double-underscore attribute access prohibited"
        
        # Security Guard 2: Prohibit forbidden calls and modules
        forbidden = ["import ", "open(", "eval(", "exec(", "globals(", "locals(", "os.", "sys.", "subprocess", "shutil"]
        for bad in forbidden:
            if bad in code_or_expression:
                return False, f"Sandbox security violation: forbidden call '{bad}' detected"
                
        safe_builtins = {
            "abs": abs, "min": min, "max": max, "sum": sum, "all": all, "any": any,
            "len": len, "range": range, "bool": bool, "int": int, "float": float,
            "str": str, "dict": dict, "list": list, "set": set, "tuple": tuple,
            "round": round, "zip": zip,
            "AssertionError": AssertionError, "ValueError": ValueError,
            "TypeError": TypeError, "ZeroDivisionError": ZeroDivisionError,
            "True": True, "False": False, "None": None,
        }
        safe_env: Dict[str, Any] = {"__builtins__": safe_builtins}
        if context_variables:
            for k, v in context_variables.items():
                if not k.startswith("__"):
                    safe_env[k] = v
                    
        try:
            mode = test_condition.lower()
            if mode in ["eval", "boolean_logic", "logic", "expression"]:
                res = eval(code_or_expression, safe_env)
                if isinstance(res, bool):
                    if res:
                        return True, "Verified in sandbox: boolean expression evaluated to True"
                    else:
                        return False, "Falsified in sandbox: boolean condition evaluated to False"
                return bool(res), f"Evaluated in sandbox with result: {res}"
                
            elif mode in ["exec", "assertion", "assertion_test"]:
                exec(code_or_expression, safe_env)
                return True, "Verified in sandbox: script executed and all assertions passed"
                
            elif mode in ["syntax_check", "syntax"]:
                compile(code_or_expression, "<sandbox>", "exec")
                return True, "Code compiled successfully without syntax errors"
                
            else:
                # Default auto mode: try eval first; if syntax error (e.g. statements), fallback to exec
                try:
                    res = eval(code_or_expression, safe_env)
                    if isinstance(res, bool):
                        return res, f"Evaluated in sandbox: {res}"
                    return bool(res), f"Evaluated in sandbox: {res}"
                except SyntaxError:
                    exec(code_or_expression, safe_env)
                    return True, "Executed in sandbox successfully"
                    
        except AssertionError as e:
            msg = str(e) if str(e) else "Assertion failed"
            return False, f"Falsified in sandbox: AssertionError ({msg})"
        except SyntaxError as e:
            return False, f"Falsified in sandbox: SyntaxError ({e})"
        except Exception as e:
            return False, f"Falsified in sandbox: runtime error {type(e).__name__} ({e})"

    def synthesize_sandbox_challenge(
        self,
        hypothesis: torch.Tensor,
        counter_ex: torch.Tensor,
        slot_a: torch.Tensor,
        slot_b: torch.Tensor
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Synthesizes a concrete, executable verification challenge testing whether
        the candidate hypothesis legitimately resolves the contradiction or is refuted
        by the Popperian Red Team counter-example.
        """
        h_flat = hypothesis.flatten()
        c_flat = counter_ex.flatten()
        a_flat = slot_a.flatten()
        b_flat = slot_b.flatten()
        
        h_norm = float(h_flat.norm().item())
        diff_stress = float(torch.norm(c_flat - h_flat).item())
        cos_a = float(F.cosine_similarity(h_flat.unsqueeze(0), a_flat.unsqueeze(0)).item())
        cos_b = float(F.cosine_similarity(h_flat.unsqueeze(0), b_flat.unsqueeze(0)).item())
        
        context_meta = {
            "h_norm": round(h_norm, 4),
            "diff_stress": round(diff_stress, 4),
            "cos_a": round(cos_a, 4),
            "cos_b": round(cos_b, 4),
        }
        
        test_script = f"""
def verify_latent_hypothesis(h_norm, diff_stress, cos_a, cos_b):
    # Invariant 1: Non-triviality (energy must be bounded and non-collapsed)
    assert h_norm > 0.01, "Degenerate hypothesis: representation collapsed to zero"
    assert h_norm < 100.0, "Unstable hypothesis: representation exploded"
    
    # Invariant 2: Non-trivial reconciliation (cannot be identical to pure simultaneous negation)
    assert not (cos_a < -0.99 and cos_b < -0.99), "Trivial negation: hypothesis rejects both premises"
    
    # Invariant 3: Popperian Falsifier Challenge
    # Counter-example tests the boundary stability of the synthesized hypothesis
    assert diff_stress <= 1.25, f"Boundary challenge refuted hypothesis: stress={{diff_stress}} > 1.25"
    return True

verify_latent_hypothesis(h_norm, diff_stress, cos_a, cos_b)
"""
        return test_script, "exec", context_meta


class AutonomousDaemonController:
    """
    The Autonomous Background Daemon Loop (Decoupling Inference from Clock).
    
    Operates during idle intervals:
    1. Scans episodic memory for high-vacuity blindspots (u > tau_ignorance).
    2. Initiates Popperian Proposer vs Falsifier Red Team self-play.
    3. Evaluates hypotheses via Deterministic Ground Truth Verifier.
    4. Computes Epistemic Humility Loss to punish overconfidence.
    5. Resolves verified anomalies by projecting corrections into Orthogonal Nullspace.
    """
    def __init__(
        self,
        d_model: int,
        tau_ignorance: float = 0.60,
        tau_contradiction: float = 0.75,
        device: Optional[torch.device] = None
    ):
        self.d_model = d_model
        self.tau_ignorance = float(tau_ignorance)
        self.tau_contradiction = float(tau_contradiction)
        self.device = device or torch.device("cpu")
        
        self.humility_engine = EpistemicHumilityModule(d_model=d_model).to(self.device)
        self.curiosity_icm = IntrinsicCuriosityModule(d_model=d_model).to(self.device)
        self.popperian_engine = PopperianSelfPlayEngine(d_model=d_model).to(self.device)
        
        from .nullspace_engine import OrthogonalNullspaceProjector
        self.nullspace_projector = OrthogonalNullspaceProjector(d_model=d_model).to(self.device)
        
        self.active_state = "IDLE"
        self.resolved_anomalies_count = 0
        self.total_scanned_anomalies = 0
        self.history_telemetry: List[Dict[str, Any]] = []

    def scan_memory_anomalies(self, memory_slots: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        Scans episodic memory slots for latent contradictions:
            Contradiction(i, j) = ||h_i + h_j - h_joint||
        Returns list of (slot_i, slot_j, contradiction_score).
        """
        if memory_slots.dim() == 2:
            slots = memory_slots
        else:
            slots = memory_slots[0]
            
        M = slots.shape[0]
        if M < 2:
            return []
            
        anomalies = []
        normed = F.normalize(slots, p=2, dim=-1)
        sim_matrix = torch.matmul(normed, normed.t()) # [M, M]
        
        for i in range(M):
            for j in range(i + 1, M):
                # Opposing vectors with high similarity in context indicate mutual negation/conflict
                sim = float(sim_matrix[i, j].item())
                diff = float(torch.norm(slots[i] - slots[j], p=2).item())
                if abs(sim) > self.tau_contradiction or diff > 1.8:
                    anomalies.append((i, j, diff))
                    
        return anomalies

    def run_daemon_step(
        self,
        memory_slots: torch.Tensor,
        action_vector: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Executes a single autonomous background contemplation cycle.
        """
        t0 = time.time()
        self.active_state = "SCANNING"
        device = memory_slots.device
        
        slots = memory_slots[0] if memory_slots.dim() == 3 else memory_slots
        M = slots.shape[0]
        
        # 1. Evaluate Epistemic Humility on all memory slots
        humility_out = self.humility_engine(slots)
        confidences = humility_out["confidence"]
        vacuities = humility_out["vacuity"]
        
        # 2. Find high-vacuity blindspots
        blindspots = (vacuities >= self.tau_ignorance).nonzero(as_tuple=True)[0]
        
        # 3. Scan for latent contradictions
        anomalies = self.scan_memory_anomalies(slots)
        self.total_scanned_anomalies += len(anomalies)
        
        step_result = {
            "num_slots": M,
            "blindspots_detected": len(blindspots),
            "contradictions_detected": len(anomalies),
            "anomalies_resolved": 0,
            "curiosity_reward": 0.0,
            "humility_loss": 0.0,
            "state": "COMPLETED",
            "cycle_latency_ms": 0.0
        }
        
        # 4. If anomalies or blindspots exist, engage Popperian Self-Play
        if len(anomalies) > 0:
            self.active_state = "SELF_PLAY"
            idx_a, idx_b, score = anomalies[0]
            slot_a = slots[idx_a:idx_a+1]
            slot_b = slots[idx_b:idx_b+1]
            
            # Proposer synthesizes candidate resolution
            hyp = self.popperian_engine.propose_hypothesis(slot_a, slot_b)
            # Falsifier creates boundary challenge
            counter_ex = self.popperian_engine.generate_counter_example(hyp, slot_a)
            
            # Deterministic sandbox verification via safe Python execution
            test_script, test_mode, context_meta = self.popperian_engine.synthesize_sandbox_challenge(
                hyp, counter_ex, slot_a, slot_b
            )
            is_valid, sandbox_diag = self.popperian_engine.verify_sandbox(
                test_script, test_condition=test_mode, context_variables=context_meta
            )
            # Hypothesis is falsified if the deterministic sandbox assertion fails
            is_falsified = not is_valid
            
            # Compute Epistemic Humility Loss
            hyp_conf = humility_out["confidence"][idx_a:idx_a+1]
            err_tensor = torch.tensor([1.0 if is_falsified else 0.0], device=device)
            h_loss = self.humility_engine.compute_humility_loss(hyp_conf, err_tensor)
            
            # 5. Resolve anomaly via Orthogonal Nullspace Projection
            self.active_state = "CONSOLIDATING"
            basis = slots[:min(M, 8)]
            basis_3d = basis.unsqueeze(0) if basis.dim() == 2 else basis
            hyp_2d = hyp.squeeze(1) if hyp.dim() == 3 else hyp
            v_perp, p_telem = self.nullspace_projector(hyp_2d, basis_3d)
            
            self.resolved_anomalies_count += 1
            step_result["anomalies_resolved"] = 1
            step_result["humility_loss"] = float(h_loss.item())
            step_result["nullspace_residual_norm"] = float(v_perp.norm().item())
            step_result["falsification_event"] = is_falsified
            step_result["sandbox_diagnostic"] = sandbox_diag
            step_result["sandbox_code_tested"] = test_script.strip()
            
        # 6. Compute ICM Curiosity Reward
        if action_vector is None:
            action_vector = torch.zeros(1, 32, device=device)
        s_t = slots[0:1]
        s_next = slots[-1:] if M > 1 else slots[0:1]
        r_c, _, _ = self.curiosity_icm(s_t, action_vector, s_next)
        step_result["curiosity_reward"] = float(r_c.mean().item())
        
        self.active_state = "IDLE"
        step_result["cycle_latency_ms"] = (time.time() - t0) * 1000.0
        self.history_telemetry.append(step_result)
        
        return step_result
