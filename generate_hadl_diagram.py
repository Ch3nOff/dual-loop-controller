import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil

def create_hadl_diagram():
    # High resolution 16:9 canvas
    fig = plt.figure(figsize=(26, 15), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Dark modern AI theme
    bg_color = "#070b14"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Color Palette
    c_card_bg = "#0f172a"
    c_card_border = "#334155"
    c_text_bright = "#f8fafc"
    c_text_muted = "#94a3b8"
    c_accent_cyan = "#38bdf8"
    c_accent_emerald = "#10b981"
    c_accent_violet = "#a855f7"
    c_accent_amber = "#f59e0b"
    c_accent_rose = "#f43f5e"

    def draw_card(x, y, w, h, bg="#0f172a", border="#334155", lw=1.8, radius=0.8):
        patch = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={radius},rounding_size=1.0",
            facecolor=bg,
            edgecolor=border,
            linewidth=lw,
            zorder=2
        )
        ax.add_patch(patch)
        return patch

    # -------------------------------------------------------------
    # 0. HEADER BANNER
    # -------------------------------------------------------------
    draw_card(2, 92, 96, 6.5, bg="#0b1329", border="#1e293b", lw=2.2)
    ax.text(50, 96.6, "HOMEOSTATIC AUTOPOIETIC DUAL-LOOP (HADL) ARCHITECTURE", 
            ha="center", va="center", fontsize=20, fontweight="bold", color=c_accent_cyan, zorder=3)
    ax.text(50, 93.6, "Biological Drive-Reduction • Active Inference Routing • Orthogonal Nullspace Synthesis • Category Theory Functors • The Brain Sandbox", 
            ha="center", va="center", fontsize=10.5, color=c_text_muted, zorder=3)

    # -------------------------------------------------------------
    # 1. LEFT COLUMN: TRANSFORMER BACKBONE & INTERCEPTION HOOK
    # -------------------------------------------------------------
    draw_card(2.5, 6, 22, 83, bg="#090d1a", border="#1e293b", lw=2.0)
    ax.text(13.5, 87.0, "TRANSFORMER BACKBONE", ha="center", va="center", fontsize=13, fontweight="bold", color=c_accent_cyan, zorder=3)
    ax.text(13.5, 84.8, "(Qwen / GLM-4 Layer Hierarchy)", ha="center", va="center", fontsize=9, color=c_text_muted, zorder=3)

    # Input Tokens
    draw_card(4.5, 75, 18, 7, bg="#131e36", border="#334155")
    ax.text(13.5, 79.5, "Input Sequence x_t", ha="center", va="center", fontsize=10, fontweight="bold", color=c_text_bright, zorder=3)
    ax.text(13.5, 76.8, "Tokens [x_1, ..., x_t]\nPrompt or Streaming Token", ha="center", va="center", fontsize=7.8, color=c_text_muted, zorder=3)

    # Layers 0..10
    draw_card(4.5, 63, 18, 8, bg="#131e36", border="#334155")
    ax.text(13.5, 68.0, "Layers 0 .. 10", ha="center", va="center", fontsize=10, fontweight="bold", color=c_text_bright, zorder=3)
    ax.text(13.5, 65.2, "11x Transformer Blocks\nSelf-Attention + SwiGLU / MLP", ha="center", va="center", fontsize=7.8, color=c_text_muted, zorder=3)

    # LAYER 11 HOOK (CRUCIAL NODE)
    draw_card(4.0, 46, 19, 13, bg="#1e1b4b", border="#818cf8", lw=2.5)
    ax.text(13.5, 56.5, "LAYER 11 FORWARD HOOK", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#c7d2fe", zorder=3)
    ax.text(13.5, 53.2, "Intercepts Hidden States\nh_11 in R^[B, S, D] (D=2048)", ha="center", va="center", fontsize=8.2, color=c_text_bright, zorder=3)
    ax.text(13.5, 49.5, "Residual Injection Point:\nh_11' = h_11 + Delta_h", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#fbbf24", zorder=3)

    # Layers 12..27
    draw_card(4.5, 30, 18, 12, bg="#131e36", border="#334155")
    ax.text(13.5, 37.5, "Layers 12 .. 27", ha="center", va="center", fontsize=10, fontweight="bold", color=c_text_bright, zorder=3)
    ax.text(13.5, 33.5, "16x Transformer Blocks\nPropagates enhanced latent\nrepresentations toward output", ha="center", va="center", fontsize=7.8, color=c_text_muted, zorder=3)

    # LM Head & Un-embedding
    draw_card(4.5, 17, 18, 9, bg="#131e36", border="#334155")
    ax.text(13.5, 22.5, "LM Head Un-embedding", ha="center", va="center", fontsize=10, fontweight="bold", color=c_text_bright, zorder=3)
    ax.text(13.5, 19.5, "RMSNorm + W_vocab\nP(x_{t+1}) = Softmax(z_t / T)", ha="center", va="center", fontsize=7.8, color=c_text_muted, zorder=3)

    # Final Output Token
    draw_card(4.5, 8, 18, 6, bg="#064e3b", border="#10b981", lw=2.0)
    ax.text(13.5, 11.8, "Generated Token x_{t+1}", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#6ee7b7", zorder=3)
    ax.text(13.5, 9.5, "Fluent, Syntactically Deterministic", ha="center", va="center", fontsize=7.5, color="#a7f3d0", zorder=3)

    # Backbone Flow Arrows
    arrow_down = dict(arrowstyle="-|>", color="#64748b", lw=2.0, mutation_scale=14)
    ax.annotate("", xy=(13.5, 71.5), xytext=(13.5, 74.5), arrowprops=arrow_down, zorder=4)
    ax.annotate("", xy=(13.5, 59.5), xytext=(13.5, 62.5), arrowprops=arrow_down, zorder=4)
    ax.annotate("", xy=(13.5, 42.5), xytext=(13.5, 45.5), arrowprops=arrow_down, zorder=4)
    ax.annotate("", xy=(13.5, 26.5), xytext=(13.5, 29.5), arrowprops=arrow_down, zorder=4)
    ax.annotate("", xy=(13.5, 14.5), xytext=(13.5, 16.5), arrowprops=arrow_down, zorder=4)

    # -------------------------------------------------------------
    # 2. MIDDLE COLUMN (TOP): HOMEOSTATIC DRIVE & ACTIVE INFERENCE
    # -------------------------------------------------------------
    draw_card(27.5, 61, 36, 28, bg="#022c22", border=c_accent_emerald, lw=2.2)
    ax.text(45.5, 86.5, "1. BIOLOGICAL HOMEOSTASIS & ACTIVE INFERENCE", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#6ee7b7", zorder=3)

    # Physiological State Vector S_t
    draw_card(29.0, 71.0, 33, 13.5, bg="#064e3b", border="#059669")
    ax.text(45.5, 82.5, "Physiological State Vector S_t in R^4", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#a7f3d0", zorder=3)
    ax.text(30.5, 79.5, "• S_1: Energy Budget (depletes on k>=1, recovers on k=0)", fontsize=7.6, color=c_text_bright, zorder=3)
    ax.text(30.5, 77.0, "• S_2: Epistemic Entropy u(x) in [0, 1] (Dirichlet vacuity)", fontsize=7.6, color=c_text_bright, zorder=3)
    ax.text(30.5, 74.5, "• S_3: Semantic Drift ||h_t - h_anchor|| / ||h_anchor||", fontsize=7.6, color=c_text_bright, zorder=3)
    ax.text(30.5, 72.0, "• S_4: Working Memory Saturation Ratio M_occ / M_total", fontsize=7.6, color=c_text_bright, zorder=3)

    # Setpoint & Drive Reduction Formula
    draw_card(29.0, 62.5, 33, 7.5, bg="#064e3b", border="#059669")
    ax.text(45.5, 68.0, "Setpoint S* = [1.0, 0.05, 0.0, 0.25]^T  |  D(S_t) = sum w_i |S_i - S*_i|^2", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#fbbf24", zorder=3)
    ax.text(45.5, 64.5, "Drive Reduction: Delta D = D(S_t) - D(S_{t+1}) > 0", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#6ee7b7", zorder=3)

    # Interception Arrow (Hook -> Homeostasis)
    ax.annotate("", xy=(27.0, 73), xytext=(23.5, 54),
                arrowprops=dict(arrowstyle="-|>", color=c_accent_cyan, lw=2.5, mutation_scale=18, connectionstyle="arc3,rad=-0.15"), zorder=4)
    ax.text(24.5, 66.5, "h_11 State\nInterception", color=c_accent_cyan, fontsize=8.2, fontweight="bold", ha="center", zorder=5)

    # -------------------------------------------------------------
    # 2.5 CLEAR HORIZONTAL CORRIDOR: SOLUTION VECTOR RETURN
    # -------------------------------------------------------------
    draw_card(27.5, 53.5, 36, 6.0, bg="#1e1808", border="#f59e0b", lw=2.0)
    ax.text(45.5, 56.5, "<== CALIBRATED SOLUTION VECTOR Delta_h (Injected to Prompt) <==", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#fbbf24", zorder=3)
    
    # Horizontal conduit arrow to Hook
    ax.annotate("", xy=(23.5, 52.5), xytext=(27.5, 56.5),
                arrowprops=dict(arrowstyle="-|>", color="#fbbf24", lw=2.5, mutation_scale=18), zorder=4)

    # -------------------------------------------------------------
    # 3. MIDDLE COLUMN (MIDDLE): PATHWAY A (SYSTEM 1 FAST BYPASS)
    # -------------------------------------------------------------
    draw_card(27.5, 32.5, 36, 19.5, bg="#022c22", border="#34d399", lw=2.0)
    ax.text(45.5, 49.5, "PATHWAY A: SYSTEM 1 FAST BYPASS (K=0)", ha="center", va="center", fontsize=11, fontweight="bold", color="#34d399", zorder=3)
    ax.text(45.5, 46.0, "Trigger: is_token_streaming (S=1) & u < 0.85\n(Active for syntax tokens: ';', '{', '}', 'div', operators)", ha="center", va="center", fontsize=8.0, color=c_text_bright, zorder=3)
    ax.text(45.5, 41.5, "Policy pi_0: k* = 0 (Exact Identity Bypass)\nDelta_h = 0 | Energy Recovers | 0ms CPU Cost", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#6ee7b7", zorder=3)
    ax.text(45.5, 36.5, "Guarantees: ~73s Native Speed & 0% Syntax Corruption", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#fbbf24", zorder=3)

    # Bypass Return Arrow
    ax.annotate("", xy=(23.5, 49.5), xytext=(27.5, 42.5),
                arrowprops=dict(arrowstyle="-|>", color="#34d399", lw=2.5, mutation_scale=18), zorder=4)
    ax.text(25.0, 43.5, "k=0\nReturn", color="#34d399", fontsize=8, fontweight="bold", ha="center", zorder=5)

    # -------------------------------------------------------------
    # 4. MIDDLE COLUMN (BOTTOM): MATHEMATICAL FOUNDATIONS
    # -------------------------------------------------------------
    draw_card(27.5, 6, 36, 25, bg="#0b1329", border="#3b82f6", lw=1.8)
    ax.text(45.5, 28.5, "MATHEMATICAL FORMULATIONS", ha="center", va="center", fontsize=11, fontweight="bold", color=c_accent_cyan, zorder=3)

    math_text = (
        "1. Active Inference Free Energy:\n"
        "   G(pi) = - E_Q[ln P(o_tau|C)] - E_Q[ln Q(s|o,pi) - ln Q(s|pi)]\n\n"
        "2. Orthogonal Nullspace Projection (Gram-Schmidt):\n"
        "   Q, R = qr(V_{known}),   P_perp = I - Q Q^T\n"
        "   h_{novel} = P_perp Delta_c  ==>  <h_{novel}, v> = 0\n\n"
        "3. Functorial Category Mapping:\n"
        "   F: C -> D  satisfies  F(f_1 o f_2) = F(f_1) o F(f_2)\n\n"
        "4. Neuro-Symbolic MDL Selection (Occam's Razor):\n"
        "   Delta MDL = Length(p*) + lambda * Error(Constraints | p*)"
    )
    ax.text(29.0, 16.5, math_text, ha="left", va="center", fontsize=7.6, color="#e2e8f0", fontfamily="monospace", zorder=3)

    # -------------------------------------------------------------
    # 5. RIGHT COLUMN: PATHWAY B (THE BRAIN SANDBOX)
    # -------------------------------------------------------------
    draw_card(66.5, 6, 31.5, 83, bg="#181024", border=c_accent_amber, lw=2.2)
    ax.text(82.2, 87.0, "PATHWAY B: THE BRAIN SANDBOX (K=3)", ha="center", va="center", fontsize=12, fontweight="bold", color="#fb923c", zorder=3)
    ax.text(82.2, 84.8, "Triggered on Prompt (S > 1) & High Uncertainty (u >= 0.65)", ha="center", va="center", fontsize=8.0, color=c_text_muted, zorder=3)

    # STAGE 1: INTENT & DOMAIN CLASSIFIER
    draw_card(68.0, 68, 28.5, 14, bg="#271838", border="#f59e0b", lw=1.6)
    ax.text(82.2, 79.5, "STAGE 1: INTENT CLASSIFIER", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#fcd34d", zorder=3)
    ax.text(82.2, 75.5, "Evaluates query embedding h_anchor\nSoftmax(W_domain h) -> [Normal QA vs Code/Script]", ha="center", va="center", fontsize=7.6, color=c_text_bright, zorder=3)
    ax.text(82.2, 71.0, "Determines execution requirements & domain rules", ha="center", va="center", fontsize=7.4, color=c_text_muted, zorder=3)

    # STAGE 2: PURPOSE & CWM GROUNDING
    draw_card(68.0, 50, 28.5, 15, bg="#1e2952", border="#3b82f6", lw=1.6)
    ax.text(82.2, 62.5, "STAGE 2: PURPOSE & CWM GROUNDING", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#93c5fd", zorder=3)
    ax.text(82.2, 58.5, "Cognitive Working Memory Compressor (CWM):\nCompresses context into M=16 slots in R^d_inner", ha="center", va="center", fontsize=7.6, color=c_text_bright, zorder=3)
    ax.text(82.2, 53.5, "Functorial Mapper: Maps relational morphisms\nacross categories: F(M_source) -> M_target", ha="center", va="center", fontsize=7.4, color="#bfdbfe", zorder=3)

    # STAGE 3: LATENT DRAFT ROLLOUT
    draw_card(68.0, 32, 28.5, 15, bg="#33104a", border="#a855f7", lw=1.6)
    ax.text(82.2, 44.5, "STAGE 3: LATENT DRAFT ROLLOUT", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#c084fc", zorder=3)
    ax.text(82.2, 40.5, "Recurrent Latent Controller: K=3 ponder steps\nH_{k+1} = TransformerBlock(H_k, CWM)", ha="center", va="center", fontsize=7.6, color=c_text_bright, zorder=3)
    ax.text(82.2, 35.5, "Orthogonal Nullspace: If u >= tau_unseen:\nh_novel = (I - Q Q^T) Delta_c ⊥ V_known", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#e9d5ff", zorder=3)

    # STAGE 4: STRESS-TEST & MDL SELECTION
    draw_card(68.0, 10, 28.5, 19, bg="#440f2b", border="#f43f5e", lw=1.6)
    ax.text(82.2, 26.5, "STAGE 4: STRESS-TEST & MDL", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#f472b6", zorder=3)
    ax.text(82.2, 22.5, "Latent Critique Unit: Computes discrepancy\ne_k = LN(Thoughts - Grounded Constraints)", ha="center", va="center", fontsize=7.6, color=c_text_bright, zorder=3)
    ax.text(82.2, 17.5, "MDL Selection: argmin [Length(p*) + lambda*Error]\nRejects bloated plans; selects most parsimonious code", ha="center", va="center", fontsize=7.4, color=c_text_bright, zorder=3)
    ax.text(82.2, 13.0, "Result: Syntactically sound latent blueprint", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#fbcfe8", zorder=3)

    # Sandbox Internal Arrows
    ax.annotate("", xy=(82.2, 65.5), xytext=(82.2, 67.5), arrowprops=arrow_down, zorder=4)
    ax.annotate("", xy=(82.2, 47.5), xytext=(82.2, 49.5), arrowprops=arrow_down, zorder=4)
    ax.annotate("", xy=(82.2, 29.5), xytext=(82.2, 31.5), arrowprops=arrow_down, zorder=4)

    # Trigger Arrow (Homeostasis -> Brain Sandbox)
    ax.annotate("", xy=(66.0, 75), xytext=(63.5, 75),
                arrowprops=dict(arrowstyle="-|>", color="#fb923c", lw=2.5, mutation_scale=18), zorder=4)
    ax.text(64.8, 77.0, "pi_2\nTrigger", color="#fb923c", fontsize=8.2, fontweight="bold", ha="center", zorder=5)

    # Connect Stage 4 Output up to the Return Conduit at (63.5, 56.5)
    ax.annotate("", xy=(63.5, 56.5), xytext=(68.0, 20.0),
                arrowprops=dict(arrowstyle="-|>", color="#fbbf24", lw=2.5, mutation_scale=18, connectionstyle="arc3,rad=-0.2"), zorder=4)
    ax.text(67.0, 38.0, "Solution\nDelta_h", color="#fbbf24", fontsize=8.2, fontweight="bold", ha="center", zorder=5)

    # Save output
    plt.tight_layout()
    output_path = "hadl_system_architecture.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[+] Clean, perfectly non-overlapping architecture diagram generated: {output_path}")

    # Copy to artifact directory
    artifact_dir = r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9"
    if os.path.exists(artifact_dir):
        dest_path = os.path.join(artifact_dir, output_path)
        shutil.copyfile(output_path, dest_path)
        print(f"[+] Copied to artifact directory: {dest_path}")

if __name__ == "__main__":
    create_hadl_diagram()
