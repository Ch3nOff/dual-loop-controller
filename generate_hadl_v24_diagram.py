import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_hadl_v24_diagram(output_path):
    fig = plt.figure(figsize=(26, 15), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    bg_color = "#070b14"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

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

    def draw_arrow(x1, y1, x2, y2, color=c_accent_cyan, lw=2.0, style="->", rad=0.0):
        connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3"
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            color=color,
            linewidth=lw,
            connectionstyle=connectionstyle,
            zorder=4
        )
        ax.add_patch(arrow)
        return arrow

    # =============================================================
    # 0. HEADER BANNER
    # =============================================================
    draw_card(2, 92, 96, 6.8, bg="#0b1329", border="#1e293b", lw=2.2)
    ax.text(50, 96.8, "DUAL-LOOP COGNITIVE CONTROLLER v2.4.0 ARCHITECTURE", 
            ha="center", va="center", fontsize=20, fontweight="bold", color=c_accent_cyan, zorder=3)
    ax.text(50, 93.6, "Gate Pruning & Allostatic Energy Modulation • Decoupled Autonomous Curiosity Daemon • Bounded Epistemic Humility • Popperian Red Team • Orthogonal Nullspace Memory", 
            ha="center", va="center", fontsize=10.5, color=c_text_muted, zorder=3)

    # =============================================================
    # 1. LEFT COLUMN: TRANSFORMER & FAST-PATH INFERENCE (Sub-5ms)
    # =============================================================
    draw_card(2.5, 6, 22.5, 83.5, bg="#090d1a", border="#1e293b", lw=2.0)
    ax.text(13.75, 87.2, "ONLINE USER INFERENCE", ha="center", va="center", fontsize=13, fontweight="bold", color=c_accent_cyan, zorder=3)
    ax.text(13.75, 85.0, "(Fast-Path Clock: Sub-5ms)", ha="center", va="center", fontsize=9.5, color=c_accent_emerald, zorder=3)

    # Input Tokens
    draw_card(4.5, 75.5, 18.5, 6.8, bg="#131e36", border="#334155")
    ax.text(13.75, 79.8, "Prompt Tokens x_t", ha="center", va="center", fontsize=10, fontweight="bold", color=c_text_bright, zorder=3)
    ax.text(13.75, 77.2, "User Query Stream", ha="center", va="center", fontsize=8.5, color=c_text_muted, zorder=3)

    # Transformer Backbone
    draw_card(4.5, 50.5, 18.5, 21, bg="#131e36", border="#334155")
    ax.text(13.75, 69.2, "Base Transformer", ha="center", va="center", fontsize=10.5, fontweight="bold", color=c_accent_cyan, zorder=3)
    ax.text(13.75, 66.8, "(Frozen Qwen / LLaMA / GLM)", ha="center", va="center", fontsize=8.5, color=c_text_muted, zorder=3)
    
    draw_card(6.0, 58.5, 15.5, 6.5, bg="#1e293b", border="#475569")
    ax.text(13.75, 62.5, "Layers 1 to L-1", ha="center", va="center", fontsize=9, color=c_text_muted, zorder=3)
    ax.text(13.75, 60.2, "Feed-Forward Context", ha="center", va="center", fontsize=8, color=c_text_muted, zorder=3)
    
    draw_card(6.0, 52.0, 15.5, 5.0, bg="#1e293b", border=c_accent_amber, lw=1.5)
    ax.text(13.75, 54.5, "Layer Hook (L_mid)", ha="center", va="center", fontsize=9, fontweight="bold", color=c_accent_amber, zorder=3)

    # Output Generation
    draw_card(4.5, 10.5, 18.5, 16.5, bg="#131e36", border="#334155")
    ax.text(13.75, 24.5, "Output Generation", ha="center", va="center", fontsize=10.5, fontweight="bold", color=c_accent_emerald, zorder=3)
    ax.text(13.75, 21.8, "Autoregressive Stream", ha="center", va="center", fontsize=8.5, color=c_text_muted, zorder=3)
    
    draw_card(6.0, 12.5, 15.5, 7.0, bg="#162e21", border=c_accent_emerald, lw=1.5)
    ax.text(13.75, 16.8, "pi_0 Syntax Bypass", ha="center", va="center", fontsize=9, fontweight="bold", color=c_accent_emerald, zorder=3)
    ax.text(13.75, 14.2, "Latency: 0.0078 ms (7.8 us)", ha="center", va="center", fontsize=8, color="#86efac", zorder=3)

    draw_arrow(13.75, 75.5, 13.75, 71.5, color=c_accent_cyan)
    draw_arrow(13.75, 50.5, 13.75, 27.0, color=c_accent_emerald, lw=2.2)

    # =============================================================
    # 2. MIDDLE COLUMN: PRUNED ALLOSTATIC ENERGY MODULATION & ROUTING
    # =============================================================
    draw_card(27.0, 6, 38.0, 83.5, bg="#090d1a", border="#1e293b", lw=2.0)
    ax.text(46.0, 87.2, "CONSOLIDATED ALLOSTATIC ENERGY ENGINE", ha="center", va="center", fontsize=13, fontweight="bold", color=c_accent_cyan, zorder=3)
    ax.text(46.0, 85.0, "Pruned Unified Gate Potential in Energy Logit Space", ha="center", va="center", fontsize=9.5, color=c_text_muted, zorder=3)

    # Friston Active Inference Policy Router
    draw_card(29.0, 69.5, 34.0, 13.5, bg="#131e36", border=c_accent_violet, lw=1.8)
    ax.text(46.0, 80.5, "Active Inference Policy Router (Friston EFE)", ha="center", va="center", fontsize=11, fontweight="bold", color=c_accent_violet, zorder=3)
    ax.text(46.0, 77.8, "Minimizes Expected Free Energy G(pi) = E_Q[ln Q(s|pi) - ln P(s,o)]", ha="center", va="center", fontsize=8.5, color=c_text_muted, zorder=3)
    
    # 3 Policy branches
    draw_card(30.2, 71.2, 10.0, 5.0, bg="#1e1b4b", border=c_accent_emerald)
    ax.text(35.2, 73.7, "pi_0: Bypass\nu < 0.65 (Fluent)", ha="center", va="center", fontsize=8, color="#86efac", zorder=3)

    draw_card(41.0, 71.2, 10.0, 5.0, bg="#1e1b4b", border=c_accent_cyan)
    ax.text(46.0, 73.7, "pi_1: Fast Check\n0.65 <= u < 0.85", ha="center", va="center", fontsize=8, color="#7dd3fc", zorder=3)

    draw_card(51.8, 71.2, 10.0, 5.0, bg="#1e1b4b", border=c_accent_amber)
    ax.text(56.8, 73.7, "pi_2: Brain Sand\nu >= 0.85 (Ponder)", ha="center", va="center", fontsize=8, color="#fde68a", zorder=3)

    # Allostatic Energy Modulator Box
    draw_card(29.0, 37.5, 34.0, 28.5, bg="#111827", border=c_accent_cyan, lw=2.0)
    ax.text(46.0, 63.5, "Allostatic Energy Modulator (Gate Pruning)", ha="center", va="center", fontsize=11.5, fontweight="bold", color=c_accent_cyan, zorder=3)
    ax.text(46.0, 61.0, "Replaces 5-gate multiplicative cascade with scalar potential logit", ha="center", va="center", fontsize=8.5, color=c_text_muted, zorder=3)

    # Equation Card
    draw_card(30.5, 47.0, 31.0, 12.0, bg="#0b1329", border="#2563eb", lw=1.5)
    ax.text(46.0, 56.0, "E_allo = w_surp*g_surp + w_beta*g_beta - w_drift*g_drift - w_vac*max(0, u - 0.50)", 
            ha="center", va="center", fontsize=7.8, color="#93c5fd", family="monospace", zorder=3)
    ax.text(46.0, 52.0, "Gamma_allostatic = sigma( E_allo / tau ) in [0.40, 0.95]", 
            ha="center", va="center", fontsize=9.0, fontweight="bold", color="#38bdf8", family="monospace", zorder=3)
    ax.text(46.0, 48.8, "delta_refined = scale * Gamma_allostatic * raw_delta", 
            ha="center", va="center", fontsize=8.5, color=c_text_bright, family="monospace", zorder=3)

    # Performance badge inside Allostasis
    draw_card(30.5, 39.0, 31.0, 6.5, bg="#162e21", border=c_accent_emerald)
    ax.text(46.0, 43.5, "Signal Preservation: 96.6% (vs 13.4% Cascade Collapse)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#86efac", zorder=3)
    ax.text(46.0, 41.0, "Fast-Path Deliberation: 5.02 ms | Gradient Flow: Zero Vanishing", ha="center", va="center", fontsize=8, color="#bbf7d0", zorder=3)

    # 4-Stage Brain Sandbox
    draw_card(29.0, 10.5, 34.0, 24.0, bg="#131e36", border=c_accent_amber, lw=1.8)
    ax.text(46.0, 32.0, "Brain Sandbox Deliberation (pi_2)", ha="center", va="center", fontsize=11, fontweight="bold", color=c_accent_amber, zorder=3)
    ax.text(46.0, 29.5, "Isolated Multi-Pass Refinement (K Steps)", ha="center", va="center", fontsize=8.5, color=c_text_muted, zorder=3)

    # 4 Sandbox steps
    step_data = [
        ("1. Propose", "Latent Idea"),
        ("2. Challenge", "Red Team"),
        ("3. Verify", "Sandbox exec"),
        ("4. Project", "Nullspace")
    ]
    for idx, (s_title, s_sub) in enumerate(step_data):
        draw_card(30.2 + idx * 7.8, 13.0, 7.2, 13.5, bg="#1e293b", border="#475569")
        ax.text(33.8 + idx * 7.8, 23.5, s_title, ha="center", va="center", fontsize=8.0, fontweight="bold", color=c_text_bright, zorder=3)
        ax.text(33.8 + idx * 7.8, 16.5, s_sub, ha="center", va="center", fontsize=7.0, color=c_text_muted, zorder=3)

    # Connecting arrows
    draw_arrow(21.5, 54.5, 29.0, 54.5, color=c_accent_cyan, lw=2.2)
    draw_arrow(46.0, 69.5, 46.0, 66.0, color=c_accent_violet, lw=2.0)
    draw_arrow(46.0, 37.5, 46.0, 34.5, color=c_accent_amber, lw=2.0)
    draw_arrow(29.0, 48.0, 23.0, 24.5, color=c_accent_emerald, lw=2.2, rad=0.2)

    # =============================================================
    # 3. RIGHT COLUMN: AUTONOMOUS BACKGROUND DAEMON & POPPERIAN RED TEAM
    # =============================================================
    draw_card(67.0, 6, 30.5, 83.5, bg="#090d1a", border="#1e293b", lw=2.0)
    ax.text(82.25, 87.2, "AUTONOMOUS BACKGROUND DAEMON", ha="center", va="center", fontsize=13, fontweight="bold", color=c_accent_rose, zorder=3)
    ax.text(82.25, 85.0, "Decoupled Clock: Continuous Idle Contemplation", ha="center", va="center", fontsize=9.5, color=c_accent_amber, zorder=3)

    # Epistemic Humility Module
    draw_card(68.5, 68.0, 27.5, 15.0, bg="#1e1222", border=c_accent_rose, lw=1.8)
    ax.text(82.25, 80.5, "Epistemic Humility Module", ha="center", va="center", fontsize=11, fontweight="bold", color=c_accent_rose, zorder=3)
    ax.text(82.25, 78.0, "Bounded Dirichlet Confidence c <= 0.95 | Vacuity u >= 0.05", ha="center", va="center", fontsize=7.8, color="#fda4af", zorder=3)
    
    draw_card(70.0, 69.5, 24.5, 6.5, bg="#2c142e", border="#f43f5e")
    ax.text(82.25, 74.0, "L_overconf = I_error * ( c / (1 - c + eps) )^2", ha="center", va="center", fontsize=8.0, color="#fecdd3", family="monospace", zorder=3)
    ax.text(82.25, 71.0, "Overconfident Errors: 0.0% | ECE: 0.2488 (vs 0.6396 Base)", ha="center", va="center", fontsize=7.5, color="#fbcfe8", zorder=3)

    # Popperian Self-Play Red Team
    draw_card(68.5, 43.5, 27.5, 22.0, bg="#131e36", border=c_accent_amber, lw=1.8)
    ax.text(82.25, 62.5, "Popperian Red Team Self-Play", ha="center", va="center", fontsize=11, fontweight="bold", color=c_accent_amber, zorder=3)
    ax.text(82.25, 60.0, "Proposer vs Falsifier Hypothesis Refutation", ha="center", va="center", fontsize=8.5, color=c_text_muted, zorder=3)

    draw_card(70.0, 51.5, 11.5, 6.5, bg="#1e293b", border=c_accent_cyan)
    ax.text(75.75, 55.5, "Proposer", ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_accent_cyan, zorder=3)
    ax.text(75.75, 53.0, "Hypothesis phi", ha="center", va="center", fontsize=7.5, color=c_text_muted, zorder=3)

    draw_card(83.0, 51.5, 11.5, 6.5, bg="#1e293b", border=c_accent_rose)
    ax.text(88.75, 55.5, "Falsifier", ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_accent_rose, zorder=3)
    ax.text(88.75, 53.0, "Counter-Ex psi", ha="center", va="center", fontsize=7.5, color=c_text_muted, zorder=3)

    draw_card(70.0, 45.0, 24.5, 5.0, bg="#1e293b", border=c_accent_emerald)
    ax.text(82.25, 47.5, "Deterministic Sandbox Verification (Truth Gate)", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#86efac", zorder=3)

    # Orthogonal Nullspace Memory Bank
    draw_card(68.5, 10.5, 27.5, 30.5, bg="#0d1f1d", border=c_accent_emerald, lw=2.0)
    ax.text(82.25, 38.5, "Orthogonal Nullspace Memory", ha="center", va="center", fontsize=11.5, fontweight="bold", color=c_accent_emerald, zorder=3)
    ax.text(82.25, 36.0, "v_ortho = v - B(B^T B)^-1 B^T v  =>  v_ortho perp Basis", ha="center", va="center", fontsize=8.0, color="#6ee7b7", family="monospace", zorder=3)

    draw_card(70.0, 22.0, 24.5, 12.0, bg="#092015", border="#059669")
    ax.text(82.25, 30.5, "Lifelong Continual Retention: 100.0%", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#86efac", zorder=3)
    ax.text(82.25, 27.5, "Cosine Overlap with Priors: 0.000000", ha="center", va="center", fontsize=8.5, color="#a7f3d0", zorder=3)
    ax.text(82.25, 24.5, "AARR (Anomaly Resolution): 100.0% (20/20)", ha="center", va="center", fontsize=8.5, color="#a7f3d0", zorder=3)

    draw_card(70.0, 12.5, 24.5, 7.5, bg="#132e27", border=c_accent_emerald)
    ax.text(82.25, 17.5, "Decoupled Background Loop", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#86efac", zorder=3)
    ax.text(82.25, 14.5, "Zero Impact on User Fast-Path Latency", ha="center", va="center", fontsize=7.5, color="#6ee7b7", zorder=3)

    # Daemon internal flow arrows
    draw_arrow(82.25, 68.0, 82.25, 65.5, color=c_accent_amber)
    draw_arrow(82.25, 43.5, 82.25, 41.0, color=c_accent_emerald)
    draw_arrow(68.5, 25.0, 63.0, 48.0, color=c_accent_emerald, lw=2.0, rad=-0.2)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Revised Architecture diagram saved to: {output_path}")

if __name__ == "__main__":
    out1 = r"C:\Users\Matthew Chen\Documents\X-Star\hadl_v24_system_architecture.png"
    out2 = r"C:\Users\Matthew Chen\Documents\bench\hadl_v24_system_architecture.png"
    create_hadl_v24_diagram(out1)
    create_hadl_v24_diagram(out2)
