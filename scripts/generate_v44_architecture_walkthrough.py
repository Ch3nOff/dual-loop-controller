"""
Publication-Grade Architecture & Evolution Walkthrough Diagram for HADL v4.4
Saves high-res diagram to:
- docs/images/hadl_v44_final_architecture_walkthrough.png
- artifact directory
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def generate_diagram():
    fig = plt.figure(figsize=(24, 15), facecolor='#090D16')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#090D16')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    C_TEXT = '#F8FAFC'
    C_SUBTEXT = '#94A3B8'
    C_CYAN = '#06B6D4'
    C_BLUE = '#3B82F6'
    C_PURPLE = '#8B5CF6'
    C_EMERALD = '#10B981'
    C_AMBER = '#F59E0B'
    C_ROSE = '#F43F5E'
    C_ACCENT = '#38BDF8'

    def draw_box(x, y, w, h, title, subtitle="", math_str="", box_color='#131B2E', border_color='#2A3B5C', badge=""):
        # Shadow
        shadow = FancyBboxPatch((x+0.35, y-0.35), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                facecolor='#000000', alpha=0.5, edgecolor='none')
        ax.add_patch(shadow)
        # Box
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                             facecolor=box_color, alpha=0.92, edgecolor=border_color, linewidth=1.6)
        ax.add_patch(box)
        # Title
        ax.text(x + w/2, y + h - 1.5, title, ha='center', va='center', fontsize=11, fontweight='bold', color=C_TEXT)
        # Badge
        if badge:
            ax.text(x + w - 1.0, y + h - 1.5, badge, ha='right', va='center', fontsize=7.5, fontweight='bold', color=C_AMBER)
        # Subtitle
        if subtitle:
            ax.text(x + w/2, y + h - 3.2, subtitle, ha='center', va='center', fontsize=8.5, color=C_SUBTEXT)
        # Math String
        if math_str:
            ax.text(x + w/2, y + 1.8, math_str, ha='center', va='center', fontsize=8.5, fontweight='semibold', color=C_ACCENT,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#060911', edgecolor=border_color, linewidth=0.8, alpha=0.9))

    def draw_arrow(x1, y1, x2, y2, label="", color='#64748B', style='->', lw=2.0, rad=0.0):
        connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                mutation_scale=16,
                connectionstyle=connectionstyle,
                shrinkA=3, shrinkB=3
            )
        )
        if label:
            mx = (x1 + x2) / 2 + (rad * 4)
            my = (y1 + y2) / 2
            ax.text(mx, my, label, ha='center', va='center', fontsize=8.5, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor='#0B0F19', edgecolor='none', alpha=0.85))

    # HEADER
    ax.text(50, 96.5, "HADL v4.4: UNIFIED COGNITIVE CONTROLLER & NEUROPLASTIC ARCHITECTURE",
            ha='center', va='center', fontsize=18, fontweight='heavy', color=C_TEXT)
    ax.text(50, 94.2, "Complete System Topology • 3 Cognitive Channels • Epistemic Metaplasticity • Ghost Verification",
            ha='center', va='center', fontsize=11, color=C_ACCENT)

    # LEFT PANEL: EVOLUTION WALKTHROUGH (v1.0 -> v4.4)
    ax.text(14, 90.0, "CHRONOLOGICAL EVOLUTION & RESOLVED OBSTACLES", ha='center', va='center',
            fontsize=12, fontweight='bold', color=C_AMBER)
    
    # Evo Box 1: v1 - v2
    draw_box(2, 73, 24, 14, "PHASE 1: v1.0 - v2.0 (Dual-Loop Genesis)",
             "• Problem: Catastrophic forgetting in single adapters\n• Problem: Rigid binary switches caused shock waves\n• Solved: Fast/Slow loop physical layer detachment",
             "L_fast || L_slow  (Raw Binary Partition)", box_color='#1E1B4B', border_color='#4338CA', badge="v1.0-v2.0")

    # Evo Box 2: v3 - v3.4
    draw_box(2, 53, 24, 15, "PHASE 2: v3.0 - v3.4 (VexDoor & Isometry)",
             "• Problem: Manifold drift & norm explosion in loops\n• Problem: Latent vectors distorted base semantics\n• Solved: Givens Unitary Rotator (0.000000 norm err)\n• Solved: Evolving Manifold Syringe & Nullspace",
             "h_rot = G(θ) · h  ||  Err_iso = 0.000000", box_color='#142B3B', border_color='#0284C7', badge="v3.0-v3.4")

    # Evo Box 3: v4.0 - v4.2
    draw_box(2, 32, 24, 16, "PHASE 3: v4.0 - v4.2 (CalLearn & OC-Dim)",
             "• Problem: Pretraining bias gravity wells (A*A=I clash)\n• Problem: Softmax unable to subtract/pop memory\n• Solved: CalLearn Prior Suppression (I - U U^T)\n• Solved: Signed In-Dynamic Softmax (sinh/cosh)\n• Solved: OC-Dim (D->D+K) Homotopy Stiefel frame",
             "S_pm = sinh / Σ cosh  ||  P_null = I - U U^T", box_color='#2A173B', border_color='#9333EA', badge="v4.0-v4.2")

    # Evo Box 4: v4.4 Current
    draw_box(2, 10, 24, 17, "PHASE 4: v4.4 (Current - Self-Aware ILTP)",
             "• Problem: Softmax exponential forced extreme 0/1\n• Problem: Hybrid tasks (story math) had no channel\n• Problem: Fine-tuning risk 1 (single-path monopoly)\n• Solved: Hermite Quintic Polynomial Q(u) Simplex\n• Solved: Continuous Jalur Tengah (I · h + α tanh)\n• Solved: Identity-Log Tangent Projector (ILTP)",
             "v_log = sign(Δ_I) ln(1+|Δ_I|) - λ ln(1+C)", box_color='#1B382B', border_color='#10B981', badge="v4.4 ACTIVE")

    # Arrows between Evo Boxes
    draw_arrow(14, 73, 14, 68, "Overcame Drift", color=C_CYAN, lw=1.8)
    draw_arrow(14, 53, 14, 48, "Overcame Bias", color=C_PURPLE, lw=1.8)
    draw_arrow(14, 32, 14, 27, "Overcame Skew", color=C_EMERALD, lw=1.8)

    # RIGHT/CENTER: COMPLETE PRODUCTION ARCHITECTURE PIPELINE
    ax.text(64, 90.0, "LIVE SYSTEM TOPOLOGY: THE TRI-MODAL CLOSED COGNITIVE LOOP", ha='center', va='center',
            fontsize=12, fontweight='bold', color=C_CYAN)

    # 1. Input Embedding
    draw_box(34, 78, 22, 9, "1. INPUT EMBEDDING LAYER",
             "Token matrix M_t ∈ R^{B x S x D}\nBase token projection & positional anchor",
             "M_t = EmbedTokens(input_ids)", box_color='#1E293B', border_color='#334155')

    # 2. Top Self-Aware Router
    draw_box(62, 74, 34, 13, "2. MANDATORY TOP ROUTER (P-MVR + ILTP)",
             "Self-Aware Riemannian Identity-Log Tangent Projector\nContinuous Order-5 Hermite Polynomial Simplex Q(u)\nHomeostatic Memory C_usage regulates fine-tuning fatigue",
             "u = 0.5(1 + tanh(z/1.5)),  Q(u) = 6u^5 - 15u^4 + 10u^3", box_color='#1B2A4A', border_color='#2563EB', badge="TOP HOOK")

    draw_arrow(45, 78, 62, 80, "Raw Embeddings", color=C_BLUE, lw=2.2)

    # 3 TRI-MODAL CHANNELS
    # Channel A: Fast Bypass
    draw_box(32, 53, 18, 14, "ROUTE 0: FAST BYPASS",
             "Conversations, Chat, Code\nPoetry, Translations\n• Latency Overhead: 0.00 ms\n• Base Weights: 100% Intact\n• Style Bleed: 0.00%",
             "P_byp ≥ 0.55  ⟹  α_eff = 0.0", box_color='#1E293B', border_color='#475569')

    # Channel B: Jalur Tengah
    draw_box(53, 53, 20, 14, "ROUTE 1: JALUR TENGAH",
             "Story Math (GSM8k), Essays\nMulti-step Narrative Logic\n• Identity Matrix Anchor I\n• Tanh-Bounded Vector Evolution\n• Balanced Dynamic α ∈ [0.15, 0.32]",
             "h_out = I · h + α_eff tanh(||Δ||) v̂", box_color='#1B382B', border_color='#059669')

    # Channel C: Heavy Deliberation
    draw_box(76, 53, 21, 14, "ROUTE 2: HEAVY LOOP",
             "Counterfactual, Bitwise Crypto\nInverted Physics, Reversible Stack\n• Directs full deliberation into\n  Layer 11 Core & Layer 23 Ghost\n• Epistemic Conflict Override",
             "P_hvy ≥ 0.55  ⟹  α_eff = 1.0", box_color='#2D1B4E', border_color='#9333EA')

    # Router Dispatch Arrows
    draw_arrow(70, 74, 41, 67, "P_byp", color=C_SUBTEXT, lw=2.0)
    draw_arrow(78, 74, 63, 67, "P_mid (Jalur Tengah)", color=C_EMERALD, lw=2.2)
    draw_arrow(86, 74, 86, 67, "P_hvy", color=C_PURPLE, lw=2.2)

    # 4. LAYER 11 CORE ENGINE
    draw_box(53, 27, 44, 18, "3. LAYER 11 INNER CORE: HADL UNIFIED ANALYTIC ENGINE",
             "Activated by Route 1 (Modulated α) and Route 2 (Full Deliberation)\n"
             "• CalLearn Engine: Prior well inversion via P_suppress = I - U U^T\n"
             "• OC-Dimension Module: Dynamic (D -> D+K) Stiefel inflation & Homotopy cooling\n"
             "• Signed In-Dynamic Softmax: Bipolar attention (sinh/cosh) for vector erasure\n"
             "• Givens & Householder: Manifold Isometry Rotator (0.000000 norm error)",
             "h_11 = CalLearn(h) + OC_Dim(h) + SignedAttn(Q,K,V)", box_color='#131B2E', border_color='#38BDF8')

    draw_arrow(63, 53, 63, 45, "Modulated Vector Injection", color=C_EMERALD, lw=2.0)
    draw_arrow(86, 53, 86, 45, "Full Deliberation Injection", color=C_PURPLE, lw=2.0)

    # Bypass links directly down past L11
    draw_arrow(41, 53, 41, 19, "Direct Flow (No Interference)", color=C_SUBTEXT, lw=2.0, rad=-0.1)

    # 5. LAYER 23 GHOST & OUTPUT
    draw_box(53, 5, 44, 14, "4. LAYER 23: GHOST VERIFICATION & HARD-WALL ASSIMILATION",
             "Dual-Stream Latent Comparator: Cross-examines Layer 11 Deliberation h_11\n"
             "against pre-LM Head representations via Cauchy-Schwarz Coherence Filter.\n"
             "• Eliminates internal hallucinations & runaway latent drift\n"
             "• Oblivion Guard Logits Processor prevents popped token resurrection",
             "h_verified = GhostVerify(h_final, h_11)  ⟹  LM_Head", box_color='#1A2333', border_color='#0284C7')

    draw_arrow(75, 27, 75, 19, "Verified Deliberation h_11", color=C_CYAN, lw=2.2)
    draw_arrow(41, 19, 53, 12, "Unmodified Base Stream", color=C_SUBTEXT, lw=2.0)

    # 6. FEEDBACK LOOP: HOMEOSTATIC METAPLASTICITY CLOSED LOOP
    draw_arrow(75, 5, 96, 5, "", color=C_ROSE, lw=1.8)
    draw_arrow(96, 5, 96, 78, "Closed Loop: C_usage Register Updates (Self-Aware Feedback)", color=C_ROSE, lw=1.8)
    draw_arrow(96, 78, 93, 78, "", color=C_ROSE, lw=1.8)

    # Save figure
    Path("docs/images").mkdir(parents=True, exist_ok=True)
    out_path = Path("docs/images/hadl_v44_final_architecture_walkthrough.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='#090D16')
    plt.close()
    print(f"[Done] Architecture diagram saved to {out_path}")

    # Copy to artifact dir
    artifact_dir = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")
    if artifact_dir.exists():
        art_path = artifact_dir / "hadl_v44_final_architecture_walkthrough.png"
        shutil.copy(out_path, art_path)
        print(f"[Done] Copied to artifact directory: {art_path}")

if __name__ == "__main__":
    generate_diagram()
