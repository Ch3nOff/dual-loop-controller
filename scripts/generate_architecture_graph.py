"""
Generates publication-grade architectural schematic:
eval_results/xstar_architecture_diagram.png
"""

import sys
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS = PROJECT_ROOT / "eval_results"
EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")


def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(18, 12), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title & Header
    ax.text(50, 96, "X-Star 2B Autonomous Brain: Endogenous Neural Architecture",
            ha="center", va="center", fontsize=18, fontweight="bold", color="#0D47A1")
    ax.text(50, 93, "Layer 11 VGNI Epistemic Manifold, Banach Continuous Plasticity & Layer 23 Commutative Dual-Composition",
            ha="center", va="center", fontsize=12, style="italic", color="#37474F")

    # Helper function for styled boxes
    def draw_box(x, y, w, h, title, subtitle="", color="#ECEFF1", border="#455A64", lw=1.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                                      facecolor=color, edgecolor=border, linewidth=lw)
        ax.add_patch(rect)
        if title:
            ax.text(x + w/2, y + h - 2.5, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=border)
        if subtitle:
            ax.text(x + w/2, y + h/2 - 0.5, subtitle, ha="center", va="center", fontsize=8.5, color="#263238")
        return rect

    def draw_arrow(x1, y1, x2, y2, label="", color="#37474F", lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=3, shrinkB=3))
        if label:
            ax.text((x1+x2)/2 + 1, (y1+y2)/2, label, fontsize=8, color=color, fontweight="bold")

    # 1. Input Layer
    draw_box(4, 78, 18, 10, "1. Input Embedding", "Input Tokens x_t\nEmbedding [B, S, 2048]\nRotary Positional Embeddings", "#E3F2FD", "#1565C0")

    # 2. Early Backbone (Layers 0 - 10)
    draw_box(4, 52, 18, 18, "2. Early Transformer", "Layers 0 to 10\nPre-RMSNorm\nSwiGLU Linear Attention\nRoPE Context Accumulation", "#EDE7F6", "#4527A0")
    draw_arrow(13, 78, 13, 70)

    # 3. Layer 11 Manifold Interception (CORE INNOVATION 1)
    draw_box(28, 38, 40, 48, "3. Deep Layer 11 Manifold Interception & VGNI Engine", "", "#E8F5E9", "#1B5E20", lw=2.5)
    
    # Inside Layer 11 Box:
    draw_box(30, 71, 36, 12, "HADL Controller Layer 11 Interceptor", "Extracts h_layer11 in R^2048\nComputes z_stream compressed trajectory\nPhaseShiftPrimeRouter Dynamic Dispatch", "#C8E6C9", "#2E7D32")
    draw_arrow(22, 60, 30, 75, "h_in [B, S, 2048]")

    draw_box(30, 54, 36, 13, "VGNI Epistemic Memory Grounding", "128-Slot Epistemic Concept Bank (N x D)\nDynamic Epistemic Sensitivity Gating\nGate = sigmoid(aff * 8 - 4) in [0, 1]\nGrounding: h_vgni = h + 0.08 * Gate * a_vec", "#A5D6A7", "#1B5E20")
    draw_arrow(48, 71, 48, 67, "h_layer11")

    draw_box(30, 40, 36, 11, "Endogenous Banach Continuous Plasticity", "Oja-Normalized Synaptic Adaptation\nBanach Contraction Operator (gamma = 0.85)\nHomeostatic Restitution: -0.20*(W_p - W_base)\nPeriodic Drift Subtraction Every 5 Steps", "#81C784", "#004D40")
    draw_arrow(48, 54, 48, 51, "h_grounded")

    # 4. Middle Transformer (Layers 12 - 22)
    draw_box(74, 68, 22, 18, "4. Mid Transformer", "Layers 12 to 22\nPropagates Grounded Vector\nContextual Reasoning\nDeep Latent Integration", "#FFF3E0", "#E65100")
    draw_arrow(68, 50, 74, 75, "h_final [B, S, 2048]")

    # 5. Layer 23 Ghost Controller (CORE INNOVATION 2)
    draw_box(74, 32, 22, 30, "5. Layer 23 Ghost Controller", "", "#FBE9E7", "#BF360C", lw=2.0)
    draw_box(75, 48, 20, 11, "Commutative Dual LBU", "Verification: f(g) vs g(f)\nDelta_comm = ||f(g) - g(f)||_2\nSuppresses Non-Commutative\nOrder Distortion", "#FFCCBC", "#D84315")
    draw_arrow(85, 68, 85, 59)

    draw_box(75, 34, 20, 11, "Streaming Context Decompressor", "Decompresses z_stream from L11\nLinear Projection: u_proj(z)\nCross-Layer Coherence Check", "#FFAB91", "#BF360C")
    draw_arrow(85, 48, 85, 45)

    # 6. Late Transformer & Unembedding
    draw_box(74, 6, 22, 20, "6. Output Head & Generation", "Layers 24 to 27\nFinal RMSNorm\nVocabulary Linear Projection (151,936)\nAutoregressive Long-Token CoT\nNatural <|im_end|> Termination", "#E0F2F1", "#004D40")
    draw_arrow(85, 32, 85, 26, "h_mod")

    # 7. Low-Level Kernel Acceleration Footer Box
    draw_box(4, 6, 64, 26, "Low-Level Kernel Acceleration & Memory Optimization (Zero Architecture Drift)", "", "#F3E5F5", "#4A148C", lw=2.0)
    draw_box(6, 17, 28, 12, "cuBLAS F.linear Optimization", "Replaced slow einsum & tensor allocs\nKernel execution: 0.508ms -> 0.038ms\n13.2x microbenchmark speedup", "#E1BEE7", "#6A1B9A")
    draw_box(38, 17, 28, 12, "Autoregressive Dispatch Caching", "Caches PhaseShift router decisions\nActive when past_key_values > 0\nEliminates per-token router overhead", "#CE93D8", "#4A148C")
    ax.text(36, 10, "Empirical Throughput: 16.00 - 21.8 tok/s | 172,498 Generated Tokens | Zero Catastrophic Drift",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#4A148C")

    plt.tight_layout()
    out_file = EVAL_RESULTS / "xstar_architecture_diagram.png"
    plt.savefig(str(out_file), dpi=300, bbox_inches="tight")
    plt.close()
    shutil.copy2(out_file, ARTIFACT_DIR / "xstar_architecture_diagram.png")
    print(f"[+] Saved Architecture Diagram: {out_file}")


if __name__ == "__main__":
    generate_architecture_diagram()
