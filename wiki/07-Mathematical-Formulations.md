# 07. Mathematical Formulations & Proofs

This document collects the theoretical foundations, optimization objectives, and stability proofs underpinning HADL v3.1.0.

---

## 1. Candès-Tao Latent Compressed Sensing

Let $z \in \mathbb{R}^D$ be the latent cognitive representation at layer $L_{mid}$, and let $\Phi \in \mathbb{R}^{M \times D}$ ($M \ll D$) be a semi-orthogonal measurement projection matrix satisfying the **Restricted Isometry Property (RIP)** of order $2k$ with constant $\delta_{2k} < \sqrt{2} - 1$:

$$(1 - \delta_{2k}) \|z\|_2^2 \le \|\Phi z\|_2^2 \le (1 + \delta_{2k}) \|z\|_2^2$$

The observation vector $y \in \mathbb{R}^M$ is given by:
$$y = \Phi z + \epsilon, \quad \|\epsilon\|_2 \le \sigma$$

The recovery problem is formulated as the convex $\ell_1$-regularized basis pursuit denoising problem:
$$\hat{z} = \arg\min_{z \in \mathbb{R}^D} \left\{ \frac{1}{2} \|\Phi z - y\|_2^2 + \lambda \|z\|_1 \right\}$$

### FISTA Convergence Guarantee
FISTA achieves optimal $O(1/k^2)$ convergence rate for non-smooth convex optimization:
$$F(z_k) - F(z^*) \le \frac{2 L \|\Phi z_0 - z^*\|_2^2}{(k + 1)^2}$$
where $L = \lambda_{max}(\Phi^T \Phi)$ is the maximum eigenvalue of the Gram matrix.

---

## 2. Orthogonal Nullspace Projection (Zero Catastrophic Forgetting)

Let $\mathcal{D}_1, \dots, \mathcal{D}_T$ be a sequence of tasks. For task $t$, let the representation subspace spanned by active activations be:
$$V_{prior} = \text{span}\left(\{h_i\}_{i=1}^{N_{prior}}\right) \subset \mathbb{R}^D$$

Using the thin QR decomposition of the basis matrix $B = [v_1, \dots, v_k] \in \mathbb{R}^{D \times k}$:
$$B = Q R$$
The orthogonal projection operator onto the nullspace $\mathcal{N}(V_{prior})$ is defined as:
$$P_{null} = I - Q Q^T$$

### Property 1 (Orthogonality):
$$\forall v \in V_{prior}, \quad P_{null} v = (I - Q Q^T) v = v - v = 0$$

### Property 2 (Zero Interference):
For any weight update $\Delta W_{new}$, the projected update $\Delta \tilde{W} = P_{null} \Delta W_{new}$ satisfies:
$$\|\Delta \tilde{W} v\|_2 = 0.000000 \quad \forall v \in V_{prior}$$
Hence, historical representations are invariant under new knowledge consolidation.

---

## 3. Active Inference & Expected Free Energy

Let $\pi \in \{\pi_0, \pi_1, \pi_2\}$ be candidate cognitive policies. The agent selects policy $\pi^*$ minimizing Expected Free Energy $G(\pi)$:

$$G(\pi) = \sum_\tau G(\pi, \tau)$$
$$G(\pi, \tau) = \underbrace{D_{KL}\left[ Q(o_\tau \mid \pi) \parallel P(o_\tau) \right]}_{\text{Instrumental / Goal Value}} + \underbrace{\mathbb{E}_{Q(s_\tau \mid \pi)}\left[ H\left[P(o_\tau \mid s_\tau)\right] \right]}_{\text{Epistemic / Exploration Value}}$$

Where:
* Policy $\pi_0$ (Bypass): High confidence ($u < 0.65$), negligible epistemic value &rarr; Route directly to LM head.
* Policy $\pi_1$ (Evidential): Moderate vacuity ($0.65 \le u < 0.85$) &rarr; Single-step verification.
* Policy $\pi_2$ (Deliberation): High vacuity ($u \ge 0.85$) &rarr; Multi-hop latent refinement loop.

---

## 4. ReZero Identity Initialization

Let $f_L(x)$ denote the native layer transformation of layer $L$. The augmented Dual-Loop layer is defined as:
$$h_{out} = f_L(x) + \alpha_{rezero} \cdot \mathcal{G}_{deliberate}(f_L(x))$$

With scalar initialization $\alpha_{rezero} = 0.0$:
$$h_{out} \Big|_{\alpha_{rezero}=0} = f_L(x) + 0 \cdot \mathcal{G}_{deliberate}(f_L(x)) \equiv f_L(x)$$

This guarantees strict mathematical identity preservation prior to gradient updates.
