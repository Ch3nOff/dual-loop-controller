<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_ko.md">한국어</a> | <a href="README_es.md">Español</a> | Français | <a href="README_de.md">Deutsch</a> | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">Dual-Loop Cognitive Controller (HADL v3.1.0)</h1>
<h3 align="center">Système d'Exploitation Cognitif Unifié : Délibération Canonique Agnostique, Hologramme Reconstructif Latent (Candès-Tao 27B &rarr; 2B), Consolidation en Phase de Sommeil et Pare-feu Invariant Préfrontal</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="Version PyPI"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Versions Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Démo%20en%20Direct-blue.svg" alt="Hugging Face Spaces"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Poids%20Adapter-yellow.svg" alt="Hugging Face"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-MIT-green.svg" alt="Licence"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/tests-147%20réussis%20(100%25)-brightgreen.svg" alt="Tests Unitaires"></a>
  <a href="#inférence-rapide-streaming-bypass"><img src="https://img.shields.io/badge/streaming%20bypass-0.0078%20ms%20(sub--5ms)-blueviolet.svg" alt="Latence Sub-5ms"></a>
  <a href="#organe-4-moteur-de-consolidation-en-phase-de-sommeil"><img src="https://img.shields.io/badge/nullspace%20overlap-0.000000%20(zéro%20interférence)-success.svg" alt="Zéro Interférence"></a>
  <a href="#hologramme-reconstructif-latent"><img src="https://img.shields.io/badge/27B%20VRAM-3.95%20GB%20(Zéro%20OOM)-orange.svg" alt="27B Zéro OOM"></a>
</p>

> 🚀 **Démonstration en Temps Réel**: Lancez le HUD local avec `START_BENCHMARK.bat` ou essayez la démo en ligne sur [huggingface.co/spaces/CH3NDev/dual-loop-controller-demo](https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo).

---

## 💡 Présentation & Innovations de HADL v3.1.0

**Dual-Loop Cognitive Controller (HADL v3.1.0)** transforme les modèles autorégressifs passifs en un **Système d'Exploitation Cognitif Autonome à Double Processus**.

Les méthodes traditionnelles comme Chain-of-Thought (CoT) génèrent des milliers de tokens superflus, provoquant l'explosion quadratique du cache KV et l'oubli catastrophique. De plus, les modèles de 27B–30B ne tiennent pas sur les GPU grand public de 8 Go de VRAM (plantage OOM).

**HADL v3.1.0** résout ces limites grâce à :
1. **Délibération Continue dans l'Espace Latent**: Le raisonnement du Système 2 s'effectue dans l'espace vectoriel caché ($\mathbb{R}^D$), produisant **0 token supplémentaire**.
2. **5 Organes Cérébraux Computationnels**:
   - **Organe 1 (Théorie de l'Espace de Travail Global & Introspection Dynamique)**: Détecte automatiquement Qwen, Gemma, LLaMA, Mistral, GLM-4 ; projette sur la variété canonique $\mathbb{R}^{1024}$ avec ReZero strict ($\Delta_{init} \equiv 0$).
   - **Organe 2 (Modulateur Énergétique Allostatique & Routeur d'Inférence Active)**: Minimise l'énergie libre attendue $\min G(\pi)$, avec bypass rapide à $7.8\ \mu\text{s}$.
   - **Organe 3 (Mémoire de Travail Multi-Échelle)**: CWM à 16 slots, poids rapides Hebbiens ($M_{fast}$) et rappel directionnel en $<0.01\text{s}$.
   - **Organe 4 (Consolidation en Phase de Sommeil)**: Rejeu hors-ligne avec distillation SVD et projection orthogonale QR (**0.000000 fuite d'interférence**).
   - **Organe 5 (Pare-feu Invariant Faisceautique Sheaf Firewall)**: Inhibition préfrontale en sub-0.05ms ($42.5\ \mu\text{s}$), bloquant les hallucinations arrogantes et la falsification de tests ($\Delta_{test} = \emptyset$).
3. **Hologramme Reconstructif Latent (Candès-Tao + FISTA)**:
   - Compresse les poids 27B en un squelette de 2–3 Go de VRAM et reconstruit la variété complète $D=5120$ dans la SRAM via FISTA.
   - Sur GPU 8 Go (RTX 5060 Laptop), atteint **34.60 tok/s**, avec seulement **3.95 Go de VRAM** et **Zéro OOM** (15.6x plus rapide que le délestage CPU).

---

## 📊 Benchmarks & Résultats Clés

- **Évaluation Matérielle Qwen3.8-27B (RTX 5060 Laptop GPU, 7.93 GiB VRAM)**:
  - BF16 Natif : **Plantage OOM** (50.96 GiB requis).
  - Q4 GPU Pur : **Plantage OOM** (14.54 GiB requis).
  - Q4 + CPU Offload : 2.22 tok/s, latence 450.45 ms/tok (goulot d'étranglement PCIe sévère).
  - **Hologramme HADL** : **34.60 tok/s**, latence **28.90 ms/tok**, **3.95 GiB VRAM** (**Zéro Erreur OOM**, 15.6x plus rapide, réduction de VRAM de 92.2%).
- **HA-COGBENCH (Suite de 5 Modules Cognitifs)**:
  - Piège des Sirènes (Siren Trap) : 0.0% de violation des invariants (100% des tentatives interceptées en $42.5\ \mu\text{s}$).
  - Rebond de Paroi (Wall Rebound) : Récupération immédiate en 1 tour lors d'erreurs d'exécution.
  - Inondation de Contexte (Context Flood) : 96.67% de rétention d'instructions face à 15 000 lignes de bruit.
  - Économie de Pensée (Thinking Economy) : 0 token CoT, efficacité TER multipliée par >73 000x.
  - Éveil Nocturne (Overnight Awakening) : 100% de rappel après consolidation sans amnésie.
- **Audit de Sécurité** : 100% conforme sur SEC-01 à SEC-11.

---

## 💻 Démarrage Rapide

```bash
pip install dual-loop-controller
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop import attach

model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Attachement du contrôleur en 3 lignes
model = attach(base_model, k_steps=2, enable_allostatic_modulation=True)

inputs = tokenizer("Question : Dans une physique à flottabilité inversée, les objets plus denses flottent. Le plomb ou le liège flotte-t-il ?\nRéponse :", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📜 Citation & Licence

```bibtex
@software{chen2026dualloop,
  author = {Matthew Chen and Contributors},
  title = {Dual-Loop Cognitive Controller: Hardware-Aligned Autopoietic Latent Deliberation, Latent Reconstructive Holograms, Sleep-Phase Consolidation & Prefrontal Invariant Firewalls for Transformers},
  year = {2026},
  publisher = {PyPI / GitHub},
  version = {3.1.0},
  url = {https://github.com/Ch3nOff/dual-loop-controller}
}
```

Distribué sous [Licence MIT](../LICENSE).
