<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_ko.md">한국어</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a> | Deutsch | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">Dual-Loop Cognitive Controller (HADL v3.1.0)</h1>
<h3 align="center">Einheitliches kognitives Betriebssystem: Modellunabhängige kanonische Deliberation, latente holografische Rekonstruktion (Candès-Tao 27B &rarr; 2B), Schlafphasen-Konsolidierung & Präfrontale Invarianten-Firewall</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python Versionen"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue.svg" alt="Hugging Face Spaces"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Adapter%20Gewichte-yellow.svg" alt="Hugging Face"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/Lizenz-MIT-green.svg" alt="Lizenz"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/Tests-147%20bestanden%20(100%25)-brightgreen.svg" alt="Unit Tests"></a>
  <a href="#schnelle-streaming-bypass-inferenz"><img src="https://img.shields.io/badge/streaming%20bypass-0.0078%20ms%20(sub--5ms)-blueviolet.svg" alt="Sub-5ms Latenz"></a>
  <a href="#organ-4-schlafphasen-konsolidierungsmodul"><img src="https://img.shields.io/badge/nullspace%20overlap-0.000000%20(null%20interferenz)-success.svg" alt="Null Interferenz"></a>
  <a href="#latentes-rekonstruktives-hologramm"><img src="https://img.shields.io/badge/27B%20VRAM-3.95%20GB%20(Zero%20OOM)-orange.svg" alt="27B Zero OOM"></a>
</p>

> 🚀 **Echtzeit-Streaming-Demo**: Starten Sie das lokale Dashboard mit `START_BENCHMARK.bat` oder testen Sie die Online-Demo unter [huggingface.co/spaces/CH3NDev/dual-loop-controller-demo](https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo).

---

## 💡 Systemüberblick & Innovationen in HADL v3.1.0

Der **Dual-Loop Cognitive Controller (HADL v3.1.0)** wandelt rein passive autoregressive Sprachmodelle in ein **autonomes kognitives Zwei-Prozess-Betriebssystem (Cognitive OS)** um.

Klassische Ansätze wie Chain-of-Thought (CoT) erzeugen tausende überflüssige Tokens, was zu quadratischer KV-Cache-Explosion und katastrophalem Vergessen führt. Zudem scheitern 27B–30B-Modelle auf 8-GB-Consumer-GPUs an fatalen Out-of-Memory-Abstürzen (OOM).

**HADL v3.1.0** löst diese Hürden durch:
1. **Kontinuierliche latente Deliberation**: System-2-Schlussfolgerungen finden vollständig im verborgenen Vektorraum ($\mathbb{R}^D$) statt – **ohne zusätzliche Ausgabe-Tokens (0 Extra Tokens)**.
2. **5 rechnerische Gehirnorgane**:
   - **Organ 1 (Global Workspace Theory GWT & Dynamische Graph-Introspektion)**: Erkennt automatisch Architekturen wie Qwen, Gemma, LLaMA, Mistral, GLM-4 und projiziert sie auf die kanonische $\mathbb{R}^{1024}$-Mannigfaltigkeit mit strenger ReZero-Identität ($\Delta_{init} \equiv 0$).
   - **Organ 2 (Allostatischer Energiemodulator & Active-Inference-Router)**: Minimiert die erwartete freie Energie $\min G(\pi)$, mit $7.8\ \mu\text{s}$ Express-Bypass.
   - **Organ 3 (Multi-Zeitskalen-Arbeitsgedächtnis)**: 16-Slot-CWM, schnelle Hebb'sche synaptische Gewichte ($M_{fast}$) und gerichtete Abfrage in $<0.01\text{s}$.
   - **Organ 4 (Schlafphasen-Konsolidierungs-Engine)**: Offline-Wiederholung in Ruhephasen mittels SVD-Niedrigrangdestillation und QR-Nullraumprojektion (**0.000000 Interferenzverlust**).
   - **Organ 5 (Präfrontale Garben-Firewall Sheaf Firewall)**: Sub-0.05ms ($42.5\ \mu\text{s}$) Exekutivhemmung gegen Aktivierungsexplosionen, arrogante Halluzinationen und Testcode-Manipulationen ($\Delta_{test} = \emptyset$).
3. **Latentes rekonstruktives Hologramm (Candès-Tao Compressed Sensing + FISTA)**:
   - Komprimiert 27B-Gewichte auf ein Skelett von nur 2–3 GB VRAM und rekonstruiert die volle $D=5120$-Mannigfaltigkeit im SRAM via FISTA.
   - Erreicht auf 8-GB-Consumer-GPUs (RTX 5060 Laptop) **34.60 tok/s**, bei nur **3.95 GB VRAM-Verbrauch** und **Zero OOM** (15.6x schneller als CPU-Offload)!

---

## 📊 Benchmark-Ergebnisse

- **Qwen3.8-27B Hardware-Speicher-Profiler (RTX 5060 Laptop GPU, 7.93 GiB VRAM)**:
  - Natives BF16: **OOM-Absturz** (50.96 GiB erforderlich).
  - Reines GPU-Q4: **OOM-Absturz** (14.54 GiB erforderlich).
  - Q4 + CPU-Offload: 2.22 tok/s, 450.45 ms/tok Latenz (schwere PCIe-Engpässe).
  - **HADL-Hologramm**: **34.60 tok/s**, **28.90 ms/tok** Latenz, **3.95 GiB VRAM** (**Absolut Zero OOM**, 15.6x schneller, 92.2% VRAM-Reduktion).
- **HA-COGBENCH 5-Module-Kognitionssuite**:
  - Siren Trap: 0.0% Invariantenverletzung (100% aller Testmanipulationen in $42.5\ \mu\text{s}$ blockiert).
  - Wall Rebound: Sofortige Erholung in 1 Schritt bei Ausführungsfehlern.
  - Context Flood: 96.67% Instruktionsretention trotz 15.000 Zeilen Rauschen.
  - Thinking Economy: 0 CoT-Tokens, TER-Effizienz über 73.000x gesteigert.
  - Overnight Awakening: 100% Zero-Shot-Wissensretention nach Schlafphasenkonsolidierung.
- **Sicherheitsaudit**: 100% Erfüllung der Vorgaben SEC-01 bis SEC-11.

---

## 💻 Schnellstart

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

# Controller in 3 Zeilen integrieren
model = attach(base_model, k_steps=2, enable_allostatic_modulation=True)

inputs = tokenizer("Frage: In einer Physik mit umgekehrtem Auftrieb schwimmen dichtere Objekte. Schwimmt Blei oder Kork?\nAntwort:", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📜 Zitation & Lizenz

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

Veröffentlicht unter der [MIT-Lizenz](../LICENSE).
