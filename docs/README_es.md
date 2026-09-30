<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_ko.md">한국어</a> | Español | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">Dual-Loop Cognitive Controller (HADL v3.1.0)</h1>
<h3 align="center">Sistema Operativo Cognitivo Unificado: Deliberación Canónica Agnóstica, Holograma Reconstructivo Latente (Candès-Tao 27B &rarr; 2B), Consolidación en Fase de Sueño y Firewall Invariante Prefrontal</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="Versión PyPI"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Versiones de Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo%20en%20Vivo-blue.svg" alt="Hugging Face Spaces"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Pesos%20Adapter-yellow.svg" alt="Hugging Face"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licencia-MIT-green.svg" alt="Licencia"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/pruebas-147%20aprobadas%20(100%25)-brightgreen.svg" alt="Pruebas Unitarias"></a>
  <a href="#inferencia-rapida-streaming-bypass"><img src="https://img.shields.io/badge/streaming%20bypass-0.0078%20ms%20(sub--5ms)-blueviolet.svg" alt="Latencia Sub-5ms"></a>
  <a href="#organo-4-motor-de-consolidacion-en-fase-de-sueno"><img src="https://img.shields.io/badge/nullspace%20overlap-0.000000%20(cero%20interferencia)-success.svg" alt="Cero Interferencia"></a>
  <a href="#holograma-reconstructivo-latente"><img src="https://img.shields.io/badge/27B%20VRAM-3.95%20GB%20(Cero%20OOM)-orange.svg" alt="27B Cero OOM"></a>
</p>

> 🚀 **Demostración en Tiempo Real**: Ejecute localmente el HUD con `START_BENCHMARK.bat` o pruebe la demo online en [huggingface.co/spaces/CH3NDev/dual-loop-controller-demo](https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo).

---

## 💡 Resumen Ejecutivo y Novedades de HADL v3.1.0

**Dual-Loop Cognitive Controller (HADL v3.1.0)** transforma los Modelos de Lenguaje Autoregresivos pasivos en un **Sistema Operativo Cognitivo Autónomo de Doble Proceso**.

Los enfoques convencionales como Chain-of-Thought (CoT) queman miles de tokens adicionales, causando una explosión cuadrática de memoria KV-cache y olvido catastrófico. Además, modelos grandes (27B–30B) no caben en GPUs de consumo de 8GB VRAM (colapso OOM).

**HADL v3.1.0** resuelve estos retos mediante:
1. **Deliberación Continua en Espacio Latente**: El razonamiento del Sistema 2 opera completamente en el espacio vectorial oculto ($\mathbb{R}^D$), generando **0 tokens adicionales**.
2. **5 Órganos Cerebrales Computacionales**:
   - **Órgano 1 (GWT & Introspección Dinámica de Grafos)**: Compatible con Qwen, Gemma, LLaMA, Mistral, GLM-4; proyecta a $\mathbb{R}^{1024}$ con identidad ReZero estricta ($\Delta_{init} \equiv 0$).
   - **Órgano 2 (Modulador Alostático & Enrutador de Inferencia Activa)**: Minimiza la energía libre esperada $\min G(\pi)$, con bypass de $7.8\ \mu\text{s}$.
   - **Órgano 3 (Memoria de Trabajo Multiescala)**: CWM de 16 slots, pesos rápidos Hebbianos ($M_{fast}$) y recuperación direccional en $<0.01\text{s}$.
   - **Órgano 4 (Consolidación en Fase de Sueño)**: Repetición offline con destilación SVD y proyección a subespacio ortogonal QR (**0.000000 fuga de interferencia**).
   - **Órgano 5 (Firewall Invariante Sheaf)**: Inhibición ejecutiva prefrontal en sub-0.05ms ($42.5\ \mu\text{s}$), bloqueando alucinaciones arrogantes y manipulación de tests ($\Delta_{test} = \emptyset$).
3. **Holograma Reconstructivo Latente (Candès-Tao + FISTA)**:
   - Comprime modelos 27B a un esqueleto de 2–3 GB VRAM y reconstruye el espacio latente completo $D=5120$ en SRAM.
   - En GPU de 8GB (RTX 5060 Laptop), alcanza **34.60 tok/s**, consumo de solo **3.95 GB VRAM** y **Cero Errores OOM** (15.6x más rápido que CPU offload).

---

## 📊 Resumen de Benchmarks Empíricos

- **Profiler de Hardware Qwen3.8-27B (RTX 5060 Laptop GPU, 7.93 GiB VRAM)**:
  - BF16 Nativo: **Colapso OOM** (Requería 50.96 GiB).
  - Q4 GPU Pura: **Colapso OOM** (Requería 14.54 GiB).
  - Q4 + CPU Offload: 2.22 tok/s, latencia 450.45 ms/tok (Grave cuello de botella PCIe).
  - **Holograma HADL**: **34.60 tok/s**, latencia **28.90 ms/tok**, **3.95 GiB VRAM** (**Cero OOM**, 15.6x más rápido, -92.2% reducción de VRAM).
- **HA-COGBENCH (Suite de 5 Módulos Cognitivos)**:
  - Trampa de la Sirena (Siren Trap): 0.0% violaciones de invariantes (100% de trampas interceptadas en $42.5\ \mu\text{s}$).
  - Rebote contra la Pared (Wall Rebound): Recuperación inmediata en 1 turno frente a errores de ejecución.
  - Inundación de Contexto (Context Flood): 96.67% de retención frente a 15,000 líneas de ruido.
  - Economía de Pensamiento (Thinking Economy): 0 tokens CoT, eficiencia TER >73,000x superior.
  - Despertar Nocturno (Overnight Awakening): 100% retención tras reinicio con consolidación offline.
- **Auditoría de Seguridad**: Cumplimiento del 100% en SEC-01 a SEC-11.

---

## 💻 Inicio Rápido

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

# Acoplar el controlador en 3 líneas
model = attach(base_model, k_steps=2, enable_allostatic_modulation=True)

inputs = tokenizer("Pregunta: En física de flotabilidad invertida, los objetos más densos flotan. ¿Flota el plomo o el corcho?\nRespuesta:", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📜 Citación y Licencia

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

Distribuido bajo la [Licencia MIT](../LICENSE).
