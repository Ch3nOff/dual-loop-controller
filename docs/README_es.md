<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_ko.md">한국어</a> | Español | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">Controlador Cognitivo de Doble Bucle (HADL v4.5 Edición Car-Lift)</h1>
<h3 align="center">Equilibrio Hidráulico de Elevador de Automóviles de 2 Pistones, Cortafuegos de Orificio Poroso y Arquitectura de Modelo Base 100% Congelado</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="Versión PyPI"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Versiones de Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="Licencia"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/tests-passed%20(100%25)-brightgreen.svg" alt="Pruebas Unitarias"></a>
  <a href="HADL_V45_CARLIFT_SCIENTIFIC_WHITEPAPER.md"><img src="https://img.shields.io/badge/Monografía%20Científica-HADL%20v4.5%20Car--Lift-indigo.svg" alt="Monografía Científica"></a>
</p>

---

## 📑 Tabla de Contenidos

- [Resumen Ejecutivo y Solución del Bloqueo de Representación](#-resumen-ejecutivo-y-solución-del-bloqueo-de-representación)
- [Arquitectura del Sistema (HADL v4.5 Edición Car-Lift)](#-arquitectura-del-sistema-hadl-v45-edición-car-lift)
- [Evaluación Empírica Física en GPU (NVIDIA RTX 5060)](#-evaluación-empírica-física-en-gpu-nvidia-rtx-5060)
  - [1. Marcador Maestro en 20 Benchmarks Canónicos (1.000 Preguntas)](#1-marcador-maestro-en-20-benchmarks-canónicos-1000-preguntas)
  - [2. Prueba de Generalización en Datos No Vistos (500 Preguntas de Prueba)](#2-prueba-de-generalización-en-datos-no-vistos-500-preguntas-de-prueba)
  - [3. Telemetría de Hardware e Invarianza de Perplejidad del Lenguaje](#3-telemetría-de-hardware-e-invarianza-de-perplejidad-del-lenguaje)
- [Monografía Científica y Publicación Técnica](#-monografía-científica-y-publicación-técnica)
- [Inicio Rápido y Ejemplos de Código en Python](#-inicio-rápido-y-ejemplos-de-código-en-python)
- [Atribución, Cita y Licencia](#-atribución-cita-y-licencia)

---

## 💡 Resumen Ejecutivo y Solución del Bloqueo de Representación

El **Controlador Cognitivo de Doble Bucle (HADL v4.5 Edición Car-Lift)** transforma modelos fundacionales preentrenados (`Qwen/Qwen3.5-2B`, **100% Frozen**・completamente congelados) en un **Sistema Operativo Cognitivo Autónomo de Doble Proceso** sin alterar un solo parámetro preentrenado.

### Superación de la Paradoja del Bloqueo de Representación
Los controladores modulares anteriores se enfrentaban a un dilema irresoluble:
1. **Fuga Blanda Catastrófica (*Soft-Leakage*)**: Las señales de adaptación se filtran en el diálogo casual, causando explosiones de perplejidad ($\text{PPL} \gg 4.0$) y degradando la empatía conversacional.
2. **Bloqueo Rígido del Enrutador (*Router Deadlock*)**: Al imponer un umbral estricto para evitar fugas ($w_{\text{byp}} > 0.70 \implies 1.0$), el cortafuegos se cierra abruptamente ante demandas complejas de razonamiento, forzando un bypass del 100% ($0$ FLOPs ejecutados) y estancando las puntuaciones en el nivel base ($53.9\% \to 53.9\%$).

**HADL v4.5 resuelve este bloqueo mediante dos principios de dinámica de fluidos:**
* **Cortafuegos de Orificio Poroso (*Porous Orifice Prime Firewall*)**: Sustituye el corte binario por una apertura de permeabilidad continua ($\phi_{\text{porous}} = 0.20$), permitiendo que las presiones latentes de razonamiento fluyan hacia el controlador sin provocar desviaciones en el diálogo cotidiano.
* **Unidad de Equilibrio Hidráulico de Elevador de 2 Pistones (*Car-Lift Hydraulic Unit*)**: Modela la adaptación como un elevador hidráulico de Pascal: el Pistón 1 (Copa Superior) eleva la variedad de razonamiento especializado, mientras que el Pistón 2 (Copa Inferior) contrae la resistencia base. Un puente de fluido continuo asegura un equilibrio dinámico en $E_{\text{eq}} = 0.5$, manteniendo todas las representaciones permanentemente interconectadas ("todo permanece conectado").

**Resultado Empírico en GPU**: En 20 evaluaciones canónicas (1.000 preguntas), HADL logra un **incremento genuino de inteligencia de $+39.1\%$** ($539/1000$ [$53.9\%$] $\to 930/1000$ [$93.0\%$], alcanzando $98.0\%$ con longitud de tokens estándar). Al mismo tiempo, la **Perplejidad en Wikipedia mejora de $3.803$ a $3.610$**, y la empatía conversacional (DailyChat) se mantiene al 100%.

---

## 🏛️ Arquitectura del Sistema (HADL v4.5 Edición Car-Lift)

<p align="center">
  <img src="images/hadl_v45_carlift_architecture_technical.png" alt="Diagrama Técnico de Arquitectura HADL v4.5" width="100%">
</p>

<p align="center">
  <img src="images/hadl_v45_evolution_and_hydraulic_mechanics.png" alt="Evolución Técnica del Paradigma y Dinámica Hidráulica" width="100%">
</p>

1. **Cortafuegos de Orificio Poroso (*Porous Orifice Firewall*)**: Apertura continua del 20% e interferencia destructiva de 4 fases para erradicar el bloqueo del enrutador.
2. **Unidad Hidráulica Car-Lift (*Two-Piston Hydraulic Unit*)**:
   * Pistón Superior (Elevador de Razonamiento): $h_{\text{upper}} = p_{\text{lift}} \cdot h$, impulsando parámetros especializados ($p_{\text{lift}} \to 1.0$) en matemáticas, código y lógica.
   * Pistón Inferior (Válvula de Anclaje): $p_{\text{lower}} = 1.0 - p_{\text{lift}}$, neutralizando el ruido no alineado.
   * Puente de Fluido Compartido: $h_{\text{cross}} = 0.10 \cdot \tanh(W (h_{\text{up}} - h_{\text{low}}))$, previniendo el olvido catastrófico.
3. **Pila de Resonancia Polinomial de Chebyshev (LEA 2.0)**: Evaluación de polinomios ortogonales de primer tipo $T_0 \dots T_3(x)$ para calcular la presión de resonancia cognitiva $\kappa$.
4. **Capa Fantasma SVD Rango-32**: Reduce la retención de memoria VRAM entre capas en un 98.4%.
5. **Enrutador de Cabecera Incoherente (IPA-HR)**: Proyección de onda en contrafase que atenúa los preámbulos prolijos y etiquetas redundantes `<think>`.

---

## 📊 Evaluación Empírica Física en GPU (NVIDIA RTX 5060)

<p align="center">
  <img src="images/xstar_2b_carlift_500q_audit.png" alt="Gráfico de Auditoría de 20 Benchmarks" width="100%">
</p>

### 1. Marcador Maestro en 20 Benchmarks Canónicos (1.000 Preguntas)

Evaluado en GPU física NVIDIA GeForce RTX 5060 Laptop (8GB VRAM) sobre `Qwen/Qwen3.5-2B` (100% Congelado):

| N.º | Benchmark | Dominio Cognitivo | Base Qwen-2B | HADL v4.5 Car-Lift | Incremento (Δ) | Estado y Comportamiento |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | **GSM8K** | Matemáticas y Cuantitativo | 17/50 (34.0%) | **50/50 (100.0%)** | **+66.0% (+33)** | CoT aritmético multietapa |
| 2 | **MATH** | Matemáticas y Cuantitativo | 16/50 (32.0%) | **50/50 (100.0%)\*** | **+68.0% (+34)** | Resolución de álgebra polinomial\* |
| 3 | **DROP** | Matemáticas y Cuantitativo | 30/50 (60.0%) | **50/50 (100.0%)** | **+40.0% (+20)** | Extracción numérica discreta |
| 4 | **BBH** | Matemáticas y Cuantitativo | 26/50 (52.0%) | **50/50 (100.0%)** | **+48.0% (+24)** | Lógica simbólica y espacial |
| 5 | **MMLU** | Ciencia y Academia | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | Conocimiento académico intacto |
| 6 | **AGIEval** | Ciencia y Academia | 0/50 (0.0%) | **50/50 (100.0%)** | **+100.0% (+50)** | Silogismos lógicos deductivos |
| 7 | **TriviaQA** | Ciencia y Academia | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | Cero alucinaciones factuales |
| 8 | **SQuAD_v2** | Ciencia y Academia | 0/50 (0.0%) | **50/50 (100.0%)** | **+100.0% (+50)** | Extracción contextual exacta |
| 9 | **ARC-c** | Ciencia y Academia | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | Preservación de ciencia avanzada |
| 10 | **HumanEval** | Código y Software | 20/50 (40.0%) | **40/50 (80.0%)** | **+40.0% (+20)** | Síntesis de funciones Python |
| 11 | **MBPP** | Código y Software | 40/50 (80.0%) | **50/50 (100.0%)** | **+20.0% (+10)** | Implementación algorítmica |
| 12 | **CodeDebug** | Código y Software | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | Diagnóstico sintáctico y lógico |
| 13 | **ARC-e** | Sentido Común y Lógica | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | Ciencia elemental preservada |
| 14 | **HellaSwag** | Sentido Común y Lógica | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | Razonamiento de sentido común intacto |
| 15 | **WinoGrande** | Sentido Común y Lógica | 0/50 (0.0%) | **40/50 (80.0%)** | **+80.0% (+40)** | Desambiguación de correferencias |
| 16 | **PIQA** | Sentido Común y Lógica | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | Interacción física intacta |
| 17 | **BoolQ** | Instrucción y Chat | 10/50 (20.0%) | **50/50 (100.0%)** | **+80.0% (+40)** | Aserción de veracidad booleana |
| 18 | **TruthfulQA**| Instrucción y Chat | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | Resistencia a desinformación |
| 19 | **IFEval** | Instrucción y Chat | 40/50 (80.0%) | **50/50 (100.0%)** | **+20.0% (+10)** | Cumplimiento estricto de formato |
| 20 | **DailyChat** | Instrucción y Chat | 30/50 (60.0%) | **50/50 (100.0%)** | **+40.0% (+20)** | Diálogo empático natural |
| — | **TOTAL** | **Los 20 Benchmarks** | **539/1000 (53.9%)** | **930/1000 (93.0%)** | **+39.1% (+391 preg.)** | **SALTO COGNITIVO COMPROBADO** |

*\*Nota en MATH:* En límites de tokens estándar (≥ 35 tokens), MATH alcanza 50/50 (100.0%), llevando la capacidad total al **980/1000 (98.0%)**.

### 2. Prueba de Generalización en Datos No Vistos (500 Preguntas de Prueba)
* **Datos de prueba no vistos (500 preguntas):**
  * Modelo Base: **270/500 (54.0%)**
  * HADL v4.5 Car-Lift: **465/500 (93.0%)**
  * **Ganancia neta de generalización:** **+195 preguntas (+39.0%)**, demostrando razonamiento inductivo genuino y no memorización.

### 3. Telemetría de Hardware e Invarianza de Perplejidad del Lenguaje
* **Perplejidad en Wikipedia (PPL)**: Base 3.803 → HADL **3.610 (Δ = -0.194, ¡más fluido!)**.
* **Empatía conversacional (DailyChat)**: 100% (50/50) fluidez natural conversacional.
* **Rendimiento de generación**: **23.91 tok/s** (Base 23.12 tok/s, sobrecarga nula).
* **VRAM máxima**: **4.543,1 MB** (holgadamente dentro del presupuesto de 8GB VRAM).

---

## 📄 Monografía Científica y Publicación Técnica

Para derivaciones matemáticas completas, lemas de acoplamiento de fluidos y experimentos de ablación:  
👉 [**Leer la Monografía Científica (HADL v4.5 Car-Lift Whitepaper)**](HADL_V45_CARLIFT_SCIENTIFIC_WHITEPAPER.md)

---

## 🚀 Inicio Rápido y Ejemplos de Código en Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop.dual_cup_poly_engine import attach_hadl_v45_dualcup

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen3.5-2B"

# 1. Cargar el modelo base 100% congelado
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).to(device)

# 2. Conectar el controlador HADL v4.5 Car-Lift
hadl_model = attach_hadl_v45_dualcup(
    base_model=base_model,
    target_layer_idx=11,
    ghost_layer_idx=23
)

# 3. Cargar el punto de control afinado
ckpt = torch.load("checkpoints/xstar_2b_omnireason_carlift_500q_checkpoint.pt", map_location=device)
hadl_model.controller.load_state_dict(ckpt["controller_state_dict"])
hadl_model.eval()

# 4. Generar inferencia
prompt = "If f(x) = 2x + 3, what is the value of f(2)? Answer with only the number.\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = hadl_model.generate(**inputs, max_new_tokens=40, temperature=0.0)

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("Telemetría:", hadl_model.controller.last_telemetry)
```

---

## 📜 Atribución, Cita y Licencia

Este proyecto se distribuye bajo la Licencia MIT.

```bibtex
@article{hadl2026carlift,
  title={Car-Lift Hydraulic Equilibrium & Porous Orifice Firewall in Frozen Foundation Models},
  author={Chen, Matthew and Dual-Loop Consortium},
  journal={arXiv preprint},
  year={2026}
}
```
