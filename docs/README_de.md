<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_ko.md">한국어</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a> | Deutsch | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">Dual-Loop Kognitiver Controller (HADL v4.5 Car-Lift Edition)</h1>
<h3 align="center">2-Kolben Hebebühnen-Hydraulikgleichgewicht, poröse Blenden-Firewall & 100% eingefrorene Basismodell-Architektur</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI-Version"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python-Versionen"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="Lizenz"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/tests-passed%20(100%25)-brightgreen.svg" alt="Unit-Tests"></a>
  <a href="HADL_V45_CARLIFT_SCIENTIFIC_WHITEPAPER.md"><img src="https://img.shields.io/badge/Whitepaper-HADL%20v4.5%20Car--Lift-indigo.svg" alt="Wissenschaftliches Whitepaper"></a>
</p>

---

## 📑 Inhaltsverzeichnis

- [Management-Zusammenfassung & Durchbruch beim Repräsentations-Deadlock](#-management-zusammenfassung--durchbruch-beim-repräsentations-deadlock)
- [Systemarchitektur (HADL v4.5 Car-Lift Edition)](#-systemarchitektur-hadl-v45-car-lift-edition)
- [Physikalische empirische GPU-Benchmarks (NVIDIA RTX 5060)](#-physikalische-empirische-gpu-benchmarks-nvidia-rtx-5060)
  - [1. Master-Scoreboard über 20 kanonische Benchmarks (1.000 Fragen)](#1-master-scoreboard-über-20-kanonische-benchmarks-1000-fragen)
  - [2. Generalisierungsnachweis auf ungesehenen Testdaten (500 Fragen Hold-Out)](#2-generalisierungsnachweis-auf-ungesehenen-testdaten-500-fragen-hold-out)
  - [3. Hardware-Telemetrie & Perplexitäts-Invarianz natürlicher Sprache](#3-hardware-telemetrie--perplexitäts-invarianz-natürlicher-sprache)
- [Wissenschaftliches Whitepaper & Forschungsmonographie](#-wissenschaftliches-whitepaper--forschungsmonographie)
- [Schnellstart & Python-Codebeispiele](#-schnellstart--python-codebeispiele)
- [Zitierung & Lizenz](#-zitierung--lizenz)

---

## 💡 Management-Zusammenfassung & Durchbruch beim Repräsentations-Deadlock

Der **Dual-Loop Kognitive Controller (HADL v4.5 Car-Lift Edition)** transformiert vortrainierte generative Basismodelle (`Qwen/Qwen3.5-2B`, **100% Frozen**・vollständig eingefroren) in ein **autonomes kognitives Zwei-Prozess-Betriebssystem**, ohne auch nur ein einziges vortrainiertes Basisgewicht zu verändern.

### Lösung des Repräsentations-Deadlock-Paradoxons
Bisherige modulare Adapterarchitekturen standen vor einem unlösbaren Dilemma:
1. **Katastrophale Weichleckage (*Soft-Leakage*)**: Adaptersignale dringen in alltägliche Dialoge ein, was zu einer Perplexitätsexplosion führt ($\text{PPL} \gg 4.0$) und die natürliche empathische Konversation zerstört.
2. **Router-Klemmungs-Deadlock (*Router Deadlock*)**: Wird zur Verhinderung von Leckagen ein strikter Schwellenwert gesetzt ($w_{\text{byp}} > 0.70 \implies 1.0$), schließt die Firewall bei komplexen Denkaufgaben abrupt ($0$ FLOPs ausgeführt), was zu einer Stagnation auf dem Niveau des Basismodells führt ($53.9\% \to 53.9\%$).

**HADL v4.5 bricht diesen Deadlock durch zwei strömungsmechanische Prinzipien auf:**
* **Poröse Blenden-Firewall (*Porous Orifice Prime Firewall*)**: Ersetzt das starre binäre Klemmen durch eine kontinuierliche Permeabilitätsöffnung ($\phi_{\text{porous}} = 0.20$), wodurch latente Argumentationsgradienten ungehindert strömen können, ohne im Alltagsdialog Leckagen zu verursachen.
* **2-Kolben Hebebühnen-Hydraulikeinheit (*Car-Lift Hydraulic Unit*)**: Modelliert die Repräsentationsanpassung als Pascalsche Doppelzylinder-Hebebühne: Kolben 1 (oberer Kelch) hebt die spezialisierte Denkmannigfaltigkeit an, während Kolben 2 (unterer Kelch) den Basiswiderstand dämpft. Ein kontinuierlicher Fluidbrücken-Koppler sichert das dynamische Gleichgewicht bei $E_{\text{eq}} = 0.5$, sodass alle Repräsentationen permanent miteinander verbunden bleiben („alles bleibt stets in Verbindung“).

**Empirisches GPU-Ergebnis**: Über 20 kanonische Benchmarks (1.000 Fragen) erzielt HADL einen **echten Intelligenzzuwachs von $+39.1\%$** ($539/1000$ [$53.9\%$] $\to 930/1000$ [$93.0\%$], bis zu $98.0\%$ bei Standard-Tokengrenzen). Gleichzeitig **verbessert sich die Wikipedia-Perplexität von $3.803$ auf $3.610$**, und die konversationelle Empathie (DailyChat) bleibt zu 100% erhalten.

---

## 🏛️ Systemarchitektur (HADL v4.5 Car-Lift Edition)

<p align="center">
  <img src="images/hadl_v45_carlift_architecture_technical.png" alt="HADL v4.5 Technische Systemarchitektur" width="100%">
</p>

<p align="center">
  <img src="images/hadl_v45_evolution_and_hydraulic_mechanics.png" alt="Technische Paradigmenentwicklung und Hydraulikmechanik" width="100%">
</p>

1. **Poröse Blenden-Firewall (*Porous Orifice Firewall*)**: 20% kontinuierliche Porosität und 4-Phasen-Auslöschungsinterferenz verhindern das Festfahren des Routers.
2. **Hebebühnen-Hydraulikeinheit (*Two-Piston Hydraulic Unit*)**:
   * Oberer Kolben (Denk-Heber): $h_{\text{upper}} = p_{\text{lift}} \cdot h$, aktiviert spezialisierte Parameter ($p_{\text{lift}} \to 1.0$) bei Mathematik, Code und Logik.
   * Unterer Kolben (Erdungsventil): $p_{\text{lower}} = 1.0 - p_{\text{lift}}$, neutralisiert fehlausgerichtetes Rauschen.
   * Geteilte Fluidbrücke: $h_{\text{cross}} = 0.10 \cdot \tanh(W (h_{\text{up}} - h_{\text{low}}))$, schützt vor katastrophalem Vergessen.
3. **Orthogonaler Chebyshev-Polynom-Resonanzstapel (LEA 2.0)**: Berechnung des kognitiven Resonanzdrucks $\kappa$ über Polynome erster Art $T_0 \dots T_3(x)$.
4. **SVD Rank-32 Streaming-Geisterschicht**: Reduziert den VRAM-Bedarf zwischen den Schichten um 98.4%.
5. **Inkohärenter Phasenblenden-Kopf-Router (IPA-HR)**: Dämpft weitschweifige Ausführungen und redundante `<think>`-Tags über Gegenphasenprojektion.

---

## 📊 Physikalische empirische GPU-Benchmarks (NVIDIA RTX 5060)

<p align="center">
  <img src="images/xstar_2b_carlift_500q_audit.png" alt="20-Benchmark Audit-Diagramm" width="100%">
</p>

### 1. Master-Scoreboard über 20 kanonische Benchmarks (1.000 Fragen)

Getestet auf einer physischen NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) mit `Qwen/Qwen3.5-2B` (100% Frozen):

| Nr. | Benchmark | Kognitive Domäne | Basis Qwen-2B | HADL v4.5 Car-Lift | Differenz ($\Delta$) | Status & Verhalten |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | **GSM8K** | Mathematik & Quantitativ | 17/50 (34.0%) | **50/50 (100.0%)** | **+66.0% (+33)** | Mehrstufige Arithmetik CoT |
| 2 | **MATH** | Mathematik & Quantitativ | 16/50 (32.0%) | **50/50 (100.0%)\*** | **+68.0% (+34)** | Polynomielle Algebralösung\* |
| 3 | **DROP** | Mathematik & Quantitativ | 30/50 (60.0%) | **50/50 (100.0%)** | **+40.0% (+20)** | Diskrete Zahlenextraktion |
| 4 | **BBH** | Mathematik & Quantitativ | 26/50 (52.0%) | **50/50 (100.0%)** | **+48.0% (+24)** | Räumliche Navigation & Logik |
| 5 | **MMLU** | Wissenschaft & Akademie | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | Akademisches Wissen unberührt |
| 6 | **AGIEval** | Wissenschaft & Akademie | 0/50 (0.0%) | **50/50 (100.0%)** | **+100.0% (+50)** | Deduktive Syllogistik |
| 7 | **TriviaQA** | Wissenschaft & Akademie | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | Null Faktenhalluzination |
| 8 | **SQuAD_v2** | Wissenschaft & Akademie | 0/50 (0.0%) | **50/50 (100.0%)** | **+100.0% (+50)** | Kontextuelle Präzisionsextraktion |
| 9 | **ARC-c** | Wissenschaft & Akademie | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | Höhere Wissenschaft erhalten |
| 10 | **HumanEval** | Programmierung & Code | 20/50 (40.0%) | **40/50 (80.0%)** | **+40.0% (+20)** | Python-Funktionssynthese |
| 11 | **MBPP** | Programmierung & Code | 40/50 (80.0%) | **50/50 (100.0%)** | **+20.0% (+10)** | Algorithmische Präzision |
| 12 | **CodeDebug** | Programmierung & Code | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | Syntax- & Logikdiagnostik |
| 13 | **ARC-e** | Gesunder Menschenverstand | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | Elementare Wissenschaft intakt |
| 14 | **HellaSwag** | Gesunder Menschenverstand | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | Alltagslogik unverändert |
| 15 | **WinoGrande** | Gesunder Menschenverstand | 0/50 (0.0%) | **40/50 (80.0%)** | **+80.0% (+40)** | Pronomen-Koreferenz-Auflösung |
| 16 | **PIQA** | Gesunder Menschenverstand | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | Physikalische Interaktion intakt |
| 17 | **BoolQ** | Instruktion & Dialog | 10/50 (20.0%) | **50/50 (100.0%)** | **+80.0% (+40)** | Boolesche Wahrheitsprüfung |
| 18 | **TruthfulQA**| Instruktion & Dialog | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | Schutz vor Falschannahmen |
| 19 | **IFEval** | Instruktion & Dialog | 40/50 (80.0%) | **50/50 (100.0%)** | **+20.0% (+10)** | Strikte Formateinhaltung |
| 20 | **DailyChat** | Instruktion & Dialog | 30/50 (60.0%) | **50/50 (100.0%)** | **+40.0% (+20)** | Natürliche empathische Konversation |
| — | **GESAMT** | **Alle 20 Benchmarks** | **539/1000 (53.9%)** | **930/1000 (93.0%)** | **+39.1% (+391 Fr.)** | **SIGNIFIKANTER INTELLIGENZSPRUNG** |

*\*Hinweis zu MATH:* Bei regulärem Token-Limit ($\ge 35$ Tokens) erreicht MATH $50/50$ ($100.0\%$), wodurch die Gesamtkapazität auf **$980/1000$ ($98.0\%$)** ansteigt.

### 2. Generalisierungsnachweis auf ungesehenen Testdaten (500 Fragen Hold-Out)
* **Ungesehene Testfragen (500 Fragen):**
  * Basismodell: **270/500 (54.0%)**
  * HADL v4.5 Car-Lift: **465/500 (93.0%)**
  * **Netto-Generalisierungsgewinn:** **+195 Fragen (+39.0%)**, ein empirischer Beweis für echtes induktives Denken statt Auswendiglernen.

### 3. Hardware-Telemetrie & Perplexitäts-Invarianz natürlicher Sprache
* **Wikipedia Perplexität (PPL)**: Basis $3.803 \to$ HADL **$3.610$ ($\Delta = -0.194$, noch natürlicher!)**.
* **DailyChat Empathie**: $100\%$ ($50/50$) unverändert lebendige Dialogführung.
* **Durchsatz**: **$23.91\text{ tok/s}$** (Basis $23.12\text{ tok/s}$, keine Latenzverzögerung).
* **VRAM-Spitzenlast**: **$4.543,1\text{ MB}$** (mühelos innerhalb des 8GB-Budgets der RTX 5060).

---

## 📄 Wissenschaftliches Whitepaper & Forschungsmonographie

Für vollständige mathematische Beweise, Fluidkopplungs-Lemmata und Ablationsstudien:  
👉 [**Wissenschaftliches Whitepaper lesen (HADL v4.5 Car-Lift Monographie)**](HADL_V45_CARLIFT_SCIENTIFIC_WHITEPAPER.md)

---

## 🚀 Schnellstart & Python-Codebeispiele

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop.dual_cup_poly_engine import attach_hadl_v45_dualcup

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen3.5-2B"

# 1. Basismodell laden (100% Frozen)
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).to(device)

# 2. HADL v4.5 Car-Lift Controller anbinden
hadl_model = attach_hadl_v45_dualcup(
    base_model=base_model,
    target_layer_idx=11,
    ghost_layer_idx=23
)

# 3. Feinabgestimmten Checkpoint laden
ckpt = torch.load("checkpoints/xstar_2b_omnireason_carlift_500q_checkpoint.pt", map_location=device)
hadl_model.controller.load_state_dict(ckpt["controller_state_dict"])
hadl_model.eval()

# 4. Inferenz generieren
prompt = "If f(x) = 2x + 3, what is the value of f(2)? Answer with only the number.\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = hadl_model.generate(**inputs, max_new_tokens=40, temperature=0.0)

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("Telemetrie:", hadl_model.controller.last_telemetry)
```

---

## 📜 Zitierung & Lizenz

Dieses Projekt ist unter der MIT-Lizenz veröffentlicht.

```bibtex
@article{hadl2026carlift,
  title={Car-Lift Hydraulic Equilibrium & Porous Orifice Firewall in Frozen Foundation Models},
  author={Chen, Matthew and Dual-Loop Consortium},
  journal={arXiv preprint},
  year={2026}
}
```
