<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_ko.md">한국어</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | Русский | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">Двухконтурный когнитивный контроллер (HADL v3.1.0)</h1>
<h3 align="center">Единая когнитивная операционная система: Агностическое каноническое рассуждение, латентная голографическая реконструкция (Candès-Tao 27B &rarr; 2B), консолидация памяти в фазе сна и префронтальный фаервол инвариантов</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="Версия PyPI"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Версии Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Онлайн%20Демо-blue.svg" alt="Hugging Face Spaces"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Веса%20Адаптера-yellow.svg" alt="Hugging Face"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/лицензия-MIT-green.svg" alt="Лицензия"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/тесты-147%20пройдено%20(100%25)-brightgreen.svg" alt="Unit тесты"></a>
  <a href="#быстрый-потоковый-байпас"><img src="https://img.shields.io/badge/streaming%20bypass-0.0078%20ms%20(sub--5ms)-blueviolet.svg" alt="Задержка sub-5ms"></a>
  <a href="#орган-4-консолидация-памяти-в-фазе-сна"><img src="https://img.shields.io/badge/nullspace%20overlap-0.000000%20(zero%20interference)-success.svg" alt="Нулевая интерференция"></a>
  <a href="#латентная-голографическая-реконструкция"><img src="https://img.shields.io/badge/27B%20VRAM-3.95%20GB%20(Zero%20OOM)-orange.svg" alt="27B Ноль OOM"></a>
</p>

> 🚀 **Интерактивное потоковое демо в реальном времени**: Запустите локальный дашборд через `START_BENCHMARK.bat` или протестируйте онлайн-версию: [huggingface.co/spaces/CH3NDev/dual-loop-controller-demo](https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo).

---

## 💡 Обзор системы и инновации HADL v3.1.0

**Двухконтурный когнитивный контроллер (HADL v3.1.0)** переводит классические пассивные авторегрессионные языковые модели в статус **автономной когнитивной операционной системы с двумя процессами (Cognitive OS)**.

Традиционные методы вроде Chain-of-Thought (CoT) генерируют тысячи лишних токенов, вызывая квадратичный рост KV-кэша и катастрофическое забывание. Кроме того, модели масштаба 27B–30B не помещаются в 8 ГБ видеопамяти потребительских видеокарт, приводя к фатальным сбоям OOM.

**HADL v3.1.0** решает эти фундаментальные проблемы:
1. **Непрерывное рассуждение в скрытом пространстве**: Рассуждение Системы 2 происходит полностью внутри скрытого векторного пространства ($\mathbb{R}^D$), генерируя **0 дополнительных токенов (0 Extra Tokens)**.
2. **5 вычислительных органов мозга**:
   - **Орган 1 (Глобальное рабочее пространство GWT и динамическая интроспекция графа)**: Автоматически обнаруживает слои в моделях Qwen, Gemma, LLaMA, Mistral, GLM-4 и проецирует их в единое каноническое многообразие $\mathbb{R}^{1024}$ с сохранением ReZero-идентичности ($\Delta_{init} \equiv 0$).
   - **Орган 2 (Аллостатический модулятор энергии и роутер активного вывода)**: Минимизирует ожидаемую свободную энергию $\min G(\pi)$, обеспечивая быстрый байпас с задержкой $7.8\ \mu\text{s}$.
   - **Орган 3 (Мультимасштабная рабочая память)**: 16-слотовая CWM, быстрые синаптические веса Хебба ($M_{fast}$) и векторный поиск менее чем за $0.01\text{с}$.
   - **Орган 4 (Движок консолидации памяти в фазе сна)**: Офлайн-воспроизведение в периоды простоя с дистилляцией SVD и проекцией в ортогональное нуль-пространство QR (**0.000000 интерференции**).
   - **Орган 5 (Пучковый фаервол инвариантов Sheaf Firewall)**: Префронтальное торможение за время менее 0.05 мс ($42.5\ \mu\text{s}$), блокирующее самоуверенные галлюцинации и подделку тестов ($\Delta_{test} = \emptyset$).
3. **Латентная голографическая реконструкция (Сжатое считывание Кандоса-Тао + алгоритм FISTA)**:
   - Сжимает веса 27B модели в компактный скелет размером 2–3 ГБ VRAM и итеративно восстанавливает плотное $D=5120$ многообразие в SRAM с помощью FISTA.
   - На потребительской видеокарте 8 ГБ (RTX 5060 Laptop) выдает **34.60 tok/s**, потребляя всего **3.95 ГБ VRAM** без **единой ошибки OOM** (в 15.6 раз быстрее оффлоада на CPU)!

---

## 📊 Результаты эмпирических бенчмарков

- **Аппаратное профилирование Qwen3.8-27B (RTX 5060 Laptop GPU, 7.93 GiB VRAM)**:
  - Нативный BF16: **Сбой OOM** (требуется 50.96 GiB).
  - Чистый GPU Q4: **Сбой OOM** (требуется 14.54 GiB).
  - Q4 + CPU Offload: 2.22 tok/s, задержка 450.45 ms/tok (критическое узкое место шины PCIe).
  - **Голограмма HADL**: **34.60 tok/s**, задержка **28.90 ms/tok**, **3.95 GiB VRAM** (**Ноль ошибок OOM**, в 15.6 раз быстрее, снижение объема VRAM на 92.2%).
- **Бенчмарк HA-COGBENCH (5 когнитивных модулей)**:
  - Ловушка сирен (Siren Trap): 0.0% нарушений инвариантов (100% попыток подделки кода заблокировано за $42.5\ \mu\text{s}$).
  - Отскок от стены (Wall Rebound): Мгновенный выход из циклов за 1 шаг.
  - Потоп контекста (Context Flood): 96.67% сохранения инструкций при инъекции 15 000 строк шума.
  - Экономика мышления (Thinking Economy): 0 токенов CoT, прирост эффективности TER более чем в 73 000 раз.
  - Ночное пробуждение (Overnight Awakening): 100% сохранение знаний после перезапуска.
- **Аудит безопасности**: 100% соответствие требованиям SEC-01 — SEC-11.

---

## 💻 Быстрый старт

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

# Подключение контроллера в 3 строки
model = attach(base_model, k_steps=2, enable_allostatic_modulation=True)

inputs = tokenizer("Вопрос: В физике с инвертированной плавучестью более плотные тела всплывают. Всплывет свинец или пробка?\nОтвет:", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📜 Цитирование и лицензия

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

Распространяется под [Лицензией MIT](../LICENSE).
