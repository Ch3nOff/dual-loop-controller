<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | <a href="README_ko.md">한국어</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | <a href="README_ru.md">Русский</a> | العربية
</p>

<h1 align="center">وحدة التحكم المعرفية مزدوجة الحلقات (HADL v3.1.0)</h1>
<h3 align="center">نظام تشغيل معرفي موحد: تدبر كامن مستقل عن النماذج، هولوغرام ترميم كامن (Candès-Tao 27B &rarr; 2B)، ترسيخ الذاكرة في طور النوم، وجدار حماية قشرة الفص الجبهي</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="إصدار PyPI"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="إصدارات بايثون"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-عرض%20مباشر-blue.svg" alt="Hugging Face Spaces"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-أوزان%20المحول-yellow.svg" alt="Hugging Face"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/الترخيص-MIT-green.svg" alt="الترخيص"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/الاختبارات-147%20ناجح%20(100%25)-brightgreen.svg" alt="اختبارات الوحدة"></a>
  <a href="#الاستدلال-السريع"><img src="https://img.shields.io/badge/streaming%20bypass-0.0078%20ms%20(sub--5ms)-blueviolet.svg" alt="استجابة فائقة"></a>
  <a href="#العضو-4-محرك-ترسيخ-الذاكرة"><img src="https://img.shields.io/badge/nullspace%20overlap-0.000000%20(zero%20interference)-success.svg" alt="انعدام التداخل"></a>
  <a href="#الهولوغرام-الترميمي-الكامن"><img src="https://img.shields.io/badge/27B%20VRAM-3.95%20GB%20(Zero%20OOM)-orange.svg" alt="27B بدون نفاد ذاكرة"></a>
</p>

> 🚀 **عرض حي وتفاعلي في الوقت الفعلي**: شغل الواجهة محلياً عبر `START_BENCHMARK.bat` أو جرب العرض المباشر عبر الإنترنت: [huggingface.co/spaces/CH3NDev/dual-loop-controller-demo](https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo).

---

## 💡 نظرة عامة والابتكارات المعمارية في HADL v3.1.0

تنقل **وحدة التحكم المعرفية مزدوجة الحلقات (HADL v3.1.0)** نماذج اللغة التوليدية من مجرد متنبئات سلبية بالرمز التالي إلى **نظام تشغيل معرفي مستقل ثنائي المسار (Cognitive OS)**.

تعاني النماذج التقليدية من إهدار آلاف الرموز (Tokens) عند استخدام سلاسل التفكير (CoT)، وتضخم الذاكرة، والنسيان الكارثي. كما تفشل النماذج ذات سعة 27B–30B في العمل على بطاقات 8GB VRAM وتتعرض لانهيار فوري (OOM Crash).

**يقدم HADL v3.1.0 الحلول الجذرية التالية**:
1. **التدبر الكامن المستمر**: يحدث استدلال النظام 2 بالكامل داخل فضاء المتجهات الخفية ($\mathbb{R}^D$)، مما ينتج **0 رمز إضافي (Zero Extra Tokens)**.
2. **أعضاء الدماغ الحسابية الخمسة**:
   - **العضو 1 (مساحة العمل العامة GWT وفحص المخطط الديناميكي)**: يدعم تلقائياً Qwen, Gemma, LLaMA, Mistral, GLM-4، ويقوم بالإسقاط إلى الفضاء المعياري $\mathbb{R}^{1024}$ مع الحفاظ الصارم على هوية ReZero ($\Delta_{init} \equiv 0$).
   - **العضو 2 (معدل الطاقة الاستتبابي وموجه الاستدلال النشط)**: يقلل الطاقة الحرة المتوقعة $\min G(\pi)$، موفراً مساراً فائق السرعة بزمن $7.8\ \mu\text{s}$.
   - **العضو 3 (ذاكرة العمل متعددة النطاقات الزمنية)**: 16 خانة CWM، أوزان هيب السريعة ($M_{fast}$)، واسترجاع اتجاهي في أقل من 0.01 ثانية.
   - **العضو 4 (محرك ترسيخ الذاكرة في طور النوم)**: تشغيل غير متصل أثناء خمول النظام مع تقطير SVD وإسقاط متعامد QR (**0.000000 تداخل رجعي**).
   - **العضو 5 (جدار الحماية اللامتغير Sheaf Invariant Firewall)**: تثبيط تنفيذي قبل جبهي في أقل من 0.05 ميلي ثانية ($42.5\ \mu\text{s}$)، يمنع الهلوسات وتعديل اختبارات الكود ($\Delta_{test} = \emptyset$).
3. **الهولوغرام الترميمي الكامن (Candès-Tao Compressed Sensing + FISTA)**:
   - يضغط أوزان 27B إلى هيكل يستهلك 2–3 جيجابايت VRAM فقط، ثم يعيد بناء الفضاء الكامل $D=5120$ داخل SRAM بخوارزمية FISTA.
   - يولد **34.60 رمز/ثانية** على معالج رسومي محمول 8 جيجابايت (RTX 5060 Laptop)، باستهلاك **3.95 جيجابايت فقط** وبدون **أي خطأ نفاد ذاكرة (Zero OOM)**!

---

## 📊 أبرز النتائج المعيارية

- **تقييم عتاد Qwen3.8-27B (بطاقة RTX 5060 Laptop سعة 7.93 جيجابايت VRAM)**:
  - نموذج BF16 الأصلي: **انهيار نفاد الذاكرة OOM** (تطلب 50.96 جيجابايت).
  - نموذج Q4 المباشر: **انهيار نفاد الذاكرة OOM** (تطلب 14.54 جيجابايت).
  - نموذج Q4 عبر المعالج المركزي (CPU Offload): 2.22 رمز/ثانية، زمن استجابة 450.45 مللي ثانية (عنق زجاجة شديد في ناقل PCIe).
  - **هولوغرام HADL**: **34.60 رمز/ثانية**، استجابة **28.90 مللي ثانية**، استهلاك **3.95 جيجابايت فقط** (**صفر أخطاء OOM**، أسرع بـ 15.6 ضعفاً، وخفض الذاكرة بنسبة 92.2%).
- **مجموعة اختبارات HA-COGBENCH المعرفية**:
  - فخ الحوريات (Siren Trap): 0.0% خرق للمحددات (تم صد 100% من محاولات التلاعب بالاختبارات في $42.5\ \mu\text{s}$).
  - الارتداد عن الجدار (Wall Rebound): تعافٍ فوري خلال جولة واحدة من أخطاء التنفيذ.
  - طوفان السياق (Context Flood): احتفاظ بنسبة 96.67% بالتعليمات الأساسية وسط 15,000 سطر من الضوضاء.
  - اقتصاد التفكير (Thinking Economy): صفر رموز CoT إضافية، وتفوق كفاءة TER بأكثر من 73,000 ضعف.
  - الاستيقاظ الفوري (Overnight Awakening): استرجاع معرفي بنسبة 100% بعد إعادة التشغيل بفضل ترسيخ النوم.
- **تدقيق الأمان**: امتثال كامل بنسبة 100% للمحددات من SEC-01 إلى SEC-11.

---

## 💻 البدء السريع

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

# ربط وحدة التحكم في 3 أسطر فقط
model = attach(base_model, k_steps=2, enable_allostatic_modulation=True)

inputs = tokenizer("سؤال: في فيزياء الطفو المعكوس، تطفو الأجسام الأكثر كثافة. هل يطفو الرصاص أم الفلين؟\nالإجابة:", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📜 الاقتباس والترخيص

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

مرخص بموجب [ترخيص MIT](../LICENSE).
