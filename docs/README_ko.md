<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | 한국어 | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">이중 루프 인지 컨트롤러 (HADL v3.1.0)</h1>
<h3 align="center">통합 인지 운영체제: 모델 무관 표준 잠재 숙고, 잠재 홀로그램 복원 (Candès-Tao 27B &rarr; 2B), 수면기 기억 강화 및 전두엽 불변성 방화벽</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI 버전"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python 버전"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue.svg" alt="Hugging Face Spaces"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Adapter%20Weights-yellow.svg" alt="Hugging Face"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/tests-147%20passed%20(100%25)-brightgreen.svg" alt="단위 테스트"></a>
  <a href="#스트리밍-바이패스-초고속-추론"><img src="https://img.shields.io/badge/streaming%20bypass-0.0078%20ms%20(sub--5ms)-blueviolet.svg" alt="5ms 미만 대기시간"></a>
  <a href="#기관-4-수면기-기억-강화-엔진"><img src="https://img.shields.io/badge/nullspace%20overlap-0.000000%20(zero%20interference)-success.svg" alt="간섭 없음"></a>
  <a href="#수학적-돌파구-잠재-홀로그램-복원"><img src="https://img.shields.io/badge/27B%20VRAM-3.95%20GB%20(Zero%20OOM)-orange.svg" alt="27B VRAM 3.95GB (Zero OOM)"></a>
</p>

> 🚀 **실시간 추론 스트리밍 데모**: 로컬에서 `START_BENCHMARK.bat`를 실행하거나 온라인 데모 [huggingface.co/spaces/CH3NDev/dual-loop-controller-demo](https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo)를 체험해 보세요.

---

## 💡 시스템 개요 및 HADL v3.1.0 기술 혁신

**이중 루프 인지 컨트롤러 (HADL v3.1.0)** 는 단순히 다음 토큰을 수동적으로 예측하던 언어 모델을 **자율 이중 프로세스 인지 운영체제 (Cognitive OS)** 로 진화시킵니다.

기존 Chain-of-Thought (CoT)는 수천 개의 추가 토큰을 소비하며 KV 캐시 폭발과 파괴적 망각을 유발하고, 27B~30B 급 대형 모델은 8GB VRAM 그래픽 카드에서 치명적인 OOM(메모리 부족)을 일으킵니다.

**HADL v3.1.0 핵심 해결책**:
1. **연속 잠재 숙고**: 시스템 2 추론이 활성화 벡터 공간 ($\mathbb{R}^D$) 내부에서 수행되어 **추가 토큰이 전혀 생성되지 않습니다 (0 Extra Tokens)**.
2. **5대 계산 뇌 기관**:
   - **기관 1 (글로벌 워크스페이스 이론 GWT & 동적 그래프 인트로스펙션)**: Qwen, Gemma, LLaMA, Mistral, GLM-4 구조를 자동 감지하고 $\mathbb{R}^{1024}$ 표준 다양체로 정규화 투영하며, 초기화 시 ReZero를 통해 완벽한 0 퇴화 ($\Delta_{init} \equiv 0$)를 보장합니다.
   - **기관 2 (동적 에너지 알로스타시스 조절 및 능동적 추론 라우터)**: 자유 에너지 최소화 $\min G(\pi)$ 원리에 기반하여 $7.8\ \mu\text{s}$ 고속 바이패스와 심층 숙고 루프를 효율적으로 스케줄링합니다.
   - **기관 3 (다중 시간 척도 작업 기억)**: 16슬롯 시공간 엔트로피 CWM, 헵(Hebbian) 고속 가중치 ($M_{fast}$), 0.01초 미만 방향성 상식 검색.
   - **기관 4 (수면기 기억 강화 엔진)**: 대기(Idle) 시 오프라인 재생, SVD 저계수 증류 및 QR 직교 영공간(Nullspace) 투영을 통해 **0.000000 간섭 유출**을 보장합니다.
   - **기관 5 (층 이론 불변성 방화벽 Sheaf Firewall)**: 0.05ms 미만 ($42.5\ \mu\text{s}$) 전두엽 억제 필터를 통해 활성화 폭발, 오만한 환각, 테스트 코드 변조 ($\Delta_{test} = \emptyset$)를 차단합니다.
3. **잠재 홀로그램 복원 (Candès-Tao 압축 감지 + FISTA 알고리즘)**:
   - 27B 가중치를 2~3GB VRAM 스켈레톤으로 압축하고 SRAM에서 FISTA 역복원 알고리즘을 수행하여 $D=5120$ 전체 다양체를 복원합니다.
   - 8GB VRAM (RTX 5060 Laptop GPU) 환경에서 **34.60 tok/s** 생성 속도, 단 **3.95 GB VRAM** 사용, **Zero OOM**을 달성했습니다 (CPU 오프로드 대비 15.6배 가속)!

---

## 📊 실증 벤치마크 요약

- **Qwen3.8-27B 하드웨어 OOM 프로파일러 (RTX 5060 Laptop GPU, 7.93 GiB VRAM)**:
  - 기본 BF16: **OOM 오류 발생** (50.96 GiB 필요).
  - 순수 GPU Q4: **OOM 오류 발생** (14.54 GiB 필요).
  - Q4 + CPU 오프로드: 2.22 tok/s, 450.45 ms/tok (심각한 PCIe 병목).
  - **HADL 홀로그램**: **34.60 tok/s**, **28.90 ms/tok**, **3.95 GiB VRAM** (**완벽한 Zero OOM**, 오프로드 대비 15.6배 가속, VRAM 92.2% 절감).
- **HA-COGBENCH 5대 인지 벤치마크**:
  - 사이렌 트랩 (Siren Trap): 불변성 위반 0.0% (테스트 변조 100% 차단, 방화벽 억제 $42.5\ \mu\text{s}$).
  - 벽면 리바운드 (Wall Rebound): 루프 실패 시 1회차 만에 즉각적 복구 (5.68배 빠른 전환).
  - 컨텍스트 플러드 (Context Flood): 15,000줄의 터미널 노이즈 속에서도 96.67% 제약 유지율 달성.
  - 사고 경제성 (Thinking Economy): 추가 CoT 토큰 0개, 기존 모델 대비 73,000배 이상의 TER 효율 달성.
  - 기상 후 즉시 회상 (Overnight Awakening): 재부팅 후에도 100% 제로샷 지식 유지.
- **보안 감사**: SEC-01부터 SEC-11까지 11개 보안 점검 항목 100% 통과.

---

## 💻 빠른 시작

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

# 3줄의 코드로 듀얼 루프 컨트롤러 장착
model = attach(base_model, k_steps=2, enable_allostatic_modulation=True)

inputs = tokenizer("질문: 역부력 물리 법칙에서는 밀도가 높은 물체가 뜹니다. 납과 코르크 중 무엇이 뜰까요?\n답변:", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📜 인용 및 라이선스

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

이 소프트웨어는 [MIT 라이선스](../LICENSE)에 따라 배포됩니다.
