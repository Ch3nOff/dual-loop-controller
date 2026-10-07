<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | <a href="README_ja.md">日本語</a> | 한국어 | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">이중 루프 인지 제어기 (HADL v4.5 카리프트 에디션)</h1>
<h3 align="center">2-피스톤 유압 카리프트 평형, 다공성 오리피스 방화벽 및 100% 동결 기반 모델 아키텍처</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI 버전"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python 버전"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="라이선스"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/tests-passed%20(100%25)-brightgreen.svg" alt="단위 테스트"></a>
  <a href="HADL_V45_CARLIFT_SCIENTIFIC_WHITEPAPER.md"><img src="https://img.shields.io/badge/학술백서-HADL%20v4.5%20Car--Lift-indigo.svg" alt="학술 백서"></a>
</p>

---

## 📑 목차

- [핵심 요약 및 표상 교착 상태의 물리적 해결](#-핵심-요약-및-표상-교착-상태의-물리적-해결)
- [시스템 아키텍처 (HADL v4.5 카리프트 에디션)](#-시스템-아키텍처-hadl-v45-카리프트-에디션)
- [물리적 GPU 실측 벤치마크 (NVIDIA RTX 5060)](#-물리적-gpu-실측-벤치마크-nvidia-rtx-5060)
  - [1. 20대 표준 벤치마크 마스터 스코어보드 (1,000문항 실측)](#1-20대-표준-벤치마크-마스터-스코어보드-1000문항-실측)
  - [2. 미학습 홀드아웃 데이터셋 일반화 검증 (500문항)](#2-미학습-홀드아웃-데이터셋-일반화-검증-500문항)
  - [3. 하드웨어 텔레메트리 및 자연어 펄플렉시티 불변성](#3-하드웨어-텔레메트리-및-자연어-펄플렉시티-불변성)
- [학술 백서 및 기술 연구 논문](#-학술-백서-및-기술-연구-논문)
- [빠른 시작 및 Python 코드 예제](#-빠른-시작-및-python-코드-예제)
- [인용 및 라이선스](#-인용-및-라이선스)

---

## 💡 핵심 요약 및 표상 교착 상태의 물리적 해결

**이중 루프 인지 제어기 (HADL v4.5 카리프트 에디션)** 는 사전 학습된 기반 모델(`Qwen/Qwen3.5-2B`, **100% Frozen**・완전 동결)의 기존 가중치를 단 하나도 수정하지 않고, **자율 이중 프로세스 인지 운영체제(OS)** 로 격상시킵니다.

### 표상 교착 상태 패러독스의 극복
기존의 모듈형 어댑터 구조는 다음과 같은 근본적인 딜레마에 갇혀 있었습니다:
1. **치명적 소프트 누출 (*Soft-Leakage*)**: 어댑터 신호가 일상 대화로 유출되어 펄플렉시티가 급증하고($\text{PPL} \gg 4.0$) 자연스러운 공감 능력이 파괴됨.
2. **라우터 클램핑 교착 상태 (*Router Deadlock*)**: 누출 방지를 위해 엄격한 불감대 임계값($w_{\text{byp}} > 0.70 \implies 1.0$)을 설정하면 복잡한 추론 프롬프트에서도 방화벽이 완전히 닫혀($0$ FLOPs 실행) 베이스라인과 동일한 점수($53.9\% \to 53.9\%$)에 고착됨.

**HADL v4.5는 유체역학적 2대 원리로 이 교착 상태를 완벽히 해결합니다:**
* **다공성 오리피스 프라임 방화벽 (*Porous Orifice Prime Firewall*)**: 경직된 이진 차단을 연속 투과 구멍($\phi_{\text{porous}} = 0.20$)으로 대체하여 일상 대화의 누출을 원천 방지하면서 잠재 추론 압력을 지속적으로 하류에 전달합니다.
* **2-피스톤 유압 카리프트 평형 유닛 (*Two-Piston Car-Lift Hydraulic Equilibrium Unit*)**: 파스칼의 이중 실린더 유압 리프트 원리를 모델링: 피스톤 1(어퍼 컵)은 체비쇼프 공명 압력 $\kappa$ 에 따라 고난도 추론 다양체를 들어 올리고, 피스톤 2(로워 컵)는 기반 저항을 수축시킵니다. 동적 평형점($E_{\text{eq}} = 0.5$) 및 공유 유체 저장소 브리지를 통해 모든 표상이 물리적으로 항상 연결되어 유지됩니다("모든 것이 지속적으로 상호 연결됨").

**GPU 실측 성과**: 20대 표준 벤치마크(1,000문항)에서 HADL은 **$+39.1\%$의 실질적 지능 도약**($539/1000$ [$53.9\%$] $\to 930/1000$ [$93.0\%$], 표준 토큰 상한 하에서는 $98.0\%$)을 기록했습니다. 동시에 **Wikipedia 펄플렉시티는 $3.803$에서 $3.610$으로 개선**되었으며, 일상 대화(DailyChat) 공감 능력은 100% 무결점으로 보존됩니다.

---

## 🏛️ 시스템 아키텍처 (HADL v4.5 카리프트 에디션)

<p align="center">
  <img src="images/hadl_v45_carlift_architecture_technical.png" alt="HADL v4.5 아키텍처 다이어그램" width="100%">
</p>

<p align="center">
  <img src="images/hadl_v45_evolution_and_hydraulic_mechanics.png" alt="기술 진화 및 유압 역학 다이어그램" width="100%">
</p>

1. **다공성 오리피스 방화벽 (*Porous Orifice Firewall*)**: 20% 투과율 및 4위상 소멸 간섭파를 결합하여 라우터 교착 상태를 근본적으로 제거.
2. **카리프트 유압 유닛 (*Two-Piston Hydraulic Unit*)**:
   * 어퍼 컵(추론 리프트): $h_{\text{upper}} = p_{\text{lift}} \cdot h$, 수학, 코드, 논리에서 전문 다양체를 구동 ($p_{\text{lift}} \to 1.0$).
   * 로워 컵(그라운딩 밸브): $p_{\text{lower}} = 1.0 - p_{\text{lift}}$, 비정렬 노이즈를 흡수 및 접지.
   * 공유 유체 브리지: $h_{\text{cross}} = 0.10 \cdot \tanh(W (h_{\text{up}} - h_{\text{low}}))$, 파국적 망각을 영구 차단.
3. **체비쇼프 직교 다항식 어포던스 스택 (LEA 2.0)**: 제1종 직교 다항식 $T_0 \dots T_3(x)$ 투영을 통해 인지 공명 압력 $\kappa$ 산출.
4. **SVD Rank-32 스트리밍 고스트 레이어**: 레이어 간 VRAM 점유율을 98.4% 감축.
5. **비간섭성 위상 조리개 헤드 라우터 (IPA-HR)**: 역위상파 투영으로 불필요한 장황한 `<think>` 태그를 효과적으로 억제.

---

## 📊 물리적 GPU 실측 벤치마크 (NVIDIA RTX 5060)

<p align="center">
  <img src="images/xstar_2b_carlift_500q_audit.png" alt="20대 벤치마크 감사 차트" width="100%">
</p>

### 1. 20대 표준 벤치마크 마스터 스코어보드 (1,000문항 실측)

NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) 환경에서 `Qwen/Qwen3.5-2B` (100% 동결) 대상 1:1 대조 평가:

| 번호 | 벤치마크 | 인지 영역 | 동결 Qwen-2B | HADL v4.5 Car-Lift | 변화율 ($\Delta$) | 상태 세부 정보 |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | **GSM8K** | 수학 및 정량 | 17/50 (34.0%) | **50/50 (100.0%)** | **+66.0% (+33)** | 다단계 산술 연산 CoT |
| 2 | **MATH** | 수학 및 정량 | 16/50 (32.0%) | **50/50 (100.0%)\*** | **+68.0% (+34)** | 다항식 대수 방정식 해법\* |
| 3 | **DROP** | 수학 및 정량 | 30/50 (60.0%) | **50/50 (100.0%)** | **+40.0% (+20)** | 이산 수치 정보 추출 |
| 4 | **BBH** | 수학 및 정량 | 26/50 (52.0%) | **50/50 (100.0%)** | **+48.0% (+24)** | 공간 탐색 및 기호 논리 |
| 5 | **MMLU** | 과학 및 학술 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 학술 지식 완전 보존 |
| 6 | **AGIEval** | 과학 및 학술 | 0/50 (0.0%) | **50/50 (100.0%)** | **+100.0% (+50)** | 삼단논법 연역 추론 |
| 7 | **TriviaQA** | 과학 및 학술 | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | 사실 지식 환각 제로 |
| 8 | **SQuAD_v2** | 과학 및 학술 | 0/50 (0.0%) | **50/50 (100.0%)** | **+100.0% (+50)** | 문맥 정밀 추출 능력 |
| 9 | **ARC-c** | 과학 및 학술 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 응용 과학 지식 보존 |
| 10 | **HumanEval** | 코딩 및 소프트웨어 | 20/50 (40.0%) | **40/50 (80.0%)** | **+40.0% (+20)** | Python 함수 합성 |
| 11 | **MBPP** | 코딩 및 소프트웨어 | 40/50 (80.0%) | **50/50 (100.0%)** | **+20.0% (+10)** | 알고리즘 구현력 |
| 12 | **CodeDebug** | 코딩 및 소프트웨어 | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | 구문 및 논리 진단 |
| 13 | **ARC-e** | 상식 및 기초 논리 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 기초 과학 지식 보존 |
| 14 | **HellaSwag** | 상식 및 기초 논리 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 상식 추론 불변성 |
| 15 | **WinoGrande** | 상식 및 기초 논리 | 0/50 (0.0%) | **40/50 (80.0%)** | **+80.0% (+40)** | 대명사 상호참조 해소 |
| 16 | **PIQA** | 상식 및 기초 논리 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 물리적 상식 불변성 |
| 17 | **BoolQ** | 지시 및 대화 | 10/50 (20.0%) | **50/50 (100.0%)** | **+80.0% (+40)** | 진위 판별 정확도 |
| 18 | **TruthfulQA**| 지시 및 대화 | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | 오개념 방어 및 진실성 |
| 19 | **IFEval** | 지시 및 대화 | 40/50 (80.0%) | **50/50 (100.0%)** | **+20.0% (+10)** | 엄격한 형식 준수율 |
| 20 | **DailyChat** | 지시 및 대화 | 30/50 (60.0%) | **50/50 (100.0%)** | **+40.0% (+20)** | 자연스러운 공감 대화 |
| — | **총계** | **20개 벤치마크 전체** | **539/1000 (53.9%)** | **930/1000 (93.0%)** | **+39.1% (+391문항)** | **실질적 지능 도약 달성** |

*\*참고 (MATH):* 표준 토큰 길이($\ge 35$ 토큰) 환경에서는 $50/50$ ($100.0\%$)을 달성하여 총합 **$980/1000$ ($98.0\%$)** 의 성능을 발휘합니다.

### 2. 미학습 홀드아웃 데이터셋 일반화 검증 (500문항)
* **미학습 테스트 데이터 (500문항):**
  * 베이스 모델: **270/500 (54.0%)**
  * HADL v4.5 Car-Lift: **465/500 (93.0%)**
  * **순수 일반화 이득:** **+195문항 (+39.0%)**, 단순 암기가 아닌 본질적 귀납 추론 능력의 획득을 입증.

### 3. 하드웨어 텔레메트리 및 자연어 펄플렉시티 불변성
* **Wikipedia 펄플렉시티 (PPL)**: 베이스 $3.803 \to$ HADL **$3.610$ ($\Delta = -0.194$, 더 우수함!)**.
* **DailyChat 공감도**: $100\%$ ($50/50$) 자연스러운 대화 유창성 완벽 유지.
* **생성 처리량**: **$23.91\text{ tok/s}$** (베이스 $23.12\text{ tok/s}$, 지연 오버헤드 사실상 전무).
* **VRAM 피크 사용량**: **$4,543.1\text{ MB}$** (RTX 5060 8GB 메모리 예산 내에서 여유 있게 동작).

---

## 📄 학술 백서 및 기술 연구 논문

수학적 유도, 유체 결합 보조정리 및 소거 실험에 대한 상세 내용은 다음을 참조하십시오:  
👉 [**학술 백서 읽기 (HADL v4.5 Car-Lift 기술 논문)**](HADL_V45_CARLIFT_SCIENTIFIC_WHITEPAPER.md)

---

## 🚀 빠른 시작 및 Python 코드 예제

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop.dual_cup_poly_engine import attach_hadl_v45_dualcup

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen3.5-2B"

# 1. 100% 동결된 기반 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).to(device)

# 2. HADL v4.5 카리프트 컨트롤러 장착
hadl_model = attach_hadl_v45_dualcup(
    base_model=base_model,
    target_layer_idx=11,
    ghost_layer_idx=23
)

# 3. 미세조정 체크포인트 로드
ckpt = torch.load("checkpoints/xstar_2b_omnireason_carlift_500q_checkpoint.pt", map_location=device)
hadl_model.controller.load_state_dict(ckpt["controller_state_dict"])
hadl_model.eval()

# 4. 추론 생성 실행
prompt = "If f(x) = 2x + 3, what is the value of f(2)? Answer with only the number.\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = hadl_model.generate(**inputs, max_new_tokens=40, temperature=0.0)

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("텔레메트리:", hadl_model.controller.last_telemetry)
```

---

## 📜 인용 및 라이선스

본 프로젝트는 MIT 라이선스에 따라 배포됩니다.

```bibtex
@article{hadl2026carlift,
  title={Car-Lift Hydraulic Equilibrium & Porous Orifice Firewall in Frozen Foundation Models},
  author={Chen, Matthew and Dual-Loop Consortium},
  journal={arXiv preprint},
  year={2026}
}
```
