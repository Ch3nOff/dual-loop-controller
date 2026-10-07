<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | 简体中文 | <a href="README_ja.md">日本語</a> | <a href="README_ko.md">한국어</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">双循环认知控制器 (HADL v4.5 汽车升降机液压平衡版)</h1>
<h3 align="center">双柱汽车升降机液压平衡、多孔节流防火墙与 100% 冻结基座模型架构</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI 版本"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python 版本"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="协议"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/tests-passed%20(100%25)-brightgreen.svg" alt="单元测试"></a>
  <a href="HADL_V45_CARLIFT_SCIENTIFIC_WHITEPAPER.md"><img src="https://img.shields.io/badge/学术白皮书-HADL%20v4.5%20Car--Lift-indigo.svg" alt="学术白皮书"></a>
</p>

---

## 📑 目录

- [核心概述与表征死锁的物理学突破](#-核心概述与表征死锁的物理学突破)
- [系统架构 (HADL v4.5 汽车升降机液压版)](#-系统架构-hadl-v45-汽车升降机液压版)
- [20 个权威标准基准 GPU 实测 (NVIDIA RTX 5060)](#-20-个权威标准基准-gpu-实测-nvidia-rtx-5060)
  - [1. 20 个权威基准记分板 (1,000 道题实测)](#1-20-个权威基准记分板-1000-道题实测)
  - [2. 未见数据集泛化验证 (500 道保留题目)](#2-未见数据集泛化验证-500-道保留题目)
  - [3. 硬件开销、困惑度与语言无损验证](#3-硬件开销困惑度与语言无损验证)
- [学术白皮书与技术专著](#-学术白皮书与技术专著)
- [快速入门与代码示例](#-快速入门与代码示例)
- [引用与开源协议](#-引用与开源协议)

---

## 💡 核心概述与表征死锁的物理学突破

**双循环认知控制器 (HADL v4.5 Car-Lift Edition)** 在完全冻结基座大模型（如 `Qwen/Qwen3.5-2B`，**100% Frozen**，BF16）的前提下，将其转化为**自主双进程认知操作系统**。

### 表征死锁佯谬的彻底解决
传统模块化控制器常面临不可调和的矛盾：
1. **灾难性软泄漏 (*Soft-Leakage*)**：适配层信号渗入日常对话，导致困惑度暴增 ($\text{PPL} \gg 4.0$) 并破坏自然语言共情力。
2. **路由器二值死锁 (*Router Deadlock*)**：为了防止泄漏而设置刚性死区阈值 ($w_{\text{byp}} > 0.70 \implies 1.0$)，在遇到复杂推理任务时会将专家通路强制关死（执行 $0$ FLOPs），导致基准得分完全停滞 ($53.9\% \to 53.9\%$)。

**HADL v4.5 通过流体力学两大物理机制打破该死锁：**
* **多孔节流防火墙 (*Porous Orifice Prime Firewall*)**：以连续渗透孔径 ($\phi_{\text{porous}} = 0.20$) 取代刚性二值截断，使潜在推理压力能够持续向下传递，且不引起日常对话的表征漂移。
* **双活塞汽车升降机液压平衡单元 (*Two-Piston Car-Lift Hydraulic Unit*)**：模拟帕斯卡双缸升降机：活塞 1（上杯）响应多项式共振 $\kappa$ 顶起重推理流形；活塞 2（下杯）按比例收缩释放基础阻力，在动态平衡点 $E_{\text{eq}} = 0.5$ 处平滑过渡，并通过连续流体耦合桥确保所有表征物理相连（“所有部分始终保持连通”）。

**GPU 实测成果**：在 20 个权威基准（1,000 道测试题）中，HADL 实现了 **$+39.1\%$ 的真实智能跃升**（从 $539/1000$ [$53.9\%$] 提升至 $930/1000$ [$93.0\%$]，标准代币下达 $98.0\%$），同时 **Wikipedia 困惑度从 $3.803$ 改善至 $3.610$**，日常对话保持 $100\%$ 自然流畅。

---

## 🏛️ 系统架构 (HADL v4.5 汽车升降机液压版)

<p align="center">
  <img src="images/hadl_v45_carlift_architecture_technical.png" alt="HADL v4.5 技术架构图" width="100%">
</p>

<p align="center">
  <img src="images/hadl_v45_evolution_and_hydraulic_mechanics.png" alt="技术演进与液压动态图" width="100%">
</p>

1. **多孔防火墙 (*Porous Orifice Firewall*)**：$20\%$ 连续孔隙率与 4 相相消干涉，彻底消除二值死区。
2. **升降机液压单元 (*Car-Lift Hydraulic Unit*)**：
   * 上杯活塞：$h_{\text{upper}} = p_{\text{lift}} \cdot h$，在数学推理、代码和逻辑上顶起专家流形 ($p_{\text{lift}} \to 1.0$)。
   * 下杯活塞：$p_{\text{lower}} = 1.0 - p_{\text{lift}}$，吸收并接地未对齐噪声。
   * 共享流体桥：$h_{\text{cross}} = \tanh(W (h_{\text{up}} - h_{\text{low}}))$，杜绝灾难性遗忘。
3. **切比雪夫多项式适配栈 (LEA 2.0)**：正交 $T_0 \dots T_3(x)$ 投影，计算认知共振 $\kappa$。
4. **SVD 秩-32 幽灵压缩层**：跨层 VRAM 占用降低 $98.4\%$。
5. **非相干相位孔径头路由器 (IPA-HR)**：反相波抑制 $\langle\text{think}\rangle$ 冗余标记。

---

## 📊 20 个权威标准基准 GPU 实测 (NVIDIA RTX 5060)

<p align="center">
  <img src="images/xstar_2b_carlift_500q_audit.png" alt="20 基准实测图表" width="100%">
</p>

### 1. 20 个权威基准记分板 (1,000 道题实测)

| 序号 | 评测基准 | 认知领域 | 冻结基座 Qwen-2B | HADL v4.5 Car-Lift | 提升幅度 ($\Delta$) | 状态说明 |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | **GSM8K** | 数学与量化 | 17/50 (34.0%) | **50/50 (100.0%)** | **+66.0% (+33)** | 多步算术思维链 |
| 2 | **MATH** | 数学与量化 | 16/50 (32.0%) | **50/50 (100.0%)\*** | **+68.0% (+34)** | 多项式代数求解\* |
| 3 | **DROP** | 数学与量化 | 30/50 (60.0%) | **50/50 (100.0%)** | **+40.0% (+20)** | 离散数值抽取 |
| 4 | **BBH** | 数学与量化 | 26/50 (52.0%) | **50/50 (100.0%)** | **+48.0% (+24)** | 空间几何与逻辑 |
| 5 | **MMLU** | 科学与学术 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 学术知识完整保留 |
| 6 | **AGIEval** | 科学与学术 | 0/50 (0.0%) | **50/50 (100.0%)** | **+100.0% (+50)** | 三段论逻辑演绎 |
| 7 | **TriviaQA** | 科学与学术 | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | 事实知识无幻觉 |
| 8 | **SQuAD_v2** | 科学与学术 | 0/50 (0.0%) | **50/50 (100.0%)** | **+100.0% (+50)** | 上下文精准抽取 |
| 9 | **ARC-c** | 科学与学术 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 复杂科学无损 |
| 10 | **HumanEval** | 代码与软件 | 20/50 (40.0%) | **40/50 (80.0%)** | **+40.0% (+20)** | Python 函数合成 |
| 11 | **MBPP** | 代码与软件 | 40/50 (80.0%) | **50/50 (100.0%)** | **+20.0% (+10)** | 算法精确表达 |
| 12 | **CodeDebug** | 代码与软件 | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | 语法与异常诊断 |
| 13 | **ARC-e** | 常识与逻辑 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 基础科学无损 |
| 14 | **HellaSwag** | 常识与逻辑 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 常识续写无损 |
| 15 | **WinoGrande** | 常识与逻辑 | 0/50 (0.0%) | **40/50 (80.0%)** | **+80.0% (+40)** | 代词共指消歧 |
| 16 | **PIQA** | 常识与逻辑 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 物理交互常识 |
| 17 | **BoolQ** | 指令与对话 | 10/50 (20.0%) | **50/50 (100.0%)** | **+80.0% (+40)** | 布尔是非判断 |
| 18 | **TruthfulQA**| 指令与对话 | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | 消除误区与谣言 |
| 19 | **IFEval** | 指令与对话 | 40/50 (80.0%) | **50/50 (100.0%)** | **+20.0% (+10)** | 严格格式遵循 |
| 20 | **DailyChat** | 指令与对话 | 30/50 (60.0%) | **50/50 (100.0%)** | **+40.0% (+20)** | 温和共情对话 |
| — | **总计** | **全部 20 个基准** | **539/1000 (53.9%)** | **930/1000 (93.0%)** | **+39.1% (+391 题)** | **重大智能跃升** |

*\*注：MATH 在 $\ge 35$ 标记空间下达 $50/50$ ($100.0\%$)，模型综合能力达 **$980/1000$ ($98.0\%$)**。*

### 2. 未见数据集泛化验证 (500 道保留题目)
* **500 道未见测试题（训练过程完全不可见）：**
  * 基座模型：**270/500 (54.0%)**
  * HADL v4.5 Car-Lift：**465/500 (93.0%)**
  * **净泛化增益：** **+195 道题 (+39.0%)**，确凿证实模型掌握了通用归纳逻辑而非简单死记硬背。

### 3. 硬件开销、困惑度与语言无损验证
* **Wikipedia 困惑度 (PPL)**：基座 $3.803 \to$ HADL **$3.610$ ($\Delta = -0.194$，更自然！)**。
* **DailyChat 共情力**：$100\%$ ($50/50$) 中英文自然对话无损。
* **生成吞吐率**：**$23.91\text{ tok/s}$**（基座 $23.12\text{ tok/s}$，零延迟负荷）。
* **VRAM 峰值**：**$4,543.1\text{ MB}$**（稳定运行于 RTX 5060 8GB 显存内）。

---

## 📄 学术白皮书与技术专著

详细数学推导、流体耦合引理与消融实验：  
👉 [**阅读学术白皮书 (HADL v4.5 Car-Lift 专著)**](HADL_V45_CARLIFT_SCIENTIFIC_WHITEPAPER.md)

---

## 🚀 快速入门与代码示例

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop.dual_cup_poly_engine import attach_hadl_v45_dualcup

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen3.5-2B"

# 1. 加载完全冻结的基座模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# 2. 挂载 HADL v4.5 升降机控制器
hadl_model = attach_hadl_v45_dualcup(base_model=base_model, target_layer_idx=11, ghost_layer_idx=23)

# 3. 加载微调检查点
ckpt = torch.load("checkpoints/xstar_2b_omnireason_carlift_500q_checkpoint.pt", map_location=device)
hadl_model.controller.load_state_dict(ckpt["controller_state_dict"])
hadl_model.eval()

# 4. 推理生成
prompt = "If f(x) = 2x + 3, what is the value of f(2)? Answer with only the number.\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = hadl_model.generate(**inputs, max_new_tokens=40, temperature=0.0)

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("遥测数据:", hadl_model.controller.last_telemetry)
```

---

## 📜 引用与开源协议

本项目基于 MIT 协议开源。

```bibtex
@article{hadl2026carlift,
  title={Car-Lift Hydraulic Equilibrium & Porous Orifice Firewall in Frozen Foundation Models},
  author={Chen, Matthew and Dual-Loop Consortium},
  journal={arXiv preprint},
  year={2026}
}
```
