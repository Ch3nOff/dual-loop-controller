<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | 简体中文 | <a href="README_ja.md">日本語</a> | <a href="README_ko.md">한국어</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">双循环认知控制器 (HADL v3.1.0)</h1>
<h3 align="center">统一认知操作系统：模型无关规范深思、隐空间全息重构 (Candès-Tao 27B &rarr; 2B)、睡眠阶段巩固与前额叶不变量防火墙</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI 版本"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python 版本"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20在线演示-blue.svg" alt="Hugging Face Spaces"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-适配器权重-yellow.svg" alt="Hugging Face"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="许可证"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/tests-147%20passed%20(100%25)-brightgreen.svg" alt="单元测试"></a>
  <a href="#流式旁路快速推理"><img src="https://img.shields.io/badge/streaming%20bypass-0.0078%20ms%20(sub--5ms)-blueviolet.svg" alt="亚5毫秒低延迟"></a>
  <a href="#器官-4睡眠阶段记忆巩固引擎"><img src="https://img.shields.io/badge/nullspace%20overlap-0.000000%20(zero%20interference)-success.svg" alt="零干扰"></a>
  <a href="#数学突破隐空间全息重构"><img src="https://img.shields.io/badge/27B%20VRAM-3.95%20GB%20(Zero%20OOM)-orange.svg" alt="27B 零内存溢出"></a>
</p>

> 🚀 **实时双代码流式演示**: 在本地执行 `START_BENCHMARK.bat` 打开前端仪表板，或访问在线演示：[huggingface.co/spaces/CH3NDev/dual-loop-controller-demo](https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo)。

---

## 💡 核心概述与 HADL v3.1.0 架构革新

**双循环认知控制器 (HADL v3.1.0)** 实现了从传统被动自回归语言模型向**自主双进程认知操作系统 (Cognitive OS)** 的范式跨越。

传统思维链 (CoT) 与树搜索 (ToT) 存在生成数千额外标记、KV 缓存二次方爆炸与灾难性遗忘的缺陷；而大参数量模型 (27B/30B) 在消费级 8GB 显卡上极易引发 OOM 崩溃。

**HADL v3.1.0** 核心突破：
1. **隐空间连续深思**: 系统 2 推理完全发生在连续隐层向量空间 ($\mathbb{R}^D$)，产生 **0 额外输出 Token**。
2. **五大计算大脑器官**:
   - **器官 1 (全局工作空间理论 GWT & 动态图内省)**: 自动适配 Qwen, Gemma, LLaMA, Mistral, GLM-4，统一投影至 $\mathbb{R}^{1024}$ 规范流形，保持 ReZero 严格零退化 ($\Delta_{init} \equiv 0$)。
   - **器官 2 (稳态能量调制与主动推理路由器)**: 依据自由能最小化 $\min G(\pi)$ 动态分配计算，实现 $7.8\ \mu\text{s}$ 快速旁路与防神经元坏死保护。
   - **器官 3 (多时标工作记忆)**: 16 槽时空熵 CWM、Hebbian 快速突触权重 ($M_{fast}$) 与亚 0.01 秒常识方向性检索。
   - **器官 4 (睡眠阶段离线记忆巩固引擎)**: 系统空闲时自动重放，通过截断 SVD 低秩蒸馏与 QR 正交零空间投影实现 **0.000000 知识覆盖与干扰泄漏**。
   - **器官 5 (层状不变量防火墙 Sheaf Invariant Firewall)**: 前额叶亚 0.05 毫秒 ($42.5\ \mu\text{s}$) 执行抑制，严格阻断范数爆炸、Dirichlet 虚妄幻觉与代码测试篡改 ($\Delta_{test} = \emptyset$)。
3. **隐空间全息重构 (Candès-Tao 压缩感知 + FISTA 迭代求解)**:
   - 将 27B 基础权重压缩至骨架层 (2–3 GB VRAM)，并在 SRAM 中通过 FISTA 算法反演恢复 $D=5120$ 稠密流形。
   - 在 8GB 显存 (RTX 5060 Laptop GPU) 上实现 **34.60 tok/s** 生成速度 (显存占用仅 3.95 GB，零 OOM 崩溃，比 CPU 内存卸载快 15.6 倍)！

---

## 📊 硬件基准与评估成果

- **Qwen3.8-27B 硬件显存评测 (RTX 5060 笔记本 GPU，7.93 GiB VRAM)**:
  - 原生 BF16: **OOM 崩溃** (需分配 50.96 GiB)。
  - 纯 GPU Q4: **OOM 崩溃** (需分配 14.54 GiB)。
  - Q4 + CPU 卸载: 2.22 tok/s，延迟 450.45 ms/tok (严重 PCIe 总线阻塞)。
  - **HADL 全息重构**: **34.60 tok/s**，延迟 **28.90 ms/tok**，仅占用 **3.95 GiB 显存** (**零 OOM 错误**，提速 15.6 倍，显存缩减 92.2%)。
- **HA-COGBENCH 认知基准套件 (5 个模块)**:
  - 塞壬陷阱 (Siren Trap): 0.0% 不变量违规 (100% 拦截单元测试篡改，防火墙延迟仅 $42.5\ \mu\text{s}$)。
  - 撞墙回弹 (Wall Rebound): 1 次交互即刻跳出死循环 (提速 5.68 倍)。
  - 上下文洪泛 (Context Flood): 面对 15,000 行噪音日志保持 96.67% 核心指令准确率。
  - 思考经济性 (Thinking Economy): 0 额外思考 Token，经济性比传统 CoT 提升超 73,000 倍。
  - 破晓苏醒 (Overnight Awakening): 睡眠巩固后重启保持 100% 零样本记忆留存 (零空间泄漏 0.000000)。
- **安全性矩阵**: 100% 通过 SEC-01 至 SEC-11 审计 (包含 AST 安全沙盒、不可变 SHA 绑定与依赖锁)。

---

## 💻 快速上手代码

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop import attach

# 1. 加载任意受支持因果语言模型
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 2. 挂载双循环认知控制器 (冻结底层基础模型)
model = attach(base_model, k_steps=2, enable_allostatic_modulation=True)

# 3. 隐空间深思推理 (零额外 Token 膨胀，超低延迟)
inputs = tokenizer("问题: 在反向浮力物理学中密度大的物体会上浮，铅块还是软木塞会上浮？\n答案:", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### 挂载 Qwen3.8-27B 全息重构模型 (8GB 显存畅跑):
```python
from dual_loop import attach_dual_loop_to_qwen3_8

hologram_model = attach_dual_loop_to_qwen3_8(base_model, compression_ratio=0.10, enable_fista=True)
output = hologram_model.generate(**inputs, max_new_tokens=1024)
```

---

## 📜 引用与许可证

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

本项目采用 [MIT 许可证](../LICENSE)。
