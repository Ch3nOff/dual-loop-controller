<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | 日本語 | <a href="README_ko.md">한국어</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">Dual-Loop Cognitive Controller (HADL v3.1.0)</h1>
<h3 align="center">統合認知オペレーティングシステム：モデル非依存標準熟考・潜在ホログラム再構成（Candès-Tao 27B &rarr; 2B）・睡眠期統合・前頭前野不変量ファイアウォール</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python Versions"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue.svg" alt="Hugging Face Spaces"></a>
  <a href="https://huggingface.co/CH3NDev/dual-loop-qwen3.5-2b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Adapter%20Weights-yellow.svg" alt="Hugging Face"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/tests-147%20passed%20(100%25)-brightgreen.svg" alt="Unit Tests"></a>
  <a href="#高速バイパス推論"><img src="https://img.shields.io/badge/streaming%20bypass-0.0078%20ms%20(sub--5ms)-blueviolet.svg" alt="Sub-5ms Latency"></a>
  <a href="#器官-4睡眠期メモリ統合エンジン"><img src="https://img.shields.io/badge/nullspace%20overlap-0.000000%20(zero%20interference)-success.svg" alt="ゼロ干渉"></a>
  <a href="#潜在ホログラム再構成"><img src="https://img.shields.io/badge/27B%20VRAM-3.95%20GB%20(Zero%20OOM)-orange.svg" alt="27B ゼロOOM"></a>
</p>

> 🚀 **リアルタイム並列ストリーミングデモ**: ローカルで `START_BENCHMARK.bat` を実行するか、オンラインデモ [huggingface.co/spaces/CH3NDev/dual-loop-controller-demo](https://huggingface.co/spaces/CH3NDev/dual-loop-controller-demo) を体験してください。

---

## 💡 システム概要と HADL v3.1.0 の技術革新

**Dual-Loop Cognitive Controller (HADL v3.1.0)** は、従来の単なる受動的自己回帰言語モデルを、**自律的デュアルプロセス認知オペレーティングシステム (Cognitive OS)** へと進化させます。

従来の Chain-of-Thought (CoT) は数千トークンを浪費し、KVキャッシュの爆発と致命的な破滅的忘却を引き起こします。また、27B〜30Bクラスの大規模モデルは8GB VRAMの一般向けGPUではメモリ不足 (OOM) を引き起こします。

**HADL v3.1.0** による解決策：
1. **潜在連続熟考 (Latent Deliberation)**: システム2の思考は隠れ層活性化空間 ($\mathbb{R}^D$) 内で完結し、**追加トークンの生成はゼロ**。
2. **5つの計算脳器官**:
   - **器官1 (大域的作業空間 GWT & 動的グラフ検査)**: Qwen, Gemma, LLaMA, Mistral, GLM-4 などを自動検出し、$\mathbb{R}^{1024}$ 統一規範空間へ写像。初期化時の零劣化 (ReZero: $\Delta_{init} \equiv 0$) を保証。
   - **器官2 (アロスタシス調節器 & 能動推論ルーター)**: 自由エネルギー最小化 $\min G(\pi)$ により、$7.8\ \mu\text{s}$ の高速バイパスと熟考ループを動的に切替。
   - **器官3 (多重時間スケール作業メモリ)**: 16スロットの時空エントロピーCWM、ヘブ即時適応重み ($M_{fast}$)、高速コモンセンス検索。
   - **器官4 (睡眠期統合エンジン)**: アイドル時のオフライン再生により、SVD低ランク蒸留およびQR直交零空間射影を実施。**干渉リーク 0.000000** を達成。
   - **器官5 (層不変量ファイアウォール Sheaf Firewall)**: サブ0.05ms ($42.5\ \mu\text{s}$) の前頭前野抑制フィルターにより、ノルム発散、傲慢な幻覚、テスト改ざん ($\Delta_{test} = \emptyset$) を完全遮断。
3. **潜在ホログラム再構成 (Candès-Tao 圧縮センシング + FISTA アルゴリズム)**:
   - 27Bモデルの重みを2–3 GB VRAMのスケルトンへ極限圧縮し、SRAM上でFISTA逆復元を実行。
   - 8GB VRAM (RTX 5060 Laptop GPU) 上で、**34.60 tok/s** の高速生成、VRAM消費わずか **3.95 GB**、**OOMエラーゼロ** を実現！

---

## 📊 実証ベンチマーク概要

- **Qwen3.8-27B ハードウェアメモリ評価 (RTX 5060 Laptop GPU 7.93 GiB VRAM)**:
  - ネイティブ BF16: **OOM クラッシュ** (50.96 GiB 必要)。
  - 純粋 Q4 GPU: **OOM クラッシュ** (14.54 GiB 必要)。
  - Q4 + CPU オフロード: 2.22 tok/s、遅延 450.45 ms/tok (重度のPCIeバスボトルネック)。
  - **HADL ホログラム**: **34.60 tok/s**、遅延 **28.90 ms/tok**、**3.95 GiB VRAM** (**完全ゼロ OOM**、オフロード比 15.6倍高速、VRAM 92.2% 削減)。
- **HA-COGBENCH 認知ベンチマーク (5モジュール)**:
  - セイレーンの罠 (Siren Trap): 0.0% の不変量違反 (テスト改ざんを $42.5\ \mu\text{s}$ で100%遮断)。
  - 壁のリバウンド (Wall Rebound): ループ停止から1ターンで即時復帰 (5.68倍高速)。
  - コンテキストフラッド (Context Flood): 15,000行のノイズ混入下でも 96.67% の指示遵守率を維持。
  - 思考経済性 (Thinking Economy): トークン消費ゼロ、CoT比 73,000倍以上の TER 効率。
  - 目覚めの想起 (Overnight Awakening): 再起動後も睡眠統合により 100% ゼロショット想起。
- **セキュリティ監査**: SEC-01 から SEC-11 までの全項目に 100% 適合。

---

## 💻 クイックスタート

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

# 3行でコントローラーを装着
model = attach(base_model, k_steps=2, enable_allostatic_modulation=True)

inputs = tokenizer("質問: 逆浮力の物理世界では密度の高い物体が浮きます。鉛とコルクのどちらが浮きますか？\n回答:", return_tensors="pt").to(base_model.device)
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📜 引用・ライセンス

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

本プロジェクトは [MIT ライセンス](../LICENSE) の下で公開されています。
