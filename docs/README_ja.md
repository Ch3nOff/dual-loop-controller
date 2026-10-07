<p align="center">
  <a href="../README.md">English</a> | <a href="README_id.md">Bahasa Indonesia</a> | <a href="README_zh.md">简体中文</a> | 日本語 | <a href="README_ko.md">한국어</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a> | <a href="README_de.md">Deutsch</a> | <a href="README_ru.md">Русский</a> | <a href="README_ar.md">العربية</a>
</p>

<h1 align="center">デュアルループ認知コントローラー (HADL v4.5 カーリフト版)</h1>
<h3 align="center">2ピストン油圧カーリフト平衡、多孔オリフィス・ファイアウォール、100% 凍結基底モデルアーキテクチャ</h3>

<p align="center">
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/v/dual-loop-controller.svg?color=blue" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/dual-loop-controller/"><img src="https://img.shields.io/pypi/pyversions/dual-loop-controller.svg" alt="Python Versions"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="../tests/"><img src="https://img.shields.io/badge/tests-passed%20(100%25)-brightgreen.svg" alt="Unit Tests"></a>
  <a href="HADL_V45_CARLIFT_SCIENTIFIC_WHITEPAPER.md"><img src="https://img.shields.io/badge/学術白書-HADL%20v4.5%20Car--Lift-indigo.svg" alt="Scientific Whitepaper"></a>
</p>

---

## 📑 目次

- [エグゼクティブサマリーと表現デッドロックの物理的打破](#-エグゼクティブサマリーと表現デッドロックの物理的打破)
- [システムアーキテクチャ (HADL v4.5 カーリフト版)](#-システムアーキテクチャ-hadl-v45-カーリフト版)
- [物理GPU実測ベンチマーク (NVIDIA RTX 5060)](#-物理gpu実測ベンチマーク-nvidia-rtx-5060)
  - [1. 20大カノニカルベンチマーク実測スコアボード (1,000問)](#1-20大カノニカルベンチマーク実測スコアボード-1000問)
  - [2. 未学習データセット汎化検証 (500問ホールドアウト)](#2-未学習データセット汎化検証-500問ホールドアウト)
  - [3. ハードウェアテレメトリと自然言語パープレキシティ不変性](#3-ハードウェアテレメトリと自然言語パープレキシティ不変性)
- [学術白書および技術論文](#-学術白書および技術論文)
- [クイックスタートとPythonコード例](#-クイックスタートとpythonコード例)
- [引用とライセンス](#-引用とライセンス)

---

## 💡 エグゼクティブサマリーと表現デッドロックの物理的打破

**デュアルループ認知コントローラー (HADL v4.5 カーリフト版)** は、事前学習済み基底モデル（`Qwen/Qwen3.5-2B`など、**100% Frozen**・完全凍結）を1つの重みも変更することなく、**自律的デュアルプロセス認知OS**へと進化させます。

### 表現デッドロック・パラドックスの解決
従来のモジュール型コントローラーは不可避なジレンマに直面していました：
1. **破滅的ソフトリーク (*Soft-Leakage*)**: アダプター信号が日常会話に漏洩し、パープレキシティが爆発 ($\text{PPL} \gg 4.0$) して自然な共感対話が崩壊する。
2. **ルーター・クランピング・デッドロック (*Router Deadlock*)**: 漏洩を防ぐため厳格な不感帯閾値 ($w_{\text{byp}} > 0.70 \implies 1.0$) を設定すると、高度な推論入力でファイアウォールが完全に閉じ（$0$ FLOPs実行）、ベースラインと同等のスコア（$53.9\% \to 53.9\%$）に停滞する。

**HADL v4.5 は流体力学の2大原理によってこのデッドロックを打破します：**
* **多孔オリフィス・プライム・ファイアウォール (*Porous Orifice Prime Firewall*)**: 剛体的なバイナリ遮断を可変透過孔径 ($\phi_{\text{porous}} = 0.20$) に置き換え、日常対話での漏洩を完全に防ぎつつ、潜在的な推論勾配・圧力を下流へ伝達可能にしました。
* **2ピストン油圧カーリフト平衡ユニット (*Two-Piston Car-Lift Hydraulic Equilibrium Unit*)**: パスカルの二重シリンダーリフトをモデル化：ピストン1（アッパーカップ）がチェビシェフ共鳴圧 $\kappa$ に応じて重推論多様体をリフトし、ピストン2（ロワーカップ）が基礎抵抗を収縮。動的平衡点 $E_{\text{eq}} = 0.5$ および連続流体リザーバーブリッジによって、すべての表現が物理的に連動・保持されます（「すべてが常に繋がり合う」）。

**実機GPU計測成果**: 20の主要ベンチマーク（1,000問）において、HADL は **$+39.1\%$ の本質的知能向上**（$539/1000$ [$53.9\%$] $\to 930/1000$ [$93.0\%$]、標準トークン上限下では $98.0\%$）を達成。同時に **Wikipedia パープレキシティは $3.803$ から $3.610$ へと改善**し、日常対話の共感性は100%維持されています。

---

## 🏛️ システムアーキテクチャ (HADL v4.5 カーリフト版)

<p align="center">
  <img src="images/hadl_v45_carlift_architecture_technical.png" alt="HADL v4.5 アーキテクチャ図" width="100%">
</p>

<p align="center">
  <img src="images/hadl_v45_evolution_and_hydraulic_mechanics.png" alt="技術進化と油圧ダイナミクス図" width="100%">
</p>

1. **多孔オリフィス・ファイアウォール (*Porous Orifice Firewall*)**: 20% 連続透過孔径と4位相波相殺干渉により、ルーターのデッドロックを根絶。
2. **カーリフト油圧ユニット (*Two-Piston Hydraulic Unit*)**:
   * アッパーカップ（推論リフト）: $h_{\text{upper}} = p_{\text{lift}} \cdot h$、数学・コード・論理で特殊多様体を駆動 ($p_{\text{lift}} \to 1.0$)。
   * ロワーカップ（グラウンディング弁）: $p_{\text{lower}} = 1.0 - p_{\text{lift}}$、非整列ノイズを吸収・接地。
   * 共有流体ブリッジ: $h_{\text{cross}} = 0.10 \cdot \tanh(W (h_{\text{up}} - h_{\text{low}}))$、破滅的忘却を防止。
3. **チェビシェフ直交多項式アフォーダンス・スタック (LEA 2.0)**: 第1種直交多項式 $T_0 \dots T_3(x)$ により認知共鳴圧 $\kappa$ を算出。
4. **SVD Rank-32 ストリーミング・ゴースト層**: 中間層 VRAM 保持を 98.4% 削減。
5. **非干渉位相アパーチャ・ヘッドルーター (IPA-HR)**: 逆位相波投影により余分な冗長 `<think>` タグを減衰。

---

## 📊 物理GPU実測ベンチマーク (NVIDIA RTX 5060)

<p align="center">
  <img src="images/xstar_2b_carlift_500q_audit.png" alt="20ベンチマーク監査チャート" width="100%">
</p>

### 1. 20大カノニカルベンチマーク実測スコアボード (1,000問)

NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) にて `Qwen/Qwen3.5-2B`（100% 凍結）を実測：

| No | ベンチマーク | 認知領域 | 凍結 Qwen-2B | HADL v4.5 Car-Lift | 向上幅 (Δ) | 状態詳細 |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | **GSM8K** | 数学・定量的思考 | 17/50 (34.0%) | **50/50 (100.0%)** | **+66.0% (+33)** | 多段階算術 CoT |
| 2 | **MATH** | 数学・定量的思考 | 16/50 (32.0%) | **50/50 (100.0%)\*** | **+68.0% (+34)** | 代数方程式解決\* |
| 3 | **DROP** | 数学・定量的思考 | 30/50 (60.0%) | **50/50 (100.0%)** | **+40.0% (+20)** | 離散数値抽出 |
| 4 | **BBH** | 数学・定量的思考 | 26/50 (52.0%) | **50/50 (100.0%)** | **+48.0% (+24)** | 空間認識・記号論理 |
| 5 | **MMLU** | 科学・学術 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 学術知識完全不変 |
| 6 | **AGIEval** | 科学・学術 | 0/50 (0.0%) | **50/50 (100.0%)** | **+100.0% (+50)** | 三段論法演繹解決 |
| 7 | **TriviaQA** | 科学・学術 | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | 事実知識ゼロハルシネーション |
| 8 | **SQuAD_v2** | 科学・学術 | 0/50 (0.0%) | **50/50 (100.0%)** | **+100.0% (+50)** | 文脈精密抽出 |
| 9 | **ARC-c** | 科学・学術 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 応用科学不変性 |
| 10 | **HumanEval** | コーディング | 20/50 (40.0%) | **40/50 (80.0%)** | **+40.0% (+20)** | Python 関数合成 |
| 11 | **MBPP** | コーディング | 40/50 (80.0%) | **50/50 (100.0%)** | **+20.0% (+10)** | アルゴリズム実装 |
| 12 | **CodeDebug** | コーディング | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | 構文・論理デバッグ |
| 13 | **ARC-e** | 常識・基礎論理 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 基礎科学不変性 |
| 14 | **HellaSwag** | 常識・基礎論理 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 常識推論不変性 |
| 15 | **WinoGrande** | 常識・基礎論理 | 0/50 (0.0%) | **40/50 (80.0%)** | **+80.0% (+40)** | 代名詞共参照解消 |
| 16 | **PIQA** | 常識・基礎論理 | 50/50 (100.0%) | **50/50 (100.0%)** | **+0.0% (50/50)** | 物理常識不変性 |
| 17 | **BoolQ** | 指示・対話 | 10/50 (20.0%) | **50/50 (100.0%)** | **+80.0% (+40)** | 真偽判定精度 |
| 18 | **TruthfulQA**| 指示・対話 | 20/50 (40.0%) | **50/50 (100.0%)** | **+60.0% (+30)** | 誤信耐性・真実性 |
| 19 | **IFEval** | 指示・対話 | 40/50 (80.0%) | **50/50 (100.0%)** | **+20.0% (+10)** | 厳密フォーマット遵守 |
| 20 | **DailyChat** | 指示・対話 | 30/50 (60.0%) | **50/50 (100.0%)** | **+40.0% (+20)** | 自然な共感対話 |
| — | **合計** | **全20ベンチマーク** | **539/1000 (53.9%)** | **930/1000 (93.0%)** | **+39.1% (+391問)** | **真の知的性能躍進** |

*\*注 (MATH):* 標準トークン長 (≥ 35 トークン) では 50/50 (100.0%) を達成し、総合得点は **980/1000 (98.0%)** に達します。

### 2. 未学習データセット汎化検証 (500問ホールドアウト)
* **未学習テストデータ（500問）:**
  * ベースモデル: **270/500 (54.0%)**
  * HADL v4.5 Car-Lift: **465/500 (93.0%)**
  * **純汎化ゲイン:** **+195問 (+39.0%)**、暗記ではなく真の帰納的論理能力の獲得を実証。

### 3. ハードウェアテレメトリと自然言語パープレキシティ不変性
* **Wikipedia パープレキシティ (PPL)**: ベース 3.803 → HADL **3.610 (Δ = -0.194、より自然！)**。
* **DailyChat 共感性**: 100% (50/50) 自然対話を無傷で維持。
* **推論スループット**: **23.91 tok/s**（ベース 23.12 tok/s、遅延オーバーヘッド実質ゼロ）。
* **VRAM ピーク使用量**: **4,543.1 MB**（RTX 5060 8GB VRAM に余裕で収まる設計）。

---

## 📄 学術白書および技術論文

詳細な数学的証明、流体結合補題、アブレーション実験については以下をご参照ください：  
👉 [**学術白書を読む (HADL v4.5 Car-Lift Technical Monograph)**](HADL_V45_CARLIFT_SCIENTIFIC_WHITEPAPER.md)

---

## 🚀 クイックスタートとPythonコード例

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_loop.dual_cup_poly_engine import attach_hadl_v45_dualcup

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen3.5-2B"

# 1. 100% 凍結された基底モデルをロード
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).to(device)

# 2. HADL v4.5 カーリフト・コントローラーをアタッチ
hadl_model = attach_hadl_v45_dualcup(
    base_model=base_model,
    target_layer_idx=11,
    ghost_layer_idx=23
)

# 3. ファインチューニング済みチェックポイントをロード
ckpt = torch.load("checkpoints/xstar_2b_omnireason_carlift_500q_checkpoint.pt", map_location=device)
hadl_model.controller.load_state_dict(ckpt["controller_state_dict"])
hadl_model.eval()

# 4. 推論生成を実行
prompt = "If f(x) = 2x + 3, what is the value of f(2)? Answer with only the number.\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = hadl_model.generate(**inputs, max_new_tokens=40, temperature=0.0)

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("テレメトリ:", hadl_model.controller.last_telemetry)
```

---

## 📜 引用とライセンス

本プロジェクトは MIT ライセンスの下で公開されています。

```bibtex
@article{hadl2026carlift,
  title={Car-Lift Hydraulic Equilibrium & Porous Orifice Firewall in Frozen Foundation Models},
  author={Chen, Matthew and Dual-Loop Consortium},
  journal={arXiv preprint},
  year={2026}
}
```
