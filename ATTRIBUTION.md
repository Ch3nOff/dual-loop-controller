# Third-Party Attributions & Open-Source Licenses

This repository, **Dual-Loop Cognitive Controller**, is developed as an open-source project licensed under the [MIT License](LICENSE). In accordance with open-source licensing standards and intellectual property guidelines, this document provides complete, transparent attribution to all upstream frameworks, model architectures, benchmark datasets, and academic literature referenced or utilized.

---

## 1. Upstream Model Architectures & Frameworks

### A. Qwen Model Architecture (Alibaba Cloud / Qwen Team)
* **Components Interfaced**: `Qwen2ForCausalLM`, `Qwen3_5ForConditionalGeneration` architectural configurations and hook interfaces in `dual_loop/adapters/qwen_adapter.py`.
* **License**: [Apache 2.0 License](https://github.com/QwenLM/Qwen2.5/blob/main/LICENSE) / Tongyi Qianwen Community License.
* **Notice**: This repository does not distribute proprietary base model weights. Integration is achieved strictly via non-invasive PyTorch forward hooks using standard open-source APIs (`transformers`).
* **Source**: [https://github.com/QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5)
* **Citation**:
  ```bibtex
  @article{qwen25,
    title={Qwen2.5 Technical Report},
    author={Qwen Team},
    journal={arXiv preprint arXiv:2412.15115},
    year={2024}
  }
  ```

### B. Hugging Face Transformers & Accelerate
* **Components Used**: Model loading, configuration primitives, and tokenization utilities.
* **License**: [Apache 2.0 License](https://github.com/huggingface/transformers/blob/main/LICENSE)
* **Copyright**: (c) 2018-present Hugging Face, Inc.
* **Source**: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

### C. PyTorch
* **Components Used**: Core tensor computations, `torch.nn`, autograd, and CUDA execution.
* **License**: [Modified BSD License](https://github.com/pytorch/pytorch/blob/main/LICENSE)
* **Copyright**: (c) 2016-present Meta Platforms, Inc. / PyTorch Foundation.
* **Source**: [https://pytorch.org/](https://pytorch.org/)

---

## 2. Evaluation Harnesses & Benchmark Datasets

### A. EleutherAI LM Evaluation Harness (`lm-evaluation-harness`)
* **Components Used**: Standalone academic evaluation runner (`run_lm_eval.py`).
* **License**: [MIT License](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/LICENSE)
* **Copyright**: (c) 2020-present EleutherAI.
* **Source**: [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* **Citation**:
  ```bibtex
  @misc{eval-harness,
    author = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and others},
    title = {A framework for few-shot language model evaluation},
    year = {2023},
    publisher = {Zenodo}
  }
  ```

### B. Benchmark Datasets Referenced in Scoreboard Audits
All benchmarks reported in `eval_results/` and visualized in project scorecards are evaluated according to their respective open-source or academic research licenses:

1. **MMLU (Massive Multitask Language Understanding)**
   * Authors: Hendrycks et al. (UC Berkeley, Columbia, UChicago)
   * License: MIT License / Creative Commons Attribution (CC-BY 4.0).
   * Paper: *Measuring Massive Multitask Language Understanding*, ICLR 2021.
2. **IFEval (Instruction Following Evaluation)**
   * Authors: Zhou et al. (Google Research)
   * License: Apache 2.0 License.
   * Paper: *Instruction-Following Evaluation for Large Language Models*, 2023.
3. **GPQA (Google-Proof Q&A)**
   * Authors: Rein et al. (NYU, Anthropic, Alignment Research Center)
   * License: Creative Commons Attribution 4.0 International (CC-BY 4.0).
   * Paper: *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*, 2023.
4. **C-Eval (Multi-Level Multi-Discipline Chinese Evaluation Suite)**
   * Authors: Huang et al. (Tsinghua University, SJTU)
   * License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).
   * Paper: *C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite*, NeurIPS 2023.
5. **BFCL (Berkeley Function Calling Leaderboard)**
   * Authors: Gorilla LLM Team (UC Berkeley)
   * License: Apache 2.0 License.
   * Source: [https://github.com/ShishirPatil/gorilla_eval](https://github.com/ShishirPatil/gorilla_eval)
6. **LongBench (Bilingual Multi-Task Long-Context Benchmark)**
   * Authors: Bai et al. (THUDM / Tsinghua University)
   * License: MIT License.
   * Source: [https://github.com/THUDM/LongBench](https://github.com/THUDM/LongBench)
7. **PIQA (Physical Interaction: Question Answering)**
   * Authors: Bisk et al. (University of Washington, Allen Institute for AI)
   * License: Academic Free License (AFL 3.0) / Apache 2.0.
8. **Public Scorecard Aggregator Reference**:
   * Data format and baseline comparisons align with the public scorecard methodology from [llm-stats.com](https://llm-stats.com/).

---

## 3. Academic Prior Art & Theoretical Foundations

The theoretical architecture of the Dual-Loop Cognitive Controller builds upon and cites foundational contributions in cognitive AI and neural recurrence:

* **Dual-Process Cognitive Theory**: Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
* **Recurrent Latent Deliberation (PonderNet)**: Banino et al. (2021). *PonderNet: Learning to Ponder*. DeepMind.
* **Mixture-of-Depths (Static Capacity Routing)**: Raposo et al. (2024). *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*. Google DeepMind.
* **Continuous Chain-of-Thought (Coconut)**: Hao et al. (2024). *Training Large Language Models to Reason in a Continuous Latent Space*.
* **Adaptive Computation Time (ACT)**: Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural Networks*. arXiv:1603.08983.
* **Chain-of-Thought Prompting**: Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 2022.

---

## 4. Statement of Originality & Intellectual Property Guarantee

1. **Original Implementation**: All code within the `dual_loop/` directory—including `RecurrentLatentController`, `CognitiveWorkingMemory`, `TopKCapacityCrossAttention`, `EntropyHaltingUnit`, `DualLoopTransformer`, and `DualLoopQwenModel`—is an original implementation authored by Matthew Chen.
2. **Clean-Room Weight Training**: The bundled 225K parameter reference checkpoint (`checkpoint_trained_dualloop.pt`) was trained from scratch using synthetic relational graph generators authored within this repository, free of external private data or proprietary weights.
3. **No GPL Contamination**: No components under copyleft licenses (GPL/AGPL) have been incorporated into this codebase, ensuring full compatibility with the MIT License and enterprise deployment.
