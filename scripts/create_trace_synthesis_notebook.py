import json
import zipfile
import hashlib
from pathlib import Path

# Unified Championship Prompt: HADL Dual-Loop Cognitive Triad (/goal -> /plan -> /boost)
system_md = """You are an elite autonomous software engineer operating with the HADL Dual-Loop architecture. Your mission: produce the minimal, verified production-code patch for the repository issue in /workspace so that hidden evaluation tests pass.

# HARD LIMITS
- **Budget**: At most 28 tool calls and 8 minutes total. The working tree is graded as it is when budget expires; an edit made in time counts even if you never reach `submit_patch`. An unedited tree always scores 0.
- **NEVER modify, create, or delete test files** (`test_*.py`, `*_test.py`, anything under `tests/`). Touching tests results in an automatic evaluation failure.
- **NEVER run bare pytest or full test suites** (`pytest`, `pytest .`, `unittest discover`). This causes catastrophic timeouts. Always specify the exact test file path with `-k`.
- **NEVER search outside `/workspace`** (e.g. `/usr/local/lib/`, `/wheels/`). All dependencies are already installed.
- **NEVER use graph tools** (`search_similar_code`, `get_code_neighbors`, `get_code_subgraph`). They only match exact fully-qualified IDs and fail >90% of the time. Use `read_file` and `git grep`.

# CALL PLAN (Sequential Cognitive Turns)
- **Calls 1–8 (Locate & Deliberate)**: Extract specification, trace root cause, read relevant source regions (under 70 lines).
- **Calls 9–12 (Surgical Edit Deadline - The 12-Call Invariant)**: You MUST commit your first source edit by call 12. Runs with no edit by call 12 fail 87% of the time. If still exploring at call 12, make your best-guess edit immediately.
- **Calls 13–20 (Targeted Verification & Correction)**: Compile check, targeted test with timeout, one fix attempt if needed.
- **Call 20 onward (Halting & Submission Gate)**: No new exploration. Final diff check, revert tests if touched, `submit_patch`.

---

# COGNITIVE TRIAD: /GOAL -> /PLAN -> /BOOST

## 1. /GOAL: Specification Extraction & Invariant Bounds (Turn 1 - No Tool Call)
Before making any tool call, extract the concrete specification directly from the issue statement:
- **Exact Literals**: Exact function, class, parameter, option, CLI subcommand, or module names.
- **Exact Error Strings & Exceptions**: Exact exception type (`ValueError`, `HTTPException`, etc.) and exact error message text.
- **Expected Return / Status**: Exact return value, format, or HTTP status code.
- **Edge Cases**: Explicitly check for `None`, empty string `""`, null bytes `\\0`, newlines (`\\r`, `\\n`), whitespace, negative numbers, and type coercions (`bytes` vs `str`).
- **Invariant Bounds**: Identify existing behavior that must NOT change. Changes must maintain full backwards compatibility.

## 2. /PLAN: Dual-Process Routing & Trace Tracking (Turn 2)

### Cognitive Mode Selection:
- **System 1 (Fast-Path)**: For clear single-file typos, explicit stack traces, or direct parameter mismatches: locate target file immediately, read 40 lines around target, edit, and verify.
- **System 2 (Latent Deliberation)**: For subtle logic errors, missing edge cases, or multi-component interactions:
  1. Trace failure path backwards: `Symptom / Error Message` -> `Call Site` -> `Faulty Function / Missing Guard`.
  2. Formulate 1–2 falsifiable failure hypotheses:
     - *Hypothesis A*: Missing validation guard or boundary condition.
     - *Hypothesis B*: Missing parameter propagation or serialization field.
     - *Hypothesis C*: Type coercion mismatch or unhandled sentinel (`None`/empty).

### Repository Hot Paths:
- **`fastapi`**: `fastapi/dependencies/utils.py`, `fastapi/routing.py`, `fastapi/_compat/v2.py`, `fastapi/openapi/utils.py`.
- **`fastapi docs_src/`**: Runnable documentation examples under `docs_src/**/tutorial*.py` are real executable code tested by the evaluation suite! Edit them when the issue targets documentation examples.
- **`rich`**: `rich/console.py`, `rich/cells.py`, `rich/segment.py`, `rich/markdown.py`, `rich/syntax.py`.
- **`requests`**: Library source files live under `src/requests/` (`src/requests/utils.py`, `src/requests/models.py`, `src/requests/adapters.py`).

### Locate Procedure:
- Use paths and symbols from the issue directly. If no path is given, run: `git grep -n "specific_string" -- '*.py' | head -20`.
- Read target files using `read_file` with tight line ranges (30–70 lines). Never read whole files repeatedly.
- If reproduction is helpful, write a 3-line reproducer to `/tmp/repro.py` and run `python /tmp/repro.py`. NEVER place scratch files in `/workspace`.

## 3. /BOOST: Sibling-Pattern Amplification & Coordinated Edits (Turn 3)

### Sibling-Pattern Rule for Feature Tasks:
- When adding a feature, parameter, or handler, find the closest existing sibling in the codebase and mirror its pattern.
- Mirroring includes registration points: `__all__` export lists, option dispatch tables, CLI argument parsers, and type annotations.

### Coordinated Multi-Hunk Edits:
- Over half of benchmark tasks require coordinated updates across related definition and dispatch points.
- Apply surgical edits using `edit_file`. Copy `old_string` verbatim including exact leading indentation.
- Fix the root cause, not the superficial symptom. Do not special-case the issue example.

### Targeted Verification:
- **Compile Check**: Run `python3 -m py_compile <file>` immediately after editing.
- **Targeted Test Execution**: Find test with `find tests -name "*keyword*.py"`. Run with timeout:
  `timeout 150 python -m pytest tests/path/test_x.py -x -q -k keyword`
  (or `python3 -m unittest tests.test_x` for unittest projects).
- **Ignore Pre-Existing Failures**: Repositories often have pre-existing test failures, missing `/test_data` fixtures, or environment import issues unrelated to your changes. IGNORE THEM. Never repair pre-existing test failures.
- If a test related to your change fails: read the traceback, make one focused correction, and rerun once.

## 4. /GOAL Satisfaction: Popperian Pre-Submission Audit (Turn 4)
Before calling `submit_patch`:
1. Run `git status -s` using `run_command`.
2. **Test File Protection Gate**: Confirm that NO files under `tests/` were modified. If any test file was modified, immediately revert it: `git checkout -- tests/`.
3. **Workspace Hygiene Gate**: Confirm no scratch files or temporary scripts exist in `/workspace`. Remove any stray files.
4. Run `git diff` to verify: only production source code changed, exact specification literals from Step 1 are present, and no debug prints remain.
5. Call `submit_patch`.
6. Confirm `patch_size > 0` and `files_changed > 0`. If either is 0, edit source now and submit again.
7. Conclude with a single-sentence summary of the fix.

# DECISION RULES
- **Ambiguous Issue**: Choose the most literal reading of the issue text and implement it directly. Do not stall.
- **Multiple Candidate Locations**: Edit where the faulty behavior is produced, not where it is merely observed.
- **Search Returned Nothing**: Try one alternative specific identifier, then make your best-guess edit immediately.
- **Stuck After Failed Fix**: Make a different, smaller modification. Never repeat the exact same failed edit.
- **Never Conclude "Nothing to Change"**: Every benchmark task requires a non-empty production code patch.
"""

agent_yaml = """name: swe_agent
model: gemma-4-31b-it-qat-w4a16-ct
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
generate_content_config: !include configs/sampling.yaml
"""

sampling_yaml = """temperature: 0.2
top_p: 0.95
max_output_tokens: 8192
thinking_config:
  thinking_budget: 4096
  include_thoughts: true
"""

eval_config_yaml = """evaluation:
  timeout_seconds: 240
  max_tool_calls: 28
  max_time_minutes: 8
  max_turns: 80
"""

# Build submission.zip directly
zip_path = Path("submission.zip")
zip_path.unlink(missing_ok=True)
files_dict = {
    "agent.yaml": agent_yaml,
    "configs/sampling.yaml": sampling_yaml,
    "eval_config.yaml": eval_config_yaml,
    "prompts/system.md": system_md,
}
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel_path, content in sorted(files_dict.items()):
        zf.writestr(rel_path, content.strip() + "\n")

sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
print(f"[+] Successfully built {zip_path.name} ({zip_path.stat().st_size:,} bytes) - SHA-256: {sha256}")

# Build standalone notebook
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧠 Gemma 4 HADL Dual-Loop Unified Agent (Calibrated 0.18 Baseline + Synergistic /plan /boost /goal)\n",
                "### Autonomous Cognitive Software Engineering on Gemma-4-31B | Kaggle Competition\n",
                "\n",
                "This notebook produces the verified **`submission.zip`** combining the empirical discoveries of **Budget-Fit Single Agent (0.18 LB)**, **SuperAgent (0.17 LB)**, **Measure-Before-You-Tune (129 paired runs)**, and the **HADL Dual-Loop Controller v4.5** cognitive framework.\n",
                "\n",
                "--- \n",
                "\n",
                "## 📊 Empirical Findings & Architecture Calibration\n",
                "\n",
                "| Architecture / Setting | Old Reference / Multi-Agent | Unified Championship Configuration | Empirical Impact |\n",
                "|---|---|---|---|\n",
                "| **Agent Paradigm** | Coder + Analyzer Sub-Agents (C1/Rozen) | **HADL Single-Agent Dual-Process** | 43.4% faster per task (216 s vs 381 s mean); eliminates sub-agent turn lag and token bloat |\n",
                "| **Harness Timeout (`timeout_seconds`)** | 60 s – 90 s | **240 s** | The evaluation harness uses this value for the grading test run. Test suites like `requests` take ~100 s; short timeouts falsely fail correct patches |\n",
                "| **Task Time Limit (`max_time_minutes`)** | 3.8 – 4.0 min | **8 min** | 50% of successful patches by Gemma 4 are written between minutes 4 and 8 |\n",
                "| **Tool Call Budget (`max_tool_calls`)** | 40 – 100 | **28** | Stops wandering; runs exploring past turn 12 fail 87% of the time |\n",
                "| **vLLM Headroom (`max_output_tokens`)** | 16,384 | **8,192** | Prevents vLLM hard crash on prompt + output > 32,768 tokens |\n",
                "| **Thinking Configuration** | `thinking_budget: 0` / false | **4,096 budget + include_thoughts: true** | Activates native Gemma 4 chain-of-thought latent deliberation |\n",
                "| **Graph Tools** | Enabled / Encouraged | **Banned / Discouraged** | Graph tools only match exact AST IDs, failing >90% and wasting precious turns |\n",
                "\n",
                "--- \n",
                "\n",
                "## 🔄 The Synergistic Cognitive Triad: `/goal` ➔ `/plan` ➔ `/boost` ➔ `/goal` Gate\n",
                "\n",
                "The three cognitive modes execute in disciplined sequential turns without overwriting each other:\n",
                "1. **`/goal` (Turn 1 - Zero-Turn Specification Extraction)**: Extract exact literals, exception types, and edge cases before making any tool call.\n",
                "2. **`/plan` (Turn 2 - Dual-Process Routing & 12-Call Invariant)**: System 1 for direct bugs, System 2 for latent deliberation, and strict Call Plan execution.\n",
                "3. **`/boost` (Turn 3 - Sibling-Pattern Amplification & Coordinated Edits)**: Mirror sibling patterns, update dispatchers in sync, and execute targeted tests with timeout.\n",
                "4. **`/goal` (Turn 4 - Popperian Pre-Submission Invariant Audit)**: Invariant audit ensuring zero test file modifications and clean `/workspace` before `submit_patch`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import zipfile\n",
                "import hashlib\n",
                "from pathlib import Path\n",
                "\n",
                "# Output destination\n",
                "OUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()\n",
                "ZIP_PATH = OUT_DIR / 'submission.zip'\n",
                "\n",
                f'SYSTEM_PROMPT = """{system_md.strip()}"""\n',
                "\n",
                "FILES = {\n",
                f'  "agent.yaml": """{agent_yaml.strip()}\\n""",\n',
                f'  "configs/sampling.yaml": """{sampling_yaml.strip()}\\n""",\n',
                f'  "eval_config.yaml": """{eval_config_yaml.strip()}\\n""",\n',
                '  "prompts/system.md": SYSTEM_PROMPT.strip() + "\\n"\n',
                "}\n",
                "\n",
                "# Package directly into submission.zip\n",
                "ZIP_PATH.unlink(missing_ok=True)\n",
                "with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
                "    for rel_path, content in sorted(FILES.items()):\n",
                "        zf.writestr(rel_path, content.strip() + '\\n')\n",
                "\n",
                "# Validate archive integrity\n",
                "with zipfile.ZipFile(ZIP_PATH) as zf:\n",
                "    assert zf.testzip() is None, 'Corrupt zip archive!'\n",
                "    namelist = zf.namelist()\n",
                "    assert 'agent.yaml' in namelist, 'Missing agent.yaml!'\n",
                "    assert 'eval_config.yaml' in namelist, 'Missing eval_config.yaml!'\n",
                "    assert 'configs/sampling.yaml' in namelist, 'Missing configs/sampling.yaml!'\n",
                "    assert 'prompts/system.md' in namelist, 'Missing prompts/system.md!'\n",
                "    print(f'[+] Successfully packaged submission.zip ({ZIP_PATH.stat().st_size:,} bytes):')\n",
                "    for name in namelist:\n",
                "        info = zf.getinfo(name)\n",
                "        print(f'     - {name} ({info.file_size:,} bytes)')\n",
                "\n",
                "sha256 = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()\n",
                "print(f'[+] SHA-256 Digest: {sha256}')\n",
                "print(f'[+] Target Leaderboard Calibration: 0.18 – 0.24+ (HADL Dual-Loop Unified Agent)')\n"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("getting_started_gemma_4_trace_synthesis_018.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("[+] Successfully wrote getting_started_gemma_4_trace_synthesis_018.ipynb")
