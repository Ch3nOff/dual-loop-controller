### CELL 29 ###
LOCALIZE = (
    "   - Call the `code_analyzer` tool with the full issue text first. It returns LOCATION / ROOT CAUSE / FIX PLAN. "
    "Verify its claim by reading those exact lines before editing.\n"
    if MULTI else
    "   - Use `search_similar_code` with a real symbol name (not a sentence) once you know one, and "
    "`get_code_neighbors` to follow callers and callees to where behaviour diverges.\n"
)

SYSTEM_PROMPT = f'''You are an autonomous senior Python engineer working inside a sandboxed checkout of a real open-source repository at /workspace.
Goal: resolve the issue in the user message with the smallest correct patch, then call `submit_patch`.

## Hard rules
- Never edit, add or delete tests, `conftest.py`, `pytest.ini`, CI or packaging files. Hidden tests are applied after you finish.
- Keep public APIs backward compatible unless the issue explicitly asks for a change.
- Scratch files go to /tmp only. Anything left in /workspace becomes part of your patch.
- The environment is pre-built: do not try to install packages.
- Always finish by calling `submit_patch`. A careful best-effort fix beats no patch.

## Reading the issue
- Many issues are pasted pull-request descriptions. Ignore the template parts (HTML comments, discussion links, checklists, AI disclaimers). The title, code snippets, error messages and API names are the real specification.
- The hidden tests usually add new test functions that exercise exactly what the issue describes. If the issue names a new parameter, function, class, option or message, use exactly that name and spelling, and implement it completely, including the edge cases it mentions.
- Runnable documentation examples (for example `docs_src/`) are real code that tests import. Change them when the issue is about them.

## Workflow
1. **Understand**: state the expected vs. actual behaviour to yourself in one or two sentences.
2. **Localize**:
{LOCALIZE}   - Extract every identifier, error message and file name from the issue and search for them: `grep -rn "<identifier>" --include=*.py . | head -30`.
   - Read only the lines you need (`read_file` with a line range or `sed -n 'START,ENDp' FILE`).
3. **Reproduce**: write a minimal script to /tmp/repro.py that shows the bug or the missing behaviour and run it with `python /tmp/repro.py`.
4. **Fix**: edit source files with `edit_file`. Copy `old_string` verbatim from the file, *including leading indentation*, and strip any line-number prefixes. Keep `old_string` short but unique. One logical change per edit. Fix the root cause, not the symptom, and also handle the edge cases the issue mentions.
5. **Verify**: run `python -m py_compile <file>` after every edit, rerun /tmp/repro.py, then run the closest existing tests: `python -m pytest <tests/path> -x -q` (narrow with `-k`).
6. **Submit**: run `git status` and `git diff`, make sure only intended source changes remain, then call `submit_patch`.

## Budget discipline
- Call `get_status` every ~8 tool calls. When less than 25% of turns or time remain, stop exploring and go straight to Fix → Verify → Submit.
- Keep outputs short: pipe through `head`, use `grep -n`, `pytest -q`. Never print whole large files.
- If an edit fails twice, re-read the exact lines and retry with a smaller unique snippet.

## Quality bar
- Match the surrounding code style, type hints and naming.
- Prefer a small, targeted change over a refactor. Most real fixes change one file and about a dozen lines. Touch other files only when the fix requires it.
'''

ANALYZER_PROMPT = '''You are `code_analyzer`, a read-only code navigation specialist. You never modify files.
Given an issue, find exactly where it must be fixed.

## Tools
- `run_command` for READ-ONLY commands only: `grep -rn`, `ls`, `sed -n`, `git log -p -S`
- `search_similar_code` with a real symbol name (not a sentence) to find related code
- `get_code_neighbors` to walk callers and callees
- `get_code_subgraph` to see how a few candidate symbols connect
- `read_file` with tight line ranges to confirm

## Method
1. Skip pull-request template boilerplate in the issue. Extract identifiers: function/class names, error messages, file paths, options, and any NEW names the issue asks for.
2. Search for each one, then follow the call chain until you reach the line where behaviour diverges from what the issue expects.
3. Confirm by reading the actual code. Never guess line numbers.

## Answer format (at most 250 words, nothing else)
LOCATION: <path>:<start>-<end> (<function or class>)
ROOT CAUSE: <one or two sentences>
FIX PLAN: <concrete change, using the exact names the issue uses>
RELATED: <other call sites or files needing the same change, or "none">
TESTS: <existing test files that exercise this code>
CONFIDENCE: high | medium | low
'''

def sampling_block():
    return {
        "temperature": CFG["temperature"], "top_p": CFG["top_p"], "top_k": CFG["top_k"],
        "max_output_tokens": CFG["max_output_tokens"],
        "thinking_config": {"thinking_budget": CFG["thinking_budget"], "include_thoughts": CFG["include_thoughts"]},
    }

display(HTML(f'<details style="margin:4px 0"><summary style="cursor:pointer;font-weight:600">📜 Coder prompt '
             f'<span style="color:{MUTED};font-weight:400">({len(SYSTEM_PROMPT.split())} words)</span></summary>'
             f'<pre style="font-family:ui-monospace,Menlo,Consolas,monospace;background:#f6f8fa;color:{INK};padding:10px 12px;border-radius:8px;font-size:12px;white-space:pre-wrap">{esc(SYSTEM_PROMPT)}</pre></details>'
             + (f'<details style="margin:4px 0"><summary style="cursor:pointer;font-weight:600">🔎 Analyzer prompt '
                f'<span style="color:{MUTED};font-weight:400">({len(ANALYZER_PROMPT.split())} words)</span></summary>'
                f'<pre style="font-family:ui-monospace,Menlo,Consolas,monospace;background:#f6f8fa;color:{INK};padding:10px 12px;border-radius:8px;font-size:12px;white-space:pre-wrap">{esc(ANALYZER_PROMPT)}</pre></details>' if MULTI else "")))

### CELL 30 ###
BUNDLE = WORK / "submission_bundle"
shutil.rmtree(BUNDLE, ignore_errors=True)

def write(rel, text):
    p = BUNDLE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")

sampling_yaml = yaml.safe_dump(sampling_block(), sort_keys=False)
write("configs/sampling.yaml", sampling_yaml)
write("prompts/system.md", SYSTEM_PROMPT)

coder_tools = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"]
if not MULTI:
    coder_tools += ["search_similar_code", "get_code_neighbors", "get_code_subgraph"]
tools_lines = [f"  - {t}" for t in coder_tools]

if MULTI:
    write("sub_agents/analyzer.md", ANALYZER_PROMPT)
    write("sub_agents/code_analyzer.yaml", "\n".join([
        "name: code_analyzer",
        f"model: {MODEL}",
        "description: Read-only code navigator. Give it the issue; it returns the files, root cause and fix plan.",
        "instruction: !include analyzer.md",
        "generate_content_config:",
        *["  " + l for l in sampling_yaml.rstrip().splitlines()],
        "tools:",
        *[f"  - {t}" for t in ["run_command", "read_file", "search_similar_code", "get_code_neighbors", "get_code_subgraph"]],
    ]))
    tools_lines += ["  - agent_tool:", "      config_path: sub_agents/code_analyzer.yaml", "      skip_summarization: true"]

write("agent.yaml", "\n".join([
    "name: pathfinder",
    f"model: {MODEL}",
    "description: Software engineer that fixes one repository issue with a small, verified patch.",
    "instruction: !include prompts/system.md",
    "generate_content_config: !include configs/sampling.yaml",
    "tools:", *tools_lines,
]))

if CFG["ship_eval_config"] and "eval_config.yaml" in SAMPLE_TEXT:
    write("eval_config.yaml", SAMPLE_TEXT["eval_config.yaml"])   # the official file, unchanged

tree = "\n".join(f"{'  ' * (len(p.relative_to(BUNDLE).parts) - 1)}{'📁' if p.is_dir() else '📄'} {p.name}"
                 for p in sorted(BUNDLE.rglob("*")))
yaml_blocks = "".join(f'<details style="margin:4px 0" open><summary style="cursor:pointer;font-weight:600">📄 {esc(p.relative_to(BUNDLE))}</summary>'
                      f'<pre style="font-family:ui-monospace,Menlo,Consolas,monospace;background:#f6f8fa;color:{INK};padding:10px 12px;border-radius:8px;font-size:12px">{esc(p.read_text())}</pre></details>'
                      for p in sorted(BUNDLE.rglob("*.yaml")))
display(HTML(f'<pre style="font-family:ui-monospace,Menlo,Consolas,monospace;background:{NAVY};color:#e8eef7;padding:12px 16px;border-radius:10px;font-size:13px">submission_bundle/\n{esc(tree)}</pre>{yaml_blocks}'))