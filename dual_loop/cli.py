import sys
import argparse
import time
import json
import torch
from typing import Optional

from . import __version__
from .curiosity_daemon import PopperianSelfPlayEngine, AutonomousDaemonController
from .nullspace_engine import OrthogonalNullspaceProjector
from .allostasis import AllostaticEnergyModulator
from .homeostasis import HomeostaticDriveEngine
from .adapters.latent_adapter import LatentDeliberationAdapter

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dual-loop",
        description="Dual-Loop Cognitive Controller (HADL v3.0.0 Unified Cognitive OS) Command-Line Suite"
    )
    parser.add_argument("-v", "--version", action="version", version=f"dual-loop-controller {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # 1. info
    p_info = subparsers.add_parser("info", help="Display environment, module telemetry, and version diagnostics")
    
    # 2. benchmark
    p_bench = subparsers.add_parser("benchmark", help="Execute authentic PyTorch benchmark suites")
    p_bench.add_argument(
        "--suite",
        choices=["plasticity", "comprehensive", "halting", "qwen", "multimodal"],
        default="plasticity",
        help="Benchmark suite to execute (default: plasticity)"
    )
    p_bench.add_argument("--d-model", type=int, default=256, help="Latent dimensionality for benchmark (default: 256)")
    
    # 3. test
    p_test = subparsers.add_parser("test", help="Run comprehensive unit test suite")
    p_test.add_argument("-v", "--verbose", action="store_true", help="Verbose test runner output")
    
    # 4. verify-sandbox
    p_box = subparsers.add_parser("verify-sandbox", help="Evaluate expression or assertion script in deterministic sandbox")
    p_box.add_argument("code", type=str, help="Python expression or assertion code to evaluate")
    p_box.add_argument("--mode", choices=["eval", "exec", "syntax"], default="eval", help="Sandbox execution mode (default: eval)")
    p_box.add_argument("--allow-untrusted-exec", action="store_true", help="Explicitly allow assertion script execution from CLI (SEC-02)")
    
    # 5. daemon-step
    p_daemon = subparsers.add_parser("daemon-step", help="Run a single autonomous background contemplation cycle")
    p_daemon.add_argument("--slots", type=int, default=6, help="Number of synthetic memory slots (default: 6)")
    p_daemon.add_argument("--d-model", type=int, default=128, help="Latent dimension (default: 128)")

    # 6. sleep-cycle (NEW in v3.0)
    p_sleep = subparsers.add_parser("sleep-cycle", help="Execute offline sleep-phase consolidation replay and SVD distillation")
    p_sleep.add_argument("--episodes", type=int, default=5, help="Number of synthetic waking episodes to consolidate (default: 5)")
    p_sleep.add_argument("--rank", type=int, default=16, help="Consolidation SVD truncation rank (default: 16)")

    # 7. publish-hf
    p_pub = subparsers.add_parser("publish-hf", help="Package and upload HADL model adapters to Hugging Face Hub")
    p_pub.add_argument("--model-type", choices=["glm4", "qwen"], default="glm4", help="Model family to package (default: glm4)")
    p_pub.add_argument("--repo-id", type=str, default=None, help="Target Hugging Face repository ID")
    p_pub.add_argument("--token", type=str, default=None, help="Hugging Face access token (SEC-08: Prefer HF_TOKEN env var to avoid exposure)")
    p_pub.add_argument("--package-only", action="store_true", help="Only assemble bundle without uploading")
    p_pub.add_argument("--output-dir", type=str, default=None, help="Local staging output directory")
    p_pub.add_argument("--private", action="store_true", help="Create private repository on HF Hub")

    # 8. validate-benchmark
    p_val = subparsers.add_parser("validate-benchmark", help="Validate benchmark JSON files for mathematical and structural integrity")
    p_val.add_argument("path", nargs="?", default="eval_results", help="File or directory of benchmark JSON files to validate (default: eval_results)")
    p_val.add_argument("--quarantine", action="store_true", help="Automatically quarantine failing files")
    p_val.add_argument("--quarantine-dir", type=str, default="eval_results/archive_deprecated", help="Quarantine directory")

    # 9. serve (OpenAI-compatible inference server with VRAM auto-tuning)
    p_serve = subparsers.add_parser("serve", help="Launch OpenAI-compatible inference server with dynamic VRAM auto-tuning")
    p_serve.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID or path (default: Qwen/Qwen2.5-7B-Instruct)")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    p_serve.add_argument("--headroom", type=str, default="auto", help="VRAM safety headroom in GiB (e.g. 4.0 or auto, default: auto)")
    p_serve.add_argument("--regime", type=str, choices=["auto", "bf16", "int8", "nf4", "hologram"], default="auto", help="Force specific quantization/hologram regime")
    p_serve.add_argument("--k-steps", type=int, default=2, help="Number of latent deliberation steps (default: 2)")
    p_serve.add_argument("--trust-remote-code", action="store_true", help="Trust remote code in HF model loaders (default: False)")
    p_serve.add_argument("--log-level", type=str, default="info", help="Uvicorn log level (default: info)")

    return parser

def cmd_info(args):
    print("=" * 78)
    print(f"  DUAL-LOOP COGNITIVE CONTROLLER (HADL) v{__version__}")
    print("  Unified Cognitive Operating System (5 Computational Brain Organs)")
    print("=" * 78)
    print(f"[*] Package Version    : {__version__}")
    print(f"[*] PyTorch Version    : {torch.__version__}")
    print(f"[*] CUDA Available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] CUDA Device Name   : {torch.cuda.get_device_name(0)}")
    print(f"[*] Active Architecture: Unified Cognitive OS (HADL v3.0.0)")
    print(f"[*] The 5 Computational Brain Organs:")
    print("    1. Multi-Time-Scale Dynamics        (Cognitive Conflict Gate, 80% Bypass FLOPs)")
    print("    2. Allostasis & Active Inference    (Vital State S_t, Expected Free Energy G(pi))")
    print("    3. Global Workspace Theory (GWT)    (L2 SRAM Spotlight, D_c=1024 Fixed Manifold)")
    print("    4. Fast Plasticity & Sleep Replay   (Hebbian M_fast -> SVD Offline Distillation)")
    print("    5. Sheaf Invariant Firewall         (Sub-0.05ms Prefrontal Executive Inhibition)")
    print(f"[*] Key Invariant Bridges:")
    print("    - Universal Model-Agnostic Adapter  (LLaMA, Gemma, Qwen, Mistral, DeepSeek)")
    print("    - Dynamic Runtime Graph Introspector(Auto-Discovers Layers & Native D_model)")
    print("    - Canonical Latent Projection       (R^D_native -> R^1024 -> R^D_native ReZero)")
    print("=" * 78)

def cmd_verify_sandbox(args):
    # SEC-02 hardening: restrict untrusted CLI exec
    if args.mode == "exec" and not getattr(args, "allow_untrusted_exec", False):
        print("=" * 68)
        print("[SECURITY BLOCKED] Direct 'exec' evaluation via untrusted CLI is disabled by default (SEC-02).")
        print("To safely verify expressions, use: dual-loop verify-sandbox '<expr>' --mode eval")
        print("To verify syntax, use: dual-loop verify-sandbox '<code>' --mode syntax")
        print("To force assertion execution from CLI, pass flag: --allow-untrusted-exec")
        print("=" * 68)
        sys.exit(1)

    print(f"[*] Executing sandbox verification in mode: '{args.mode}'")
    print(f"[*] Code snippet: {args.code}")
    is_valid, diag = PopperianSelfPlayEngine.verify_sandbox(args.code, test_condition=args.mode)
    print("-" * 60)
    print(f"[*] Result Valid / Survived : {is_valid}")
    print(f"[*] Diagnostic Output       : {diag}")
    print("-" * 60)
    sys.exit(0 if is_valid else 1)

def cmd_daemon_step(args):
    print(f"[*] Initializing Autonomous Background Daemon (D={args.d_model}, Slots={args.slots})...")
    daemon = AutonomousDaemonController(d_model=args.d_model)
    
    # Generate synthetic slots with contradiction between slot 0 and 1
    slots = torch.randn(args.slots, args.d_model)
    slots[1] = -slots[0] * 1.5
    
    print("[*] Running background contemplation step...")
    result = daemon.run_daemon_step(slots)
    print("-" * 60)
    print(json.dumps(result, indent=2))
    print("-" * 60)
    print(f"[OK] Contemplation completed in {result.get('cycle_latency_ms', 0.0):.2f} ms")

def cmd_benchmark(args):
    if args.suite == "plasticity":
        from .benchmarks.epistemic_plasticity_benchmark import main as run_plasticity
        run_plasticity()
    elif args.suite == "comprehensive":
        from .benchmarks.comprehensive_suite import main as run_comp
        run_comp()
    elif args.suite == "halting":
        from .benchmarks.halting_audit import main as run_halt
        run_halt()
    elif args.suite == "qwen":
        from .benchmarks.benchmark_qwen_reasoning import main as run_qwen
        run_qwen()
    elif args.suite == "multimodal":
        from .benchmarks.benchmark_bidirectional_multimodal import main as run_multimodal
        run_multimodal()
    else:
        print(f"Unknown benchmark suite: {args.suite}")
        sys.exit(1)

def cmd_test(args):
    import unittest
    import os
    print(f"[*] Discovering and executing unit tests...")
    loader = unittest.TestLoader()
    tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
    if not os.path.exists(tests_dir):
        # Fallback to local tests
        tests_dir = "tests"
    suite = loader.discover(tests_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    res = runner.run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)

def cmd_publish_hf(args):
    from .runtime.hf_publisher import package_hf_bundle, publish_to_huggingface
    default_repo = "CH3NDev/dual-loop-glm4-9b-adapter" if args.model_type == "glm4" else "CH3NDev/dual-loop-qwen3.5-2b-adapter"
    target_repo = args.repo_id or default_repo

    bundle_dir = package_hf_bundle(model_type=args.model_type, output_dir=args.output_dir)
    if args.package_only:
        print(f"[*] Packaging complete. Staging folder: {bundle_dir}")
        return

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and not args.package_only:
        print("[*] Note: No HF token passed via CLI; checked HF_TOKEN environment variable.")

    success = publish_to_huggingface(
        repo_id=target_repo,
        bundle_dir=bundle_dir,
        token=token,
        private=args.private
    )
    sys.exit(0 if success else 1)

def cmd_validate_benchmark(args):
    import os
    import shutil
    from .validation.benchmark_validator import BenchmarkValidator, _print_result, validate_benchmark_directory
    validator = BenchmarkValidator()
    
    target = args.path
    if os.path.isfile(target):
        res = validator.validate_file(target)
        _print_result(res)
        if not res.is_valid and args.quarantine:
            os.makedirs(args.quarantine_dir, exist_ok=True)
            dest = os.path.join(args.quarantine_dir, os.path.basename(target))
            shutil.move(target, dest)
            print(f"\n[QUARANTINED] Moved {target} -> {dest}")
        sys.exit(0 if res.is_valid else 1)
    elif os.path.isdir(target):
        passed, failed = validate_benchmark_directory(
            target,
            validator=validator,
            quarantine_invalid=args.quarantine,
            quarantine_dir=args.quarantine_dir
        )
        print("=" * 80)
        print(f"  HADL BENCHMARK INTEGRITY AUDIT: {target}")
        print("=" * 80)
        print(f"[*] Total Audited Files: {len(passed) + len(failed)}")
        print(f"[+] PASSED             : {len(passed)}")
        print(f"[-] FAILED / REJECTED  : {len(failed)}")
        print("-" * 80)
        if failed:
            print("\nFAILURES / ARITHMETIC REJECTIONS:")
            for f_res in failed:
                _print_result(f_res)
            sys.exit(1)
        else:
            print("\nALL FILES PASSED MATHEMATICAL INTEGRITY AUDIT.")
            sys.exit(0)
    else:
        print(f"Error: Path '{target}' not found.")
        sys.exit(1)

def cmd_sleep_cycle(args):
    from .sleep_consolidation import SleepPhaseConsolidationEngine
    print(f"[*] Initializing Sleep-Phase Consolidation Engine (D=1024, Rank={args.rank})...")
    engine = SleepPhaseConsolidationEngine(d_model=1024, rank=args.rank)
    print(f"[*] Simulating {args.episodes} waking episodes with synthetic surprises...")
    for i in range(args.episodes):
        v = torch.randn(1, 1024)
        u = torch.randn(1, 1024)
        engine.record_episode(v, u, surprise_score=0.85 + i * 0.05, reward_weight=1.0)
    print("[*] Running offline truncated SVD consolidation and QR nullspace orthogonalization...")
    telem = engine.consolidate()
    print("-" * 60)
    print(json.dumps(telem, indent=2))
    print("-" * 60)
    print(f"[OK] Sleep consolidation completed in {telem.get('latency_ms', 0.0):.2f} ms")

def cmd_serve(args):
    from .server import start_server
    headroom = "auto" if str(args.headroom).lower() == "auto" else float(args.headroom)
    regime = None if str(args.regime).lower() == "auto" else args.regime.upper()
    start_server(
        model_id_or_path=args.model,
        host=args.host,
        port=args.port,
        headroom_gib=headroom,
        forced_regime=regime,
        k_steps=args.k_steps,
        trust_remote_code=args.trust_remote_code,
        log_level=args.log_level
    )

def main():
    parser = get_parser()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    if args.command == "info":
        cmd_info(args)
    elif args.command == "verify-sandbox":
        cmd_verify_sandbox(args)
    elif args.command == "daemon-step":
        cmd_daemon_step(args)
    elif args.command == "sleep-cycle":
        cmd_sleep_cycle(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "publish-hf":
        cmd_publish_hf(args)
    elif args.command == "validate-benchmark":
        cmd_validate_benchmark(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
