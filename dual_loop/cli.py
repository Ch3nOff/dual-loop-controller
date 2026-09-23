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
        description="Dual-Loop Cognitive Controller (HADL v2.4.0) Command-Line Suite"
    )
    parser.add_argument("-v", "--version", action="version", version=f"dual-loop-controller {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # 1. info
    p_info = subparsers.add_parser("info", help="Display environment, module telemetry, and version diagnostics")
    
    # 2. benchmark
    p_bench = subparsers.add_parser("benchmark", help="Execute authentic PyTorch benchmark suites")
    p_bench.add_argument(
        "--suite",
        choices=["plasticity", "comprehensive", "halting", "qwen"],
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
    
    # 5. daemon-step
    p_daemon = subparsers.add_parser("daemon-step", help="Run a single autonomous background contemplation cycle")
    p_daemon.add_argument("--slots", type=int, default=6, help="Number of synthetic memory slots (default: 6)")
    p_daemon.add_argument("--d-model", type=int, default=128, help="Latent dimension (default: 128)")
    
    return parser

def cmd_info(args):
    print("=" * 78)
    print(f"  DUAL-LOOP COGNITIVE CONTROLLER (HADL) v{__version__}")
    print("  Hardware-Aligned Latent Deliberation & Autonomous Inference")
    print("=" * 78)
    print(f"[*] Package Version    : {__version__}")
    print(f"[*] PyTorch Version    : {torch.__version__}")
    print(f"[*] CUDA Available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] CUDA Device Name   : {torch.cuda.get_device_name(0)}")
    print(f"[*] Active Architecture: Autopoietic Dual-Process Engine")
    print(f"[*] Key Components     :")
    print("    - Consolidated Allostatic Energy Modulator  (Gate Pruning, sub-5ms)")
    print("    - Decoupled Autonomous Curiosity Daemon     (ICM + Popperian Self-Play)")
    print("    - Popperian Deterministic Execution Sandbox (Safe AST/builtins exec & eval)")
    print("    - Epistemic Humility Module                 (c <= 0.95, Hyperbolic Odds Loss)")
    print("    - Orthogonal Nullspace Memory Engine        (QR Nullspace Projection, col=0)")
    print("    - Parsimony-Driven Plan Selector            (MDL-inspired L1 + variance proxy)")
    print("    - Functorial Cross-Domain Mapper            (Relational morphism graph alignment)")
    print("=" * 78)

def cmd_verify_sandbox(args):
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
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
