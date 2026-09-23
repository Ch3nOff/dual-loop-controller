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
    
    # 6. serve
    p_serve = subparsers.add_parser("serve", help="Launch HADL Cognitive Runtime Server & Interactive Cockpit")
    p_serve.add_argument("--model", type=str, default=None, help="Backbone model ID or path (auto-detects if None)")
    p_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    p_serve.add_argument("--mock", action="store_true", help="Launch lightweight mock engine for zero-download test")
    p_serve.add_argument("--k-steps", type=int, default=2, help="Deliberation steps (default: 2)")
    p_serve.add_argument("--bottleneck-dim", type=int, default=None, help="Latent bottleneck dimension")

    # 7. publish-hf
    p_pub = subparsers.add_parser("publish-hf", help="Package and upload HADL model adapters to Hugging Face Hub")
    p_pub.add_argument("--model-type", choices=["glm4", "qwen"], default="glm4", help="Model family to package (default: glm4)")
    p_pub.add_argument("--repo-id", type=str, default=None, help="Target Hugging Face repository ID")
    p_pub.add_argument("--token", type=str, default=None, help="Hugging Face access token with write permission")
    p_pub.add_argument("--package-only", action="store_true", help="Only assemble bundle without uploading")
    p_pub.add_argument("--output-dir", type=str, default=None, help="Local staging output directory")
    p_pub.add_argument("--private", action="store_true", help="Create private repository on HF Hub")

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

def cmd_serve(args):
    from .runtime.server import start_server
    start_server(
        model=args.model,
        host=args.host,
        port=args.port,
        mock=args.mock,
        k_steps=args.k_steps,
        bottleneck_dim=args.bottleneck_dim
    )

def cmd_publish_hf(args):
    from .runtime.hf_publisher import package_hf_bundle, publish_to_huggingface
    default_repo = "CH3NDev/dual-loop-glm4-9b-adapter" if args.model_type == "glm4" else "CH3NDev/dual-loop-qwen3.5-2b-adapter"
    target_repo = args.repo_id or default_repo

    bundle_dir = package_hf_bundle(model_type=args.model_type, output_dir=args.output_dir)
    if args.package_only:
        print(f"[*] Packaging complete. Staging folder: {bundle_dir}")
        return

    success = publish_to_huggingface(
        repo_id=target_repo,
        bundle_dir=bundle_dir,
        token=args.token,
        private=args.private
    )
    sys.exit(0 if success else 1)

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
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "publish-hf":
        cmd_publish_hf(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
