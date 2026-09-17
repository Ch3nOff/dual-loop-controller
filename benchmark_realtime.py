"""
Real-Time Benchmark Runner for Dual-Loop Cognitive Controller v2.1
==================================================================
Runs authentic 20-benchmark evaluation with live visualization via:
  - Terminal mode: Rich ANSI colored output with progress bars and tables
  - Web mode: Local HTTP server with SSE-powered HTML dashboard

Usage:
  python benchmark_realtime.py --mode web --samples 10 --port 8765
  python benchmark_realtime.py --mode terminal --samples 5

All evaluations are 100% genuine PyTorch forward passes on frozen Qwen3.5-2B.
Zero mocked or fabricated data.
"""

import os
import sys
import time
import json
import argparse
import threading
import queue
import webbrowser
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Enable ANSI escape codes on Windows 10+
if sys.platform == "win32":
    os.system("")

# ==============================================================================
# SSE EVENT BUS (Thread-safe broadcast for web mode)
# ==============================================================================

class BenchmarkEventBus:
    """Thread-safe event bus for Server-Sent Events streaming."""

    def __init__(self):
        self.history = []
        self.subscribers = []
        self.lock = threading.Lock()

    def publish(self, event_type: str, data: dict):
        """Broadcast an event to all connected SSE clients."""
        event = {"type": event_type, "data": data, "ts": time.time()}
        with self.lock:
            self.history.append(event)
            for q in list(self.subscribers):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def subscribe(self):
        """Create a new subscriber queue with history replay."""
        q = queue.Queue(maxsize=500)
        with self.lock:
            for event in self.history:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    break
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


# ==============================================================================
# HTTP/SSE SERVER (Web mode)
# ==============================================================================

class SSERequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the dashboard HTML and SSE event stream."""

    event_bus = None
    html_path = None

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/events":
            self._serve_sse()
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def _serve_html(self):
        try:
            with open(self.html_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, "Dashboard HTML not found")

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = self.event_bus.subscribe()
        try:
            while True:
                try:
                    event = q.get(timeout=25)
                    sse_data = json.dumps(event["data"], ensure_ascii=False)
                    payload = f"event: {event['type']}\ndata: {sse_data}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    # Keepalive comment
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            self.event_bus.unsubscribe(q)

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logging


def start_web_server(port: int, html_path: str, event_bus: BenchmarkEventBus):
    """Start the HTTP/SSE server in a daemon thread."""
    SSERequestHandler.event_bus = event_bus
    SSERequestHandler.html_path = html_path
    server = HTTPServer(("0.0.0.0", port), SSERequestHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ==============================================================================
# TERMINAL RENDERER (ANSI colored terminal output)
# ==============================================================================

class TerminalRenderer:
    """Rich ANSI terminal output with progress indicators and live tables."""

    # ANSI Escape Codes
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    BG_GREEN = "\033[42m"
    BG_RED   = "\033[41m"

    BLOCK_FULL = "█"
    BLOCK_EMPTY = "░"

    def __init__(self):
        self.completed_benchmarks = []

    def _bar(self, pct, width=30):
        filled = int(pct / 100 * width)
        bar = self.GREEN + self.BLOCK_FULL * filled + self.GRAY + self.BLOCK_EMPTY * (width - filled) + self.RESET
        return bar

    def print_header(self, model_id, device, samples_per_task, total_benchmarks):
        line = "═" * 78
        print(f"\n{self.CYAN}{line}{self.RESET}")
        print(f"  {self.BOLD}🧠  DUAL-LOOP COGNITIVE CONTROLLER — REAL-TIME BENCHMARK{self.RESET}")
        print(f"  {self.GRAY}Model: {self.WHITE}{model_id}{self.GRAY}  |  Device: {self.WHITE}{device.upper()}{self.GRAY}  |  Layer: {self.WHITE}11{self.GRAY}  |  K={self.WHITE}2{self.RESET}")
        print(f"  {self.GRAY}Benchmarks: {self.WHITE}{total_benchmarks}{self.GRAY}  |  Samples/task: {self.WHITE}{samples_per_task}{self.GRAY}  |  Total items: {self.WHITE}{total_benchmarks * samples_per_task}{self.RESET}")
        print(f"{self.CYAN}{line}{self.RESET}\n")

    def print_model_loading(self):
        print(f"  {self.YELLOW}⏳{self.RESET}  Loading Qwen3.5-2B base model + Dual-Loop Controller...")

    def print_model_loaded(self, load_time, adapter_status):
        print(f"  {self.GREEN}✓{self.RESET}  Model loaded in {self.WHITE}{load_time:.1f}s{self.RESET} — Adapter: {adapter_status}\n")

    def print_benchmark_start(self, index, total, name, domain, total_items):
        print(f"\n  {self.CYAN}▶{self.RESET}  {self.BOLD}[{index:2d}/{total}]{self.RESET} {self.WHITE}{name}{self.RESET}")
        print(f"     {self.GRAY}Domain: {domain}  |  Items: {total_items}{self.RESET}")

    def print_item_result(self, item_idx, total_items, result):
        pct = ((item_idx + 1) / total_items * 100)
        bar = self._bar(pct, 20)

        base_mark = f"{self.GREEN}✓{self.RESET}" if result["base_ok"] else f"{self.RED}✗{self.RESET}"
        dl_mark = f"{self.GREEN}✓{self.RESET}" if result["delib_ok"] else f"{self.RED}✗{self.RESET}"

        status = result["status"]
        if "Rescued" in status:
            status_color = f"{self.GREEN}{self.BOLD}✨ {status}{self.RESET}"
        elif "Degraded" in status:
            status_color = f"{self.RED}{self.BOLD}⚠️  {status}{self.RESET}"
        else:
            status_color = f"{self.GRAY}{status}{self.RESET}"

        print(f"     {bar} {pct:5.1f}%  Item {item_idx+1:2d}/{total_items} | Base: {base_mark}({result['pred_base']}) → DL: {dl_mark}({result['pred_delib']}) | u={result['vacuity_u']:.3f} | {status_color}")

        # Spotlight detail for rescued or degraded items
        if "Rescued" in status or "Degraded" in status:
            prompt = result.get("prompt", "").replace("\nAnswer:", "").replace("Question: ", "").strip()
            choices = result.get("choices", [])
            labels = result.get("labels", [])
            print(f"     {self.CYAN}┌── [QUALITY SPOTLIGHT: {status}] ──────────────────────────────────{self.RESET}")
            print(f"     {self.CYAN}│{self.RESET} {self.BOLD}Question:{self.RESET} {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
            if choices and labels:
                c_str = " | ".join([f"[{l}] {c[:25]}" for l, c in zip(labels, choices)])
                print(f"     {self.CYAN}│{self.RESET} {self.GRAY}Choices :{self.RESET} {c_str}")
            print(f"     {self.CYAN}│{self.RESET} {self.RED}Base Model S1 (K=0)  :{self.RESET} {result['pred_base']} ({'Correct' if result['base_ok'] else 'INCORRECT'}) | Margin: {result.get('margin_base', 0):.3f}")
            print(f"     {self.CYAN}│{self.RESET} {self.GREEN}Dual-Loop S2 (K=2)   :{self.RESET} {result['pred_delib']} ({'CORRECT' if result['delib_ok'] else 'Incorrect'}) | Post-Margin: {result.get('margin_post', 0):.3f} | u={result['vacuity_u']:.3f}")
            print(f"     {self.CYAN}└───────────────────────────────────────────────────────────────────{self.RESET}")

    def print_benchmark_complete(self, name, base_acc, delib_acc, delta, rescued, degraded, elapsed, mean_vacuity):
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        delta_color = self.GREEN if delta > 0 else (self.RED if delta < 0 else self.GRAY)

        print(f"     {self.GRAY}{'─' * 60}{self.RESET}")
        print(f"     Result: Base={self.WHITE}{base_acc:.1f}%{self.RESET} → DL={self.GREEN}{delib_acc:.1f}%{self.RESET} | Δ={delta_color}{delta_str}{self.RESET} | R:{self.GREEN}{rescued}{self.RESET} D:{self.RED}{degraded}{self.RESET} | u̅={mean_vacuity:.3f} | ⏱ {elapsed:.1f}s")

    def print_running_totals(self, completed, total_benchmarks, macro_base, macro_delib, rescued, degraded, elapsed):
        delta = macro_delib - macro_base
        delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
        delta_color = self.GREEN if delta > 0 else (self.RED if delta < 0 else self.GRAY)

        print(f"\n  {self.GRAY}{'─' * 74}{self.RESET}")
        print(f"  {self.BOLD}📊 Running Totals ({completed}/{total_benchmarks} benchmarks){self.RESET}: "
              f"Base {self.WHITE}{macro_base:.2f}%{self.RESET} → DL {self.GREEN}{macro_delib:.2f}%{self.RESET} "
              f"({delta_color}{delta_str}{self.RESET}) | "
              f"R:{self.GREEN}+{rescued}{self.RESET} D:{self.RED}-{degraded}{self.RESET} | ⏱ {elapsed:.0f}s")

    def print_suite_complete(self, macro_base, macro_delib, delta, rescued, degraded, total_time, total_items):
        delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
        line = "═" * 78
        print(f"\n\n{self.CYAN}{line}{self.RESET}")
        print(f"  {self.BOLD}🏁  BENCHMARK SUITE COMPLETED{self.RESET}")
        print(f"{self.CYAN}{line}{self.RESET}")
        print(f"  Macro Base Accuracy   : {self.WHITE}{macro_base:.2f}%{self.RESET}")
        print(f"  Macro Dual-Loop Score : {self.GREEN}{self.BOLD}{macro_delib:.2f}%{self.RESET}")
        print(f"  Net Macro Delta       : {self.GREEN if delta >= 0 else self.RED}{self.BOLD}{delta_str}{self.RESET}")
        print(f"  Questions Rescued     : {self.GREEN}+{rescued}{self.RESET}")
        print(f"  Questions Degraded    : {self.RED if degraded > 0 else self.GRAY}{degraded}{self.RESET}")
        regression_status = f"{self.GREEN}{self.BOLD}ZERO REGRESSION ✓{self.RESET}" if degraded == 0 else f"{self.RED}REGRESSIONS DETECTED{self.RESET}"
        print(f"  Regression Status     : {regression_status}")
        print(f"  Total Items Evaluated : {total_items}")
        print(f"  Wall-Clock Time       : {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{self.CYAN}{line}{self.RESET}\n")


# ==============================================================================
# MAIN BENCHMARK RUNNER
# ==============================================================================

BENCHMARK_ORDER = [
    ("ARC-Easy", "Science & Facts", "Elementary Science QA"),
    ("ARC-Challenge", "Science & Facts", "Deep Scientific Deduction"),
    ("OpenBookQA", "Science & Facts", "Multi-Hop Fact Chaining"),
    ("PIQA", "Physical & Commonsense", "Physical Commonsense Dynamics"),
    ("BBH-LogicalDeduction", "Multi-Step Deductive Logic", "Relational Constraint Graphs"),
    ("BBH-DateUnderstanding", "Multi-Step Deductive Logic", "Temporal Calendar Arithmetic"),
    ("BBH-TrackingShuffledObjects", "Multi-Step Deductive Logic", "Sequential State Permutation"),
    ("BBH-BooleanExpressions", "Multi-Step Deductive Logic", "Nested Boolean Truth Logic"),
    ("BBH-CausalJudgement", "Physical & Commonsense", "Counterfactual Attribution"),
    ("BBH-FormalFallacies", "Formal Logic", "Syllogistic Entailment"),
    ("BBH-GeometricShapes", "Spatial & Symbolic", "SVG Geometry Parsing"),
    ("BBH-Hyperbaton", "Linguistic & Structural", "English Adjective Ordering"),
    ("BBH-Navigate", "Spatial & Symbolic", "Coordinate Navigation"),
    ("BBH-ColoredObjects", "Multi-Step Deductive Logic", "Multi-Attribute Binding"),
    ("BBH-WebOfLies", "Multi-Step Deductive Logic", "Alternating Parity Liar Chains"),
    ("Sector1-InvertedPhysics", "Counterfactual Simulation", "Inverted Physical Axioms"),
    ("Sector2-5HopTransitive", "Multi-Step Deductive Logic", "5-Hop Relational Constraints"),
    ("Sector3-CounterSyllogisms", "Formal Logic", "Counter-Intuitive Belief Bias"),
    ("Sector4-ModularCalendar", "Multi-Step Deductive Logic", "Modular Clock/Calendar Math"),
    ("Sector5-StateAutomata", "Spatial & Symbolic", "3-State DFA Machine Tracking"),
]


def run_realtime_benchmark(mode="web", samples_per_task=10, port=8765):
    """
    Main entry point: loads model, runs 20-benchmark suite with live output.
    
    Args:
        mode: 'web' for HTML dashboard + SSE, 'terminal' for ANSI terminal output
        samples_per_task: Number of samples per benchmark (default 10)
        port: HTTP server port for web mode (default 8765)
    """
    import torch
    import numpy as np
    import random

    # Deferred heavy imports
    print("[*] Importing evaluation modules...")
    from benchmark_full_20_suite import (
        load_20_benchmarks_suite, evaluate_item, set_seed,
        MODEL_ID, REVISION, ADAPTER_PATH
    )
    from dual_loop import attach_dual_loop_to_qwen
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Setup
    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_benchmarks = len(BENCHMARK_ORDER)

    # Initialize output handlers
    event_bus = None
    renderer = None
    server = None

    if mode == "web":
        event_bus = BenchmarkEventBus()
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_dashboard.html")
        if not os.path.exists(html_path):
            print(f"[!] ERROR: Dashboard HTML not found at: {html_path}")
            sys.exit(1)

        server = start_web_server(port, html_path, event_bus)
        print(f"[+] Dashboard server started at http://localhost:{port}")
        print(f"[*] Opening browser...")
        webbrowser.open(f"http://localhost:{port}")
        time.sleep(1)  # Give browser time to connect

        # Publish init event
        benchmark_list = [
            {"index": i + 1, "name": b[0], "category": b[1]}
            for i, b in enumerate(BENCHMARK_ORDER)
        ]
        event_bus.publish("init", {
            "model_id": MODEL_ID,
            "device": device,
            "total_benchmarks": total_benchmarks,
            "samples_per_task": samples_per_task,
            "benchmark_list": benchmark_list
        })
    else:
        renderer = TerminalRenderer()

    # ── Load Model ──────────────────────────────────────────────
    if renderer:
        renderer.print_header(MODEL_ID, device, samples_per_task, total_benchmarks)
        renderer.print_model_loading()

    t_load_start = time.time()

    print("[*] Loading Qwen3.5-2B base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    print("[*] Attaching Dual-Loop Cognitive Controller at Layer 11...")
    wrapped_model = attach_dual_loop_to_qwen(
        base_model,
        layer_idx=11,
        k_steps=2,
        enable_plasticity=True,
        use_evidential_gate=True,
        use_open_concept=True,
        use_surprise_gate=True,
        use_hypothesis_verification=True,
        use_contrastive_evidence=True
    )

    adapter_status = "not found"
    if os.path.exists(ADAPTER_PATH):
        print(f"[*] Loading calibrated deliberation weights from {ADAPTER_PATH}...")
        wrapped_model.load_adapter(ADAPTER_PATH, strict=False)
        adapter_status = "loaded ✓"

    wrapped_model.adapter.eval()
    wrapped_model.adapter.set_continual_mode(enabled=True, decay=0.92)

    load_time = time.time() - t_load_start

    if renderer:
        renderer.print_model_loaded(load_time, adapter_status)
    if event_bus:
        event_bus.publish("model_loaded", {
            "model_id": MODEL_ID,
            "load_time": load_time,
            "adapter_status": adapter_status,
            "device": device
        })

    # ── Load Benchmark Suite ────────────────────────────────────
    print(f"[*] Loading 20-benchmark suite ({samples_per_task} samples per task)...")
    suite = load_20_benchmarks_suite(samples_per_task=samples_per_task)

    # ── Run Evaluation Loop ─────────────────────────────────────
    total_base_correct = 0
    total_delib_correct = 0
    total_items = 0
    total_rescued = 0
    total_degraded = 0
    all_results = {}

    t_suite_start = time.time()

    for b_idx, (b_name, b_cat, b_domain) in enumerate(BENCHMARK_ORDER, 1):
        if b_name not in suite:
            print(f"  [!] Benchmark '{b_name}' not found in loaded suite, skipping.")
            continue

        b_data = suite[b_name]
        items = b_data["items"]
        n = len(items)

        # Notify benchmark start
        if renderer:
            renderer.print_benchmark_start(b_idx, total_benchmarks, b_name, b_domain, n)
        if event_bus:
            event_bus.publish("benchmark_start", {
                "index": b_idx,
                "name": b_name,
                "category": b_cat,
                "domain": b_domain,
                "total_items": n
            })

        b_base_correct = 0
        b_delib_correct = 0
        b_rescued = 0
        b_degraded = 0
        u_list = []

        t_bench_start = time.time()

        for item_idx, item in enumerate(items):
            result = evaluate_item(wrapped_model, tokenizer, item)

            if result["base_ok"]:
                b_base_correct += 1
            if result["delib_ok"]:
                b_delib_correct += 1
            if result["status"] == "Rescued (Wrong->Right)":
                b_rescued += 1
            if result["status"] == "Degraded (Right->Wrong)":
                b_degraded += 1
            u_list.append(result["vacuity_u"])

            # Notify item complete
            if renderer:
                renderer.print_item_result(item_idx, n, result)
            if event_bus:
                event_bus.publish("item_complete", {
                    "benchmark": b_name,
                    "item_index": item_idx,
                    "total_items": n,
                    "base_ok": result["base_ok"],
                    "delib_ok": result["delib_ok"],
                    "pred_base": result["pred_base"],
                    "pred_delib": result["pred_delib"],
                    "status": result["status"],
                    "vacuity_u": result["vacuity_u"],
                    "margin_base": result["margin_base"],
                    "margin_post": result["margin_post"],
                    "prompt": result.get("prompt", ""),
                    "choices": result.get("choices", []),
                    "labels": result.get("labels", []),
                    "scores_base": result.get("scores_base", []),
                    "combo_scores": result.get("combo_scores", []),
                    "target": result.get("target", "")
                })

        # Benchmark complete
        b_elapsed = time.time() - t_bench_start
        base_acc = (b_base_correct / n) * 100.0
        delib_acc = (b_delib_correct / n) * 100.0
        delta = delib_acc - base_acc
        mean_vacuity = float(np.mean(u_list)) if u_list else 0.0

        total_base_correct += b_base_correct
        total_delib_correct += b_delib_correct
        total_items += n
        total_rescued += b_rescued
        total_degraded += b_degraded

        all_results[b_name] = {
            "base_acc": base_acc, "delib_acc": delib_acc, "delta": delta,
            "rescued": b_rescued, "degraded": b_degraded,
            "mean_vacuity": mean_vacuity, "elapsed": b_elapsed
        }

        if renderer:
            renderer.print_benchmark_complete(b_name, base_acc, delib_acc, delta, b_rescued, b_degraded, b_elapsed, mean_vacuity)
        if event_bus:
            event_bus.publish("benchmark_complete", {
                "index": b_idx,
                "name": b_name,
                "base_acc": base_acc,
                "delib_acc": delib_acc,
                "delta": delta,
                "rescued": b_rescued,
                "degraded": b_degraded,
                "mean_vacuity": mean_vacuity,
                "elapsed": b_elapsed
            })

        # Print running totals in terminal mode
        if renderer:
            macro_base = (total_base_correct / total_items) * 100.0
            macro_delib = (total_delib_correct / total_items) * 100.0
            suite_elapsed = time.time() - t_suite_start
            renderer.print_running_totals(
                b_idx, total_benchmarks, macro_base, macro_delib,
                total_rescued, total_degraded, suite_elapsed
            )

    # ── Suite Complete ──────────────────────────────────────────
    total_time = time.time() - t_suite_start
    macro_base = (total_base_correct / total_items) * 100.0 if total_items > 0 else 0.0
    macro_delib = (total_delib_correct / total_items) * 100.0 if total_items > 0 else 0.0
    macro_delta = macro_delib - macro_base

    summary = {
        "macro_base": macro_base,
        "macro_delib": macro_delib,
        "delta": macro_delta,
        "total_rescued": total_rescued,
        "total_degraded": total_degraded,
        "total_items": total_items,
        "total_time": total_time
    }

    if renderer:
        renderer.print_suite_complete(
            macro_base, macro_delib, macro_delta,
            total_rescued, total_degraded, total_time, total_items
        )
    if event_bus:
        event_bus.publish("suite_complete", summary)

    # ── Save Results JSON ───────────────────────────────────────
    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "realtime_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": MODEL_ID,
            "device": device,
            "samples_per_task": samples_per_task,
            "summary": summary,
            "benchmarks": all_results
        }, f, indent=2)
    print(f"\n[+] Results saved to: {output_path}")

    # ── Keep server alive for web mode ──────────────────────────
    if mode == "web" and server:
        print(f"\n[*] Dashboard still available at http://localhost:{port}")
        print("[*] Press Ctrl+C to stop the server.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[*] Shutting down server...")
            server.shutdown()

    return summary


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dual-Loop Cognitive Controller — Real-Time Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_realtime.py --mode web --samples 10
  python benchmark_realtime.py --mode terminal --samples 5
  python benchmark_realtime.py --mode web --port 9000 --samples 10
        """
    )
    parser.add_argument(
        "--mode", choices=["web", "terminal"], default="web",
        help="Output mode: 'web' for HTML dashboard (default), 'terminal' for ANSI terminal"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of samples per benchmark task (default: 10, total = 20 × N)"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="HTTP server port for web mode (default: 8765)"
    )
    args = parser.parse_args()

    print("=" * 78)
    print("  DUAL-LOOP COGNITIVE CONTROLLER v2.1 — REAL-TIME BENCHMARK RUNNER")
    print("=" * 78)
    print(f"  Mode           : {args.mode.upper()}")
    print(f"  Samples/task   : {args.samples}")
    print(f"  Total items    : {20 * args.samples}")
    if args.mode == "web":
        print(f"  Dashboard URL  : http://localhost:{args.port}")
    print("=" * 78)
    print()

    try:
        run_realtime_benchmark(
            mode=args.mode,
            samples_per_task=args.samples,
            port=args.port
        )
    except KeyboardInterrupt:
        print("\n\n[!] Benchmark interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
