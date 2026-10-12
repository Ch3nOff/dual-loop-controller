import json

with open("eval_results/benchmark_200q_20parts_real_audit.json", "r", encoding="utf-8") as f:
    d = json.load(f)

base = d["models_evaluated"]["model_dasar"]["details"]
prev = d["models_evaluated"]["model_sebelumnya"]["details"]
auto = d["models_evaluated"]["model_sekarang_autonomous"]["details"]

print("=== PER-BENCHMARK COMPARISON (20 BENCHMARKS) ===")
for b, bc in d["benchmark_comparison"].items():
    delta = bc["auto"] - bc["prev"]
    sign = f"+{delta}" if delta > 0 else str(delta)
    print(f"{b:12s} | Base: {bc['base']:2d}/10 | Prev: {bc['prev']:2d}/10 | Auto: {bc['auto']:2d}/10 | Delta vs Prev: {sign}")

print("\n=== DIFFERENCES: PREV vs AUTO ===")
for i in range(len(base)):
    p_corr = prev[i]["correct"]
    a_corr = auto[i]["correct"]
    if p_corr != a_corr:
        qid = auto[i]["id"]
        b = auto[i]["benchmark"]
        p_str = "PASS" if p_corr else "FAIL"
        a_str = "PASS" if a_corr else "FAIL"
        print(f"[{qid}] ({b}): Prev={p_str} -> Auto={a_str}")
        print(f"   Target : {auto[i]['target']}")
        print(f"   Auto Out: {auto[i]['clean_output'][:80]}")

print("\n=== ALL FAILURES FOR AUTO (MODEL SEKARANG) ===")
for i, item in enumerate(auto):
    if not item["correct"]:
        print(f"FAIL #{i+1:03d} [{item['id']}] ({item['benchmark']}):")
        print(f"   Prompt : {item['prompt'][:60]}...")
        print(f"   Target : {item['target']}")
        print(f"   Output : {item['clean_output'][:80]}")
