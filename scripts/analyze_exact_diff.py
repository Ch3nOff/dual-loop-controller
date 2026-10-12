import json
from pathlib import Path

audit_path = Path("eval_results/benchmark_200q_20parts_real_audit.json")
data = json.load(open(audit_path, "r", encoding="utf-8"))

prev = data["models_evaluated"]["model_sebelumnya"]["details"]
curr = data["models_evaluated"]["model_sekarang_autonomous"]["details"]
base = data["models_evaluated"]["model_dasar"]["details"]

print("=" * 80)
print("HEAD-TO-HEAD EXACT QUESTION DIFF (Model Sebelumnya vs Model Sekarang)")
print("=" * 80)

wins = []
losses = []

for b, p, c in zip(base, prev, curr):
    idx = c["idx"]
    bench = c["benchmark"]
    qid = c["id"]
    target = c["target"]
    p_ok = p["correct"]
    c_ok = c["correct"]
    b_ok = b["correct"]

    if not p_ok and c_ok:
        wins.append((idx, bench, qid, target, p, c))
    elif p_ok and not c_ok:
        losses.append((idx, bench, qid, target, p, c))

print(f"\n[+] SOAL DI MANA MODEL SEKARANG MENANG (+1 Kenaikan): {len(wins)} Soal")
for idx, bench, qid, target, p, c in wins:
    print(f"  * Q{idx:03d} | {bench:12s} | {qid:10s} | Target: {target}")
    print(f"      Model Sebelumnya : {repr(p['clean_output'][:70])} (FAIL)")
    print(f"      Model Sekarang   : {repr(c['clean_output'][:70])} (PASS)")

print(f"\n[-] SOAL DI MANA MODEL SEKARANG KALAH (-1 Penurunan): {len(losses)} Soal")
for idx, bench, qid, target, p, c in losses:
    print(f"  * Q{idx:03d} | {bench:12s} | {qid:10s} | Target: {target}")
    print(f"      Prompt           : {repr(c['prompt'][:80])}")
    print(f"      Model Sebelumnya : {repr(p['clean_output'][:70])} (PASS)")
    print(f"      Model Sekarang   : {repr(c['clean_output'][:70])} (FAIL)")

print("\n" + "=" * 80)
print(f"Net Delta: +{len(wins)} - {len(losses)} = {len(wins) - len(losses)} Poin")
print(f"Model Sebelumnya: {data['models_evaluated']['model_sebelumnya']['score']} ({data['models_evaluated']['model_sebelumnya']['accuracy_pct']}%)")
print(f"Model Sekarang  : {data['models_evaluated']['model_sekarang_autonomous']['score']} ({data['models_evaluated']['model_sekarang_autonomous']['accuracy_pct']}%)")
print("=" * 80)
