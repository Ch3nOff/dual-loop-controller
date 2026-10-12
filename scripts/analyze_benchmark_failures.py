import json
from pathlib import Path

audit_path = Path("eval_results/benchmark_100q_20parts_real_audit.json")
data = json.load(open(audit_path, encoding="utf-8"))

auto_d = data["models_evaluated"]["model_sekarang_autonomous"]["details"]
prev_d = data["models_evaluated"]["model_sebelumnya"]["details"]
base_d = data["models_evaluated"]["model_dasar"]["details"]

print("=" * 80)
print("AUDIT KEGAGALAN / PENURUNAN MODEL SEKARANG (10 DARI 100 SOAL)")
print("=" * 80)

for idx in range(len(auto_d)):
    a = auto_d[idx]
    p = prev_d[idx]
    b = base_d[idx]
    
    # Check if auto failed
    if not a["correct"]:
        status = "TURUN VS SEBELUMNYA" if p["correct"] else ("SAMA-SAMA GAGAL" if not p["correct"] else "")
        print(f"\n[{a['benchmark']}] Soal #{a['idx']} (ID: {a['id']}) - Status: {status}")
        print(f"  Pillar   : {a['pillar']}")
        print(f"  Target   : {a['target']}")
        print(f"  Model Sekarang (Auto) : '{a['prediction']}' (Correct: {a['correct']})")
        print(f"  Model Sebelumnya      : '{p['prediction']}' (Correct: {p['correct']})")
        print(f"  Model Dasar           : '{b['prediction']}' (Correct: {b['correct']})")

print("\n" + "=" * 80)
print("SOAL DIMANA MODEL SEKARANG NAIK (LEBIH UNGGUL DARI SEBELUMNYA):")
print("=" * 80)
for idx in range(len(auto_d)):
    a = auto_d[idx]
    p = prev_d[idx]
    if a["correct"] and not p["correct"]:
        print(f"\n[{a['benchmark']}] Soal #{a['idx']} (ID: {a['id']}) - NAIK!")
        print(f"  Target   : {a['target']}")
        print(f"  Model Sekarang : '{a['prediction']}'")
        print(f"  Model Sebelumnya: '{p['prediction']}'")
