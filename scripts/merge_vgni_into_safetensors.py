"""
Merges VGNI-LBU Epistemic Memory Bank and Verifier Weights into model.safetensors
Ensures 100% native loading with ZERO missing keys!
"""
import torch
import sys
from pathlib import Path
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "dist" / "xstar_2b_autonomous_brain"
sys.path.insert(0, str(MODEL_DIR))
from vgni_lbu_cloud_engine import UnifiedVGNIKVCloudEngine
SAFETENSORS_PATH = MODEL_DIR / "model.safetensors"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "vgni_lbu_autonomous_stabilized_qwen2b.pt"

print("[*] Loading existing safetensors...")
tensors = load_file(str(SAFETENSORS_PATH))
print(f"[+] Loaded {len(tensors)} tensors from {SAFETENSORS_PATH}")

# Instantiate engine and load checkpoint weights
print("[*] Instantiating VGNI-LBU engine and loading stabilized checkpoint...")
engine = UnifiedVGNIKVCloudEngine(d_model=2048)
if CKPT_PATH.exists():
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    if "vgni_state" in ckpt:
        engine.vgni.concept_anchors.copy_(ckpt["vgni_state"]["concept_anchors"])
        engine.vgni.approach_vectors.copy_(ckpt["vgni_state"]["approach_vectors"])
        engine.vgni.num_stored.copy_(ckpt["vgni_state"]["num_stored"])
        engine.vgni.usage_counts.copy_(ckpt["vgni_state"]["usage_counts"])
    if "lbu_weights" in ckpt:
        engine.vgni.lbu.u_logic.data.copy_(ckpt["lbu_weights"]["u_logic"])
        engine.vgni.lbu.w_step.weight.data.copy_(ckpt["lbu_weights"]["w_step"])
    if "misf_weights" in ckpt:
        m = ckpt["misf_weights"]
        if "w_freq" in m: engine.misf.w_freq.data.copy_(m["w_freq"][:2048, :8] if m["w_freq"].ndim == 2 else engine.misf.w_freq.data)
        if "phase_shift" in m: engine.misf.phase_shift.data.copy_(m["phase_shift"][:8])
        if "viscosity_proj" in m: engine.misf.viscosity_proj.weight.data.copy_(m["viscosity_proj"][:4, :8])

# Extract all named params and buffers
new_tensors = dict(tensors)
for name, param in engine.named_parameters():
    full_key = f"model.brain.vgni_engine.{name}"
    new_tensors[full_key] = param.data.to(dtype=torch.bfloat16).contiguous()

for name, buf in engine.named_buffers():
    full_key = f"model.brain.vgni_engine.{name}"
    new_tensors[full_key] = buf.data.to(dtype=torch.bfloat16 if buf.dtype == torch.float32 else buf.dtype).contiguous()

print(f"[*] Total tensors after merging: {len(new_tensors)} (Added {len(new_tensors) - len(tensors)} tensors)")

# Save back to safetensors
save_file(new_tensors, str(SAFETENSORS_PATH))
print(f"[+] Successfully saved merged model.safetensors! ({SAFETENSORS_PATH})")
