import torch
from dual_loop import DualLoopTransformer
from dual_loop.benchmarks import MultiHopGraphDataset

def test_inference_halting():
    model = DualLoopTransformer(vocab_size=19, d_model=64, num_cwm_slots=12, max_ponder_steps=3, entropy_threshold=1.25)
    state_dict = torch.load('checkpoint_trained_dualloop.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset = MultiHopGraphDataset(num_samples=500, num_nodes=16, num_edges=6, hops=3)
    x, y_all = dataset.get_batch(500)
    y = y_all[:, -1]

    logits, info = model(x, dynamic_halting=True)
    acc = (logits.argmax(-1) == y).float().mean().item() * 100.0
    st = info['steps_taken']

    print(f"Accuracy with dynamic_halting=True : {acc:.1f}%")
    print(f"Effective K (Average Ponder Steps) : {info['effective_k']:.2f} steps")
    print(f"Ponder Distribution across Batch   : K=1: {(st==1).float().mean()*100:.1f}%, K=2: {(st==2).float().mean()*100:.1f}%, K=3: {(st==3).float().mean()*100:.1f}%")

if __name__ == "__main__":
    test_inference_halting()
