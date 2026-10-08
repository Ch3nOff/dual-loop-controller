import os
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Supported archs:", torch.cuda.get_arch_list())
    try:
        x = torch.randn(10, 10, device="cuda")
        y = x @ x
        print("CUDA matmul success! Output shape:", y.shape)
    except Exception as e:
        print("CUDA execution failed:", e)
