import torch, os
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available(), " device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("torch.version.cuda:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version(), " enabled:", torch.backends.cudnn.enabled)

mods = []
for name in ["selective_scan_cuda", "mamba_ssm", "causal_conv1d"]:
    try:
        m = __import__(name)
        print(f"[OK] import {name} ->", getattr(m, "__file__", "builtin"))
        mods.append(name)
    except Exception as e:
        print(f"[FAIL] import {name} ->", e)
print("loaded:", mods)