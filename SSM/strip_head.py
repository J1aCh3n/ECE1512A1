import torch, sys, os

ckpt_in  = "/home/jiachen/1512/experiments/vssmtiny_dp01_ckpt_epoch_292.pth"
ckpt_out = "/home/jiachen/1512/experiments/vssm_tiny_pretrainv0.pth"

print(f"Loading: {ckpt_in}")
raw = torch.load(ckpt_in, map_location="cpu")

# Find a dict that actually contains tensors
state = None
for k in ("model", "state_dict", "ema", "model_ema", "module"):
    if isinstance(raw, dict) and k in raw and isinstance(raw[k], dict):
        state = raw[k]
        break
if state is None and isinstance(raw, dict):
    # maybe it's already a flat state dict
    state = raw
if state is None:
    raise RuntimeError("Could not find state_dict inside checkpoint")

# Drop classifier / head parameters so NUM_CLASSES can differ
drop_prefixes = ("classifier.head.", "head.")
kept = {}
dropped = []
for k, v in state.items():
    if any(k.startswith(p) for p in drop_prefixes):
        dropped.append(k)
        continue
    kept[k] = v

print(f"Dropped {len(dropped)} head params")
# Wrap back in a common format expected by many repos
to_save = {"model": kept}
torch.save(to_save, ckpt_out)
print(f"Saved: {ckpt_out} ({len(kept)} tensors kept)")
