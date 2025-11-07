from transformers import CLIPVisionModel
import torch
import torch.nn as nn

# Load ViT-L/14 encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
model_name = "openai/clip-vit-large-patch14"
vision_tower = CLIPVisionModel.from_pretrained(model_name).to(device).eval().to(dtype)

# Projectors
class LinearProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
    def forward(self, x): return self.linear(x)

class LowRankProjector(nn.Module):
    def __init__(self, d_in, d_out, r=256):
        super().__init__()
        self.W1 = nn.Linear(d_in, r, bias=False)
        self.W2 = nn.Linear(r, d_out, bias=False)
    def forward(self, x): return self.W2(self.W1(x))

class MLPProjector(nn.Module):
    def __init__(self, d_in, d_out, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out)
        )
    def forward(self, x): return self.net(x)

class GatedProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.Wp = nn.Linear(d_in, d_out)
        self.Wg = nn.Linear(d_in, d_out)
    def forward(self, x):
        gate = torch.sigmoid(self.Wg(x))
        return gate * self.Wp(x)

class LoRAProjector(nn.Module):
    def __init__(self, d_in, d_out, r=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.A = nn.Linear(d_in, r, bias=False)
        self.B = nn.Linear(r, d_out, bias=False)
        self.scaling = alpha / r
    def forward(self, x): return self.linear(x) + self.B(self.A(x)) * self.scaling

# Instantiate all projectors
projectors = {
    "Linear": LinearProjector(1024, 4096),
    "LowRank": LowRankProjector(1024, 4096),
    "MLP": MLPProjector(1024, 4096),
    "Gated": GatedProjector(1024, 4096),
    "LoRA": LoRAProjector(1024, 4096)
}

# Helper functions
def count_params(model):
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())

def param_memory_mb(model, dtype=torch.float16):
    """Compute parameter memory in MB given dtype precision."""
    bytes_per_param = torch.finfo(dtype).bits / 8
    return count_params(model) * bytes_per_param / (1024**2)

# Generate dummy inputs
num_samples = 10
inputs = torch.randn(num_samples, 3, 224, 224, device=device, dtype=dtype)

with torch.no_grad():
    feats = []
    for i in range(num_samples):
        feat = vision_tower(inputs[i:i+1]).pooler_output
        feats.append(feat)
    feats = torch.cat(feats, dim=0)

# Display
print("\nProjector Parameter Summary (averaged over multiple inputs)")
print("-------------------------------------------------------------")
print(f"{'Projector':<10} | {'Params(M)':>9} | {'Mem(MB)':>8}")
print("-------------------------------------------------------------")

for name, proj in projectors.items():
    proj = proj.to(device).to(dtype).eval()
    params = count_params(proj)
    mem = param_memory_mb(proj, dtype)
    print(f"{name:<10} | {params/1e6:>9.3f} | {mem:>8.2f}")
