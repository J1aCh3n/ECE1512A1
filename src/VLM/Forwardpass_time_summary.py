import numpy as np
from transformers import CLIPVisionModel, CLIPImageProcessor
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt

# Load ViT-L/14 encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-large-patch14"
vision_tower = CLIPVisionModel.from_pretrained(model_name).to(device).half()
vision_tower.eval()

# Projectors
class LinearProjector(torch.nn.Module):
    def __init__(self, d_in, d_out):
      super().__init__()
      self.linear = torch.nn.Linear(d_in, d_out)
    def forward(self, x): return self.linear(x)

class LowRankProjector(torch.nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.W1 = torch.nn.Linear(d_in, r, bias=False)
        self.W2 = torch.nn.Linear(r, d_out, bias=False)
    def forward(self, x): return self.W2(self.W1(x))

class MLPProjector(torch.nn.Module):
    def __init__(self, d_in, d_out, hidden=2048):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d_out))
    def forward(self, x): return self.net(x)

class GatedProjector(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.Wp = torch.nn.Linear(d_in, d_out)
        self.Wg = torch.nn.Linear(d_in, d_out)
    def forward(self, x): return torch.sigmoid(self.Wg(x)) * self.Wp(x)

class LoRAProjector(torch.nn.Module):
    def __init__(self, d_in, d_out, r=8, alpha=16):
        super().__init__()
        self.linear = torch.nn.Linear(d_in, d_out)
        self.A = torch.nn.Linear(d_in, r, bias=False)
        self.B = torch.nn.Linear(r, d_out, bias=False)
        self.scaling = alpha / r
    def forward(self, x): return self.linear(x) + self.B(self.A(x)) * self.scaling

# Initialize projectors
d_in, d_out = 1024, 4096
projectors = {
    "Linear": LinearProjector(d_in, d_out).half().to(device),
    "LowRank": LowRankProjector(d_in, d_out, r=256).half().to(device),
    "MLP": MLPProjector(d_in, d_out).half().to(device),
    "Gated": GatedProjector(d_in, d_out).half().to(device),
    "LoRA": LoRAProjector(d_in, d_out, r=8, alpha=16).half().to(device)
}

# Generate dummy inputs
num_samples = 200
dummy_inputs = [torch.randn(1, 3, 224, 224).half().to(device) for _ in range(num_samples)]

# Compute features once per input for fair comparison
feats_list = []
with torch.no_grad():
    for x in dummy_inputs:
        feats = vision_tower(x).pooler_output
        feats_list.append(feats)
torch.cuda.synchronize()  # warm-up

# Profile projectors
times = {}

for name, proj in projectors.items():
    time_total = 0.0
    for feats in feats_list:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            with record_function(f"{name}_forward"):
                out = proj(feats)
            torch.cuda.synchronize()

        # Parse table output
        table_str = prof.key_averages().table(sort_by="cuda_time_total", row_limit=100)
        time_ms = 0.0
        for line in table_str.splitlines():
            if "ms" in line and "%" in line:
                parts = line.split()
                for p in parts:
                    if "ms" in p:
                        try:
                            val = float(p.replace("ms",""))
                            time_ms += val
                        except:
                            pass
        time_total += time_ms

    times[name] = time_total / len(feats_list)

# Print results and plot comparison
print("Average Forward Pass Time per Projector (ms):")
for k in projectors.keys():
    print(f"{k:10s} | time: {times[k]:7.3f} ms ")

labels = list(projectors.keys())
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10,5))
plt.bar(x - width/2, [times[l] for l in labels], width, color='skyblue')
plt.xticks(x, labels)
plt.ylabel("Average Forward Pass Time (ms)")
plt.title("Projector Forward Pass Time Comparison")
plt.legend()
plt.grid(axis='y')
plt.show()
