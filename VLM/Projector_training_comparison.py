import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from transformers import CLIPVisionModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Image transform
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

# Tiny-ImageNet paths
data_dir = "/content/tiny-imagenet-200"
train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_dataset   = ImageFolder(os.path.join(data_dir, "val_fixed"), transform=transform)

# Subset for speed
train_dataset = Subset(train_dataset, range(5000))
val_dataset   = Subset(val_dataset, range(500))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)

# CLIP Feature Extraction
model_name = "openai/clip-vit-large-patch14"
vision_tower = CLIPVisionModel.from_pretrained(model_name).to(device).eval()

@torch.no_grad()
def extract_features(dataloader):
    feats, labels = [], []
    for imgs, lbls in tqdm(dataloader, desc="Extracting CLIP features"):
        imgs = imgs.to(device)
        outputs = vision_tower(imgs)
        feats.append(outputs.pooler_output.cpu())
        labels.append(lbls)
    return torch.cat(feats), torch.cat(labels)

train_X, _ = extract_features(train_loader)
val_X, _   = extract_features(val_loader)

# Projection dimensions
d_in  = train_X.shape[1]  # 1024
d_out = 1536              # Match Qwen2 embedding

train_Y = torch.zeros((len(train_X), d_out))
val_Y   = torch.zeros((len(val_X), d_out))

# Projection Architectures
class LinearProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.linear(x)

class LowRankProjector(nn.Module):
    def __init__(self, d_in, d_out, r=256):
        super().__init__()
        self.W1 = nn.Linear(d_in, r, bias=False)
        self.W2 = nn.Linear(r, d_out, bias=False)
    def forward(self, x):
        return self.W2(self.W1(x))

class MLPProjector(nn.Module):
    def __init__(self, d_in, d_out, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out)
        )
    def forward(self, x):
        return self.net(x)

class GatedProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.Wp = nn.Linear(d_in, d_out)
        self.Wg = nn.Linear(d_in, d_out)
    def forward(self, x):
        gate = torch.sigmoid(self.Wg(x))
        return gate * self.Wp(x)

class LoRAProjector(nn.Module):
    def __init__(self, d_in, d_out, r=64, alpha=16):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.A = nn.Linear(d_in, r, bias=False)
        self.B = nn.Linear(r, d_out, bias=False)
        self.scaling = alpha / r
    def forward(self, x):
        return self.linear(x) + self.B(self.A(x)) * self.scaling

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# Training Loop
def estimate_flops(name, d_in, d_out, r=256, hidden=2048):
    if name == "Linear":
        return d_in*d_out
    elif name == "LowRank":
        return d_in*r + r*d_out
    elif name == "MLP":
        return d_in*hidden + hidden*d_out
    elif name == "Gated":
        return 2*d_in*d_out
    elif name == "LoRA":
        return d_in*d_out + d_in*r + r*d_out
    return 0

def train_model(model, name, epochs=500, lr=1e-3, patience=10):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_epoch = 0
    losses = []

    X_train = train_X.to(device)
    Y_train = train_Y.to(device)
    X_val   = val_X.to(device)
    Y_val   = val_Y.to(device)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            print(f"[{name}] Early stopping at epoch {epoch}")
            break

    runtime = time.time() - start_time
    params = count_params(model)
    flops = estimate_flops(name, d_in, d_out)
    return losses, best_loss, runtime, params, flops


# Train All Projectors
models = [
    ("Linear", LinearProjector(d_in, d_out)),
    ("LowRank", LowRankProjector(d_in, d_out)),
    ("MLP", MLPProjector(d_in, d_out)),
    ("Gated", GatedProjector(d_in, d_out))#,
    #("LoRA", LoRAProjector(d_in, d_out))
]

results = {}
loss_histories = {}

for name, model in models:
    print(f"\n=== Training {name} Projector ===")
    losses, best_loss, runtime, params, flops = train_model(model, name)
    results[name] = (best_loss, runtime, params, flops)
    loss_histories[name] = losses

# Summary
print("\n Summary of Model Performance")
print("Model      | Test Loss | Params (M) | FLOPs (M) | Runtime (s)")
print("------------------------------------------------------------")
for name, (loss, runtime, params, flops) in results.items():
    print(f"{name:10s} | {loss:10.4f} | {params/1e6:10.2f} | {flops/1e6:10.2f} | {runtime:10.2f}")

# Plot
plt.figure(figsize=(8,5))
for name, losses in loss_histories.items():
    plt.plot(losses, label=name)
plt.xlabel("Epoch")
plt.ylabel("Train MSE Loss")
plt.title("Training Convergence: Tiny-ImageNet CLIP Projection")
plt.legend()
plt.grid(True)
plt.show()
