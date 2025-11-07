import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import CLIPVisionModel, CLIPImageProcessor
from tqdm import tqdm

# Setup & data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = Subset(datasets.CIFAR10(root="./data", train=True, transform=transform, download=True), range(200))
test_dataset  = Subset(datasets.CIFAR10(root="./data", train=False, transform=transform, download=True), range(20))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False)

# Feature extraction via CLIP ViT-L/14
model_name = "openai/clip-vit-large-patch14"
vision_tower = CLIPVisionModel.from_pretrained(model_name).to(device).eval()
image_processor = CLIPImageProcessor.from_pretrained(model_name)

@torch.no_grad()
def extract_features(dataloader):
    feats, labels = [], []
    for imgs, lbls in tqdm(dataloader, desc="Extracting CLIP features"):
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = vision_tower(imgs)
        feats.append(outputs.pooler_output.float().cpu())
        labels.append(lbls)
        torch.cuda.empty_cache()
    return torch.cat(feats), torch.cat(labels)

train_X, train_Y = extract_features(train_loader)
test_X, test_Y = extract_features(test_loader)

train_Y = train_X.clone()
test_Y = test_X.clone()

d_in = train_X.shape[1]
d_out = d_in

# Projector architectures
class LinearProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.linear(x)

class LowRankProjector(nn.Module):
    def __init__(self, d_in, d_out, r):
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
    def __init__(self, d_in, d_out, r=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.A = nn.Linear(d_in, r, bias=False)
        self.B = nn.Linear(r, d_out, bias=False)
        self.scaling = alpha / r
    def forward(self, x):
        return self.linear(x) + self.B(self.A(x)) * self.scaling

# Helper functions
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model_name, d_in, d_out, r=256, hidden=2048):
    if model_name == "Linear":
        return d_in * d_out
    elif model_name == "LowRank":
        return d_in * r + r * d_out
    elif model_name == "MLP":
        return d_in * hidden + hidden * d_out
    elif model_name == "Gated":
        return 2 * d_in * d_out
    elif model_name == "LoRA":
        return d_in * d_out + d_in * r + r * d_out
    else:
        return 0

# Training function
def train_model(model, name, epochs=400, lr=1e-3, patience=10):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_epoch = 0
    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_X.to(device))
        loss = loss_fn(pred, train_Y.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(test_X.to(device)), test_Y.to(device)).item()

        # Early stopping
        if val_loss < best_loss - 1e-3:
            best_loss = val_loss
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            print(f"[{name}] Early stopping at epoch {epoch}")
            break

    runtime = time.time() - start_time
    params = count_params(model)
    flops = estimate_flops(name, d_in, d_out)

    return losses, best_loss, runtime, params, flops

# Run all models
models = [
    ("Linear", LinearProjector(d_in, d_out)),
    ("LowRank", LowRankProjector(d_in, d_out, r=256)),
    ("MLP", MLPProjector(d_in, d_out)),
    ("Gated", GatedProjector(d_in, d_out)),
    ("LoRA", LoRAProjector(d_in, d_out))
]

results = []
loss_histories = {}

for name, model in models:
    print(f"\n=== Training {name} Projector ===")
    losses, test_loss, runtime, params, flops = train_model(model, name)
    results.append((name, test_loss, runtime, params, flops))
    loss_histories[name] = losses

# Summary & visualization
print("\n Summary of Model Performance")
print("Model      | Test Loss | Params (M) | FLOPs (M) | Runtime (s)")
print("------------------------------------------------------------")
for name, loss, t, params, flops in results:
    print(f"{name:10s} | {loss:10.4f} | {params/1e6:10.2f} | {flops/1e6:10.2f} | {t:10.2f}")

plt.figure(figsize=(8,5))
for name, losses in loss_histories.items():
    plt.plot(losses, label=name)
plt.xlabel("Epoch")
plt.ylabel("Train MSE Loss")
plt.title("Training Convergence: CIFAR-10 CLIP Feature Projection")
plt.legend()
plt.grid(True)
plt.show()
