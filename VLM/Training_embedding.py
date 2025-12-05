import os, zipfile, time
from urllib.request import urlretrieve
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from transformers import (
    CLIPVisionModel, CLIPImageProcessor,
    AutoTokenizer, AutoModelForCausalLM
)

# Device and seeds
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
torch.manual_seed(0)

# Download + extract Tiny-ImageNet (if not present)
data_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
zip_path = "/content/tiny-imagenet-200.zip"
extract_path = "/content/tiny-imagenet-200"

if not os.path.exists(extract_path):
    print("Downloading Tiny-ImageNet (~250MB)...")
    urlretrieve(data_url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("/content")
print("Tiny-ImageNet ready at:", extract_path)

# Dataset transforms and loader (small subset for fast cycles)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

data_dir = extract_path
train_dir = os.path.join(data_dir, "train")
val_images_dir = os.path.join(data_dir, "val", "images")
val_ann = os.path.join(data_dir, "val", "val_annotations.txt")

# fix val structure -> val_fixed/<class>/*.JPEG
fixed_val_dir = os.path.join(data_dir, "val_fixed")
if not os.path.exists(fixed_val_dir):
    print("Fixing validation directory structure...")
    os.makedirs(fixed_val_dir, exist_ok=True)
    with open(val_ann, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                fname, cls = parts[0], parts[1]
                dst_dir = os.path.join(fixed_val_dir, cls)
                os.makedirs(dst_dir, exist_ok=True)
                src = os.path.join(val_images_dir, fname)
                dst = os.path.join(dst_dir, fname)
                if os.path.exists(src):
                    # copy
                    os.system(f"cp '{src}' '{dst}'")

# create ImageFolder datasets
train_dataset_full = ImageFolder(train_dir, transform=transform)
val_dataset_full   = ImageFolder(fixed_val_dir, transform=transform)

# Use small subsets for quick iteration; change ranges for larger training
train_subset_size = 100   # change to e.g., 5000 for fuller training
val_subset_size   = 10
train_dataset = Subset(train_dataset_full, range(min(train_subset_size, len(train_dataset_full))))
val_dataset   = Subset(val_dataset_full,   range(min(val_subset_size, len(val_dataset_full))))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

print("Train samples:", len(train_dataset), "Val samples:", len(val_dataset))
print("Classes:", len(train_dataset_full.classes))

# Load CLIP vision encoder (ViT-L/14) to extract visual features
clip_name = "openai/clip-vit-large-patch14"
print("Loading CLIP:", clip_name)
vision = CLIPVisionModel.from_pretrained(clip_name).to(device).eval()
clip_processor = CLIPImageProcessor.from_pretrained(clip_name)

# helper to extract pooled CLIP features (1 x clip_dim)
@torch.no_grad()
def extract_clip_features_from_loader(dataloader):
    feats = []
    labels = []
    for imgs, lbls in tqdm(dataloader, desc="Extracting CLIP features"):
        imgs = imgs.to(device)
        out = vision(imgs)
        # pooler_output shape: [B, clip_dim]
        feats.append(out.pooler_output.cpu())
        labels.append(lbls)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels

print("Extracting CLIP features (train)...")
train_X, train_labels = extract_clip_features_from_loader(train_loader)
print("Extracting CLIP features (val)...")
val_X, val_labels = extract_clip_features_from_loader(val_loader)

d_in = train_X.shape[1]
print("CLIP feature dim d_in =", d_in)  # should be 1024

# Load text LM (Qwen2-1.5B) and build caption-based target embeddings
text_model_name = "Qwen/Qwen2-1.5B-Instruct"
print("Loading text model (this may take a bit):", text_model_name)
tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=False)
text_model = AutoModelForCausalLM.from_pretrained(text_model_name).to(device).eval()

# obtain LM embedding dim
lm_dim = text_model.get_input_embeddings().weight.shape[1]
print("LM embedding dim (d_out) =", lm_dim)

# helper: caption -> pooled LM embedding (CPU tensor)
@torch.no_grad()
def caption_to_embedding(caption):
    toks = tokenizer(caption, return_tensors="pt", truncation=True, padding="longest").to(device)
    emb = text_model.get_input_embeddings()(toks.input_ids)  # [1, L, dim]
    mask = (toks.input_ids != tokenizer.pad_token_id).float().unsqueeze(-1)  # [1, L, 1]
    summed = (emb * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    pooled = (summed / denom).squeeze(0).cpu()
    return pooled  # cpu tensor [lm_dim]

# Build caption targets for train and val based on class names
class_names = train_dataset_full.classes
print("Number of classes (Tiny-ImageNet):", len(class_names))

# Build targets (fast by caching per class)
class_target_cache = {}
def get_target_for_label(label):
    if label not in class_target_cache:
        clsname = class_names[label]
        caption = f"a photo of a {clsname}"
        class_target_cache[label] = caption_to_embedding(caption)
    return class_target_cache[label]

train_Y_list = []
for i in range(len(train_X)):
    lbl = int(train_labels[i].item())
    train_Y_list.append(get_target_for_label(lbl))
val_Y_list = []
for i in range(len(val_X)):
    lbl = int(val_labels[i].item())
    val_Y_list.append(get_target_for_label(lbl))

train_Y = torch.stack(train_Y_list, dim=0)  # [N_train, lm_dim] CPU
val_Y   = torch.stack(val_Y_list,   dim=0)  # [N_val, lm_dim] CPU

train_X_dev = train_X.to(device)
val_X_dev   = val_X.to(device)
train_Y_dev = train_Y.to(device)
val_Y_dev   = val_Y.to(device)

d_out = lm_dim
print("Projectors will map d_in -> d_out:", d_in, "->", d_out)

# Define projectors (with LayerNorm at output)
class LinearProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)
    def forward(self, x):
        return self.ln(self.linear(x))

class LowRankProjector(nn.Module):
    def __init__(self, d_in, d_out, r=256):
        super().__init__()
        self.W1 = nn.Linear(d_in, r, bias=False)
        self.W2 = nn.Linear(r, d_out, bias=False)
        self.ln = nn.LayerNorm(d_out)
    def forward(self, x):
        return self.ln(self.W2(self.W1(x)))

class MLPProjector(nn.Module):
    def __init__(self, d_in, d_out, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out)
        )
        self.ln = nn.LayerNorm(d_out)
    def forward(self, x):
        return self.ln(self.net(x))

class GatedProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.Wp = nn.Linear(d_in, d_out)
        self.Wg = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)
    def forward(self, x):
        gate = torch.sigmoid(self.Wg(x))
        return self.ln(gate * self.Wp(x))

class LoRAProjector(nn.Module):
    def __init__(self, d_in, d_out, r=64, alpha=16):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.A = nn.Linear(d_in, r, bias=False)
        self.B = nn.Linear(r, d_out, bias=False)
        self.scaling = alpha / r
        self.ln = nn.LayerNorm(d_out)
    def forward(self, x):
        return self.ln(self.linear(x) + self.B(self.A(x)) * self.scaling)

# Training routine (MSE + cosine) with early stop + simple logging
def train_projector(model, name, X_train, Y_train, X_val, Y_val,
                    epochs=300, lr=1e-3, patience=20, batch_size=None):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    mse = nn.MSELoss()
    cos = nn.CosineEmbeddingLoss()
    N = X_train.size(0)
    if batch_size is None:
        batch_size = N  # full-batch default for small subsets

    best_val = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_mse": [], "val_cos_sim":[]}

    for epoch in range(1, epochs+1):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx]
            yb = Y_train[idx]

            optimizer.zero_grad()
            pred = model(xb)             # [B, d_out]
            loss_mse = mse(pred, yb)
            # cosine loss requires target labels: +1 for similar
            y_target = torch.ones(pred.size(0)).to(device)
            loss_cos = cos(pred, yb, y_target)
            loss = loss_mse + 0.5 * loss_cos
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        # epoch stats
        epoch_loss /= N
        # validation
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            val_mse = mse(pred_val, Y_val).item()
            # cosine similarity (mean)
            pred_norm = pred_val / (pred_val.norm(dim=1, keepdim=True) + 1e-8)
            y_norm = Y_val / (Y_val.norm(dim=1, keepdim=True) + 1e-8)
            cos_sim = (pred_norm * y_norm).sum(dim=1).mean().item()

        history["train_loss"].append(epoch_loss)
        history["val_mse"].append(val_mse)
        history["val_cos_sim"].append(cos_sim)

        if val_mse < best_val - 1e-5:
            best_val = val_mse
            best_epoch = epoch
            # save best
            torch.save(model.state_dict(), f"{name}_projector.pt")
        elif epoch - best_epoch >= patience:
            print(f"[{name}] Early stop at epoch {epoch} (best {best_epoch}, val_mse={best_val:.6f})")
            break

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{name}] epoch {epoch:03d} train_loss={epoch_loss:.6f} val_mse={val_mse:.6f} val_cos_sim={cos_sim:.4f}")

    # final load best
    model.load_state_dict(torch.load(f"{name}_projector.pt", map_location=device))
    return model, history

# Run training for all projectors (names)
projector_specs = {
    "Linear": lambda: LinearProjector(d_in, d_out),
    "LowRank": lambda: LowRankProjector(d_in, d_out, r=256),
    "MLP": lambda: MLPProjector(d_in, d_out, hidden=2048),
    "Gated": lambda: GatedProjector(d_in, d_out),
    "LoRA": lambda: LoRAProjector(d_in, d_out, r=64, alpha=16)
}

trained_models = {}
histories = {}

for name, constructor in projector_specs.items():
    print("\n" + "="*60)
    print(f"Training {name} projector")
    model = constructor()
    model, hist = train_projector(model, name,
                                  X_train=train_X_dev, Y_train=train_Y_dev,
                                  X_val=val_X_dev, Y_val=val_Y_dev,
                                  epochs=300, lr=1e-3, patience=20, batch_size=32)
    trained_models[name] = model
    histories[name] = hist

# Validation prints: compute avg val mse & cosine again and show top-k vocab tokens for a sample
print("\n=== Post-training validation checks ===")
mse = nn.MSELoss()
vocab_emb = text_model.get_input_embeddings().weight.data  # [V, d_out]
vocab_norm = vocab_emb / (vocab_emb.norm(dim=1, keepdim=True) + 1e-8)

# pick a sample index from validation
sample_idx = 0
sample_img_idx = sample_idx
sample_clip_feat = val_X_dev[sample_img_idx:sample_img_idx+1]  # [1, d_in]
sample_label = int(val_labels[sample_img_idx].item())
print("Sample true class:", class_names[sample_label])

for name, model in trained_models.items():
    model.eval()
    with torch.no_grad():
        pred = model(sample_clip_feat)  # [1, d_out]
        val_mse = float(mse(pred, val_Y_dev[sample_img_idx:sample_img_idx+1]).item())
        # cos sim
        pred_n = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
        true_n = val_Y_dev[sample_img_idx:sample_img_idx+1] / (val_Y_dev[sample_img_idx:sample_img_idx+1].norm(dim=1, keepdim=True) + 1e-8)
        cos_sim = float((pred_n * true_n).sum(dim=1).item())
        # nearest vocab tokens
        sims = (pred_n @ vocab_norm.t()).squeeze(0)  # [V]
        topk = torch.topk(sims, k=12)
        tokens = [tokenizer.decode([int(i)]) for i in topk.indices.tolist()]
        print(f"\n[{name}] val_mse={val_mse:.6f} cos_sim={cos_sim:.4f}")
        print("Top tokens:", tokens)

print("\nAll projectors trained and saved as <Name>_projector.pt")
