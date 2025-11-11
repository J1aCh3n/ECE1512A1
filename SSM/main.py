import argparse, time, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from timm import create_model

def get_dataloaders(data_root, use_imnet100, img_size=224, bs=128, workers=6, n_train=2048, n_val=512):
    tf_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    if use_imnet100:
        train_set = datasets.ImageFolder(os.path.join(data_root, "train"), tf_train)
        val_set   = datasets.ImageFolder(os.path.join(data_root, "val"), tf_val)
        # (Optional) subsample to ensure we finish in ~minutes:
        if n_train: train_set = Subset(train_set, range(min(n_train, len(train_set))))
        if n_val:   val_set   = Subset(val_set,   range(min(n_val,   len(val_set))))
        num_classes = 100
    else:
        # FakeData: super fast sanity check, 100 classes by default
        train_set = datasets.FakeData(size=n_train, image_size=(3, img_size, img_size), num_classes=100, transform=tf_train)
        val_set   = datasets.FakeData(size=n_val,   image_size=(3, img_size, img_size), num_classes=100, transform=tf_val)
        num_classes = 100

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, num_classes

@torch.no_grad()
def evaluate(model, loader, amp=True):
    model.eval()
    correct = total = 0
    t0 = time.time()
    for x, y in loader:
        x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    dt = time.time() - t0
    return correct / max(1,total), dt

def train_one_epoch(model, loader, opt, scaler, amp=True):
    model.train()
    running = 0.0
    t0 = time.time()
    nimg = 0
    for x, y in loader:
        x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item() * y.size(0)
        nimg += y.size(0)
    dt = time.time() - t0
    return running / nimg, nimg / dt  # avg loss, imgs/sec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="", help="ImageNet-100 root (with train/ and val/). If empty, uses FakeData.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=5e-2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", type=str, default="vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-imnet", action="store_true", help="Force FakeData even if data-path is set")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_imnet100 = bool(args.data_path) and not args.no_imnet
    train_loader, val_loader, num_classes = get_dataloaders(
        args.data_path, use_imnet100, args.img_size, args.batch_size, args.workers
    )

    # Build model
    model = create_model(args.model, pretrained=False, num_classes=num_classes).cuda()
    # Simple AdamW
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Model: {args.model}")
    print(f"Data: {'ImageNet-100' if use_imnet100 else 'FakeData'} | "
          f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | "
          f"BS: {args.batch_size} | ImgSize: {args.img_size}")

    # Quick eval before training
    acc0, t_eval0 = evaluate(model, val_loader)
    print(f"[Eval@0] acc={acc0*100:.2f}% ({t_eval0:.2f}s)")

    for ep in range(args.epochs):
        loss, ips = train_one_epoch(model, train_loader, opt, scaler)
        acc, t_eval = evaluate(model, val_loader)
        print(f"[Ep {ep+1:02d}/{args.epochs}] loss={loss:.4f} | val@1={acc*100:.2f}% | "
              f"train_throughput={ips:.1f} img/s | eval_time={t_eval:.2f}s")

    # latency + memory probe
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        _ = model(x)
    torch.cuda.synchronize()
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"[Probe] bs={args.batch_size}, latency={dt*1000:.1f} ms, peak_mem={mem:.0f} MB")

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required for this quick test."
    main()
