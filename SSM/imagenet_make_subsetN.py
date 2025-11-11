#!/usr/bin/env python3
import os, glob, shutil, random, argparse, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root with train/ and val/ organized as train/<wnid>/*.JPEG, val/<wnid>/*.JPEG")
    ap.add_argument("--dst", required=True, help="Output root")
    ap.add_argument("--n_classes", type=int, default=50)
    ap.add_argument("--k_train",   type=int, default=25)
    ap.add_argument("--k_val",     type=int, default=10)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--classlist", type=str, default=None, help="Optional path to a file containing wnids to use (one per line). If provided, overrides n_classes.")
    args = ap.parse_args()

    random.seed(args.seed)
    src_train = os.path.join(args.src, "train")
    src_val   = os.path.join(args.src, "val")

    # Collect classes present in BOTH splits
    train_cls = sorted([d for d in os.listdir(src_train) if os.path.isdir(os.path.join(src_train, d))])
    val_cls   = sorted([d for d in os.listdir(src_val)   if os.path.isdir(os.path.join(src_val, d))])
    common    = sorted(set(train_cls).intersection(val_cls))

    # Choose classes
    if args.classlist and os.path.isfile(args.classlist):
        with open(args.classlist) as f:
            chosen = [ln.strip() for ln in f if ln.strip()]
        missing = [c for c in chosen if c not in common]
        if missing:
            raise SystemExit(f"Classes missing from dataset: {missing[:5]} ...")
    else:
        # deterministic pick: sort then take first N (or shuffle first if you prefer)
        chosen = common[:args.n_classes]

    # Save class list for reproducibility
    os.makedirs(args.dst, exist_ok=True)
    cl_path = os.path.join(args.dst, f"classlist_{len(chosen)}.txt")
    with open(cl_path, "w") as f:
        f.write("\n".join(chosen))
    print(f"[info] Using {len(chosen)} classes. Saved to: {cl_path}")

    def _pick_k(src_root, split, k):
        total = 0
        for wnid in chosen:
            in_dir = os.path.join(src_root, wnid)
            imgs = sorted(glob.glob(os.path.join(in_dir, "*.JPEG")))
            if len(imgs) < k:
                print(f"[warn] {wnid} has only {len(imgs)} images in {split}, taking all.")
            pick = imgs[:min(k, len(imgs))]
            out_dir = os.path.join(args.dst, split, wnid)
            os.makedirs(out_dir, exist_ok=True)
            for p in pick:
                shutil.copy2(p, os.path.join(out_dir, os.path.basename(p)))
            total += len(pick)
        print(f"[{split}] wrote {total} images ({total//len(chosen)} per class avg)")

    _pick_k(src_train, "train", args.k_train)
    _pick_k(src_val,   "val",   args.k_val)
    print("Subset ready at:", args.dst)

if __name__ == "__main__":
    main()
