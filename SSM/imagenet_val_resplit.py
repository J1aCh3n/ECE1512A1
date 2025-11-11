# /home/jiachen/1512/imagenet_val_resplit.py
import os, random, shutil, glob
random.seed(42)
src = "/home/jiachen/data/ILSVRC2012_img_val"         # reorganized val/<wnid>/*.JPEG
dst = "/home/jiachen/data/ILSVRC2012-valsplit"    # new dataset root
K_TRAIN, K_VAL = 40, 10                            # tweak as you like
os.makedirs(dst, exist_ok=True)
for split in ["train","val"]:
    os.makedirs(os.path.join(dst, split), exist_ok=True)

classes = sorted([d for d in os.listdir(src) if os.path.isdir(os.path.join(src,d))])
for wnid in classes:
    imgs = sorted(glob.glob(os.path.join(src, wnid, "*.JPEG")))
    if len(imgs) < K_TRAIN + K_VAL: continue
    random.shuffle(imgs)
    tr, va = imgs[:K_TRAIN], imgs[K_TRAIN:K_TRAIN+K_VAL]
    for p, split in [(tr,"train"), (va,"val")]:
        outd = os.path.join(dst, split, wnid)
        os.makedirs(outd, exist_ok=True)
        for f in p: shutil.copy2(f, os.path.join(outd, os.path.basename(f)))
print("val-only split ready at:", dst)
