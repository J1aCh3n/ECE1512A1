import os, shutil, random, glob, collections

src = "/home/jiachen/data/tiny-imagenet-200"
dst = "/home/jiachen/data/tiny-imagenet-200-sub"
random.seed(42)

def _list_images(dirpath):
    exts = ("*.JPEG", "*.jpeg", "*.jpg", "*.png", "*.bmp")
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(dirpath, e)))
    return sorted(out)

def _ensure_clean_dir(d):
    os.makedirs(d, exist_ok=True)

def subset_train(k_per_class=100):
    train_root = os.path.join(src, "train")
    dst_root = os.path.join(dst, "train")
    _ensure_clean_dir(dst_root)
    total = 0
    for wnid in sorted(os.listdir(train_root)):
        csrc = os.path.join(train_root, wnid)
        if not os.path.isdir(csrc):
            continue
        # images may live under train/<wnid>/images
        img_dir = os.path.join(csrc, "images")
        if not os.path.isdir(img_dir):
            img_dir = csrc
        imgs = _list_images(img_dir)
        if not imgs:
            continue
        random.shuffle(imgs)
        pick = imgs[:min(k_per_class, len(imgs))]
        cdst = os.path.join(dst_root, wnid)
        _ensure_clean_dir(cdst)
        for p in pick:
            shutil.copy2(p, os.path.join(cdst, os.path.basename(p)))
        total += len(pick)
    print(f"[train] wrote {total} images into {dst_root}")

def subset_val(k_per_class=25):
    val_root = os.path.join(src, "val")
    dst_root = os.path.join(dst, "val")
    _ensure_clean_dir(dst_root)

    ann = os.path.join(val_root, "val_annotations.txt")
    val_images_dir = os.path.join(val_root, "images")
    class_to_imgs = collections.defaultdict(list)

    if os.path.isfile(ann) and os.path.isdir(val_images_dir):
        # Unreorganized Tiny-IN val: use annotations to group by class
        with open(ann, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    fname, wnid = parts[0], parts[1]
                    img_path = os.path.join(val_images_dir, fname)
                    if os.path.isfile(img_path):
                        class_to_imgs[wnid].append(img_path)
    else:
        # Reorganized val/<wnid>/*.JPEG
        for wnid in sorted(os.listdir(val_root)):
            csrc = os.path.join(val_root, wnid)
            if not os.path.isdir(csrc):
                continue
            # images may live directly inside class dir
            imgs = _list_images(csrc)
            if not imgs:
                # or in an inner images/ dir
                img_dir = os.path.join(csrc, "images")
                if os.path.isdir(img_dir):
                    imgs = _list_images(img_dir)
            if imgs:
                class_to_imgs[wnid].extend(imgs)

    total = 0
    for wnid, imgs in class_to_imgs.items():
        random.shuffle(imgs)
        pick = imgs[:min(k_per_class, len(imgs))]
        cdst = os.path.join(dst_root, wnid)
        _ensure_clean_dir(cdst)
        for p in pick:
            shutil.copy2(p, os.path.join(cdst, os.path.basename(p)))
        total += len(pick)
    print(f"[val] wrote {total} images into {dst_root}")

if __name__ == "__main__":
    subset_train(k_per_class=50)  # ~20k train
    subset_val(k_per_class=10)     # ~5k val
    print("subset ready:", dst)
