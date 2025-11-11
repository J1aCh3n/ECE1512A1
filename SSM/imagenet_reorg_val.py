#!/usr/bin/env python3
import os, shutil, glob, sys

ROOT = "/home/jiachen/data"                  # where you put the files
VAL_DIR = os.path.join(ROOT, "ILSVRC2012_img_val")  # contains ILSVRC2012_val_00000001.JPEG ...
DEVKIT_ROOT = os.path.join(ROOT, "ILSVRC2012_devkit_t12")  # this folder may contain a nested copy

def find_file(patterns):
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None

def load_idx2wnid_from_meta(meta_path):
    """
    Try to read meta.mat with multiple SciPy load styles and extract
    [(ILSVRC2012_ID, WNID)] -> list of wnids ordered 1..1000.
    Returns None if parsing fails.
    """
    try:
        from scipy.io import loadmat
    except Exception as e:
        print("Warn: scipy not available for meta.mat parsing:", e)
        return None

    def try_load(**kwargs):
        try:
            md = loadmat(meta_path, **kwargs)
            return md
        except Exception as e:
            return None

    md = (try_load(squeeze_me=True, struct_as_record=False)
          or try_load(squeeze_me=False, struct_as_record=True)
          or try_load())

    if not md or "synsets" not in md:
        print("Warn: meta.mat has no 'synsets' key")
        return None

    syn = md["synsets"]

    # Normalize syn into an iterable of rows with attributes/keys
    items = []
    try:
        # Case A: squeeze_me=True => array of objects with .WNID / .ILSVRC2012_ID
        for row in syn:
            # Some scipy versions give a scalar object, others arrays
            try:
                ilsvrc_id = int(getattr(row, "ILSVRC2012_ID"))
                wnid = str(getattr(row, "WNID"))
            except Exception:
                # Case B: dict-like / numpy.void with dtype.names
                try:
                    rn = row
                    def fetch(r, k):
                        v = r[k]
                        # v could be nested arrays of shape (1,1) or just a string
                        while hasattr(v, "__array_interface__") and hasattr(v, "shape") and v.shape == (1,1):
                            v = v[0,0]
                        if isinstance(v, (bytes, bytearray)):
                            v = v.decode("utf-8")
                        # char arrays to str
                        if hasattr(v, "dtype") and v.dtype.kind in ("U","S"):
                            try: v = str(v)
                            except: pass
                        return v
                    ilsvrc_id = int(fetch(rn, "ILSVRC2012_ID"))
                    wnid = str(fetch(rn, "WNID"))
                except Exception:
                    continue
            items.append((ilsvrc_id, wnid))
    except Exception as e:
        print("Warn: meta.mat parse error:", e)
        return None

    # Filter to 1..1000 and sort
    items = [(i, w) for (i, w) in items if isinstance(i, int) and 1 <= i <= 1000 and isinstance(w, str) and w.startswith("n")]
    if not items:
        print("Warn: meta.mat produced no (id, wnid) pairs in 1..1000")
        return None
    items = sorted(items, key=lambda x: x[0])
    idx2wnid = [w for (_, w) in items]
    if len(idx2wnid) < 1000:
        print(f"Warn: meta.mat yielded only {len(idx2wnid)} classes (<1000)")
        # still usable but not ideal; continue anyway
    return idx2wnid

def load_idx2wnid_from_synset_words(devkit_root):
    """
    Fall back to synset_words.txt (commonly at devkit root or devkit/data/..).
    """
    cand = find_file([
        os.path.join(devkit_root, "**", "synset_words.txt"),
        os.path.join(devkit_root, "synset_words.txt"),
    ])
    if not cand or not os.path.isfile(cand):
        print("Warn: synset_words.txt not found for fallback")
        return None
    wnids = []
    with open(cand, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            wnid = line.split(" ", 1)[0]
            if wnid.startswith("n"): wnids.append(wnid)
    if len(wnids) < 1000:
        print(f"Warn: synset_words.txt only has {len(wnids)} lines")
        return None
    return wnids[:1000]

def main():
    if not os.path.isdir(VAL_DIR):
        print(f"ERROR: val dir not found: {VAL_DIR}")
        sys.exit(2)

    # locate devkit/data and files
    gt_path = find_file([
        os.path.join(DEVKIT_ROOT, "**", "data", "ILSVRC2012_validation_ground_truth.txt"),
        os.path.join(DEVKIT_ROOT, "data", "ILSVRC2012_validation_ground_truth.txt"),
    ])
    if not gt_path:
        print("ERROR: could not find ILSVRC2012_validation_ground_truth.txt under devkit root")
        sys.exit(2)
    devkit_data_dir = os.path.dirname(gt_path)
    meta_path = find_file([
        os.path.join(devkit_data_dir, "meta.mat"),
        os.path.join(DEVKIT_ROOT, "**", "data", "meta.mat"),
    ])

    # Build idx->wnid mapping (prefer meta.mat)
    idx2wnid = None
    if meta_path and os.path.isfile(meta_path):
        idx2wnid = load_idx2wnid_from_meta(meta_path)

    if idx2wnid is None:
        # fallback to synset_words
        idx2wnid = load_idx2wnid_from_synset_words(DEVKIT_ROOT)

    if idx2wnid is None:
        raise RuntimeError("Could not build idx->wnid mapping from meta.mat or synset_words.txt")

    # Read 50k val ground truth (indices 1..1000)
    with open(gt_path, "r") as f:
        gt = [int(x.strip()) for x in f if x.strip()]
    if len(gt) != 50000:
        print(f"ERROR: expected 50000 val labels, got {len(gt)}")
        sys.exit(2)

    moved = 0
    for i in range(1, len(gt)+1):
        idx = gt[i-1]  # 1..1000
        wnid = idx2wnid[idx-1] if idx-1 < len(idx2wnid) else None
        if not wnid:
            continue
        fname = f"ILSVRC2012_val_{i:08d}.JPEG"
        src = os.path.join(VAL_DIR, fname)
        if not os.path.isfile(src):
            continue  # already moved or missing
        dst_dir = os.path.join(VAL_DIR, wnid)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, os.path.join(dst_dir, fname))
        moved += 1
        if moved % 5000 == 0:
            print(f"moved {moved}/50000")
    print(f"Done. Reorganized val/ into class subfolders. Moved {moved} files.")

if __name__ == "__main__":
    main()
