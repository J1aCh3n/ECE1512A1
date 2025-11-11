#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_runs.py — VMamba log summarizer with plots

Parses:
  * log_rank0.txt : Acc@1/5 (last), Max accuracy, step times, peak MB,
                    #params, GFLOPs, train loss samples, val losses
  * config.json   : IMG_SIZE, SSM_FORWARDTYPE, DEPTHS, EMBED_DIM, NUM_CLASSES

Outputs:
  * JSON blob (per run)
  * Markdown table (compact or wide with --wide)
  * Optional CSV (--csv path)
  * Optional figures (--outdir path):
      - throughput_vs_mem.png
      - step_time_hist.png
      - val_loss_curve.png
"""
import argparse, glob, json, os, re, sys, statistics as stats, csv
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------- regexes ----------
ACC_RE      = re.compile(r"\*\s*Acc@1\s+([0-9.]+)\s+Acc@5\s+([0-9.]+)")
MAX_RE      = re.compile(r"Max accuracy(?:\s*EMA)?:\s*([0-9.]+)")
STEP_RE     = re.compile(r"\btime\s+([0-9.]+)\s+\(([0-9.]+)\)")
PEAK_RE     = re.compile(r"\bmem\s+([0-9]+)MB\b")
EPOCH_RE    = re.compile(r"Train:\s*\[(\d+)/(\d+)\]\[(\d+)/(\d+)\]")
PARAMS_RE   = re.compile(r"number of params:\s*([0-9]+)")
GFLOP_RE    = re.compile(r"number of GFLOPs:\s*([0-9.eE+-]+)")
# typical training line has "... loss 2.7314 (2.7601) ..."
TRAIN_LOSS_RE = re.compile(r"\bloss\s+([0-9.]+)\s+\(([0-9.]+)\)")
# validation loss comes from "Test:" lines
VAL_LOSS_RE   = re.compile(r"Test:\s*\[\d+/\d+\].*?\bLoss\s+([0-9.]+)")

# scrape effective cfg (fallback)
RE_FWD  = re.compile(r"^\s*SSM_FORWARDTYPE:\s*([A-Za-z0-9_]+)", re.M)
RE_IMG  = re.compile(r"^\s*IMG_SIZE:\s*([0-9]+)", re.M)
RE_PARM = re.compile(r"number of params:\s*([0-9]+)")
RE_GFL  = re.compile(r"number of GFLOPs:\s*([0-9.]+)")

# ---------- helpers ----------
def find_log_files(run_dir: str) -> List[str]:
    pats = [
        os.path.join(run_dir, "**", "log_rank0.txt"),
        os.path.join(run_dir, "**", "*.log"),
        os.path.join(run_dir, "**", "log*.txt"),
        os.path.join(run_dir, "*.txt"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    return sorted(set(files), key=lambda p: os.path.getmtime(p))

def find_config_json(run_dir: str) -> Optional[str]:
    cands = sorted(
        glob.glob(os.path.join(run_dir, "**", "config.json"), recursive=True),
        key=os.path.getmtime
    )
    return cands[-1] if cands else None

def load_cfg_fields(run_dir: str, log_path: str) -> dict:
    out = {"forward_type": None, "img_size": None, "gflops": None, "params": None, "config_path": None}
    cfg_path = find_config_json(run_dir)
    if cfg_path:
        out["config_path"] = cfg_path
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            out["img_size"] = cfg.get("DATA", {}).get("IMG_SIZE") or cfg.get("MODEL", {}).get("IMG_SIZE")
            out["forward_type"] = cfg.get("MODEL", {}).get("VSSM", {}).get("SSM_FORWARDTYPE")
        except Exception:
            pass
    # fallback: scrape log
    try:
        with open(log_path, "r", errors="ignore") as f:
            txt = f.read()
        if out["forward_type"] is None:
            m = RE_FWD.search(txt)
            if m: out["forward_type"] = m.group(1)
        if out["img_size"] is None:
            m = RE_IMG.search(txt)
            if m: out["img_size"] = int(m.group(1))
        m = RE_PARM.search(txt)
        if m: out["params"] = int(m.group(1))
        m = RE_GFL.search(txt)
        if m: out["gflops"] = float(m.group(1))
    except Exception:
        pass
    return out

def parse_log(log_path: str, get_last_n_steps: int = 20,
              get_last_n_train_losses: int = 200,
              get_last_n_val_losses: int = 200) -> Dict[str, Any]:
    acc1 = acc5 = max_acc1 = None
    step_times: List[float] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    peak_mem = None
    epochs_seen = set()
    steps_per_epoch = None

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = ACC_RE.search(line)
            if m:
                acc1, acc5 = float(m.group(1)), float(m.group(2))
            m = MAX_RE.search(line)
            if m:
                try: max_acc1 = float(m.group(1))
                except: pass
            m = STEP_RE.search(line)
            if m:
                try: step_times.append(float(m.group(1)))
                except: pass
            m = PEAK_RE.search(line)
            if m:
                try:
                    val = int(m.group(1))
                    peak_mem = val if (peak_mem is None or val > peak_mem) else peak_mem
                except: pass
            m = EPOCH_RE.search(line)
            if m:
                try:
                    ep_idx, ep_total, st_idx, st_total = map(int, m.groups())
                    epochs_seen.add(ep_idx)
                    steps_per_epoch = st_total
                except: pass
            m = TRAIN_LOSS_RE.search(line)
            if m:
                # take instantaneous loss (group 1)
                try: train_losses.append(float(m.group(1)))
                except: pass
            m = VAL_LOSS_RE.search(line)
            if m:
                try: val_losses.append(float(m.group(1)))
                except: pass

    med_time = None
    if step_times:
        window = step_times[-get_last_n_steps:] if len(step_times) >= get_last_n_steps else step_times
        med_time = stats.median(window)

    if get_last_n_train_losses and len(train_losses) > get_last_n_train_losses:
        train_losses = train_losses[-get_last_n_train_losses:]
    if get_last_n_val_losses and len(val_losses) > get_last_n_val_losses:
        val_losses = val_losses[-get_last_n_val_losses:]

    out = {
        "acc1": acc1,
        "acc5": acc5,
        "max_acc1": max_acc1,
        "step_times": step_times[-get_last_n_steps:] if step_times else [],
        "step_time_median": med_time,
        "peak_mem_mb": peak_mem,
        "epochs_seen": sorted(list(epochs_seen)),
        "steps_per_epoch": steps_per_epoch,
        "train_losses": train_losses,
        "train_loss_median": (stats.median(train_losses) if train_losses else None),
        "val_losses": val_losses,
        "log_path": log_path,
    }
    return out

def summarize_run(run_dir: str, get_last_n_steps: int = 20,
                  get_last_n_train_losses: int = 200,
                  get_last_n_val_losses: int = 200) -> Dict[str, Any]:
    logs = find_log_files(run_dir)
    if not logs:
        return {"error": f"No logs found in {run_dir}", "run_dir": run_dir}
    log = logs[-1]
    summary = parse_log(
        log,
        get_last_n_steps=get_last_n_steps,
        get_last_n_train_losses=get_last_n_train_losses,
        get_last_n_val_losses=get_last_n_val_losses
    )
    cfgbits = load_cfg_fields(run_dir, log)
    summary.update(cfgbits)
    summary["run_dir"] = run_dir
    return summary

def to_markdown_table(rows: List[Tuple[str, Dict[str, Any]]],
                      batch_size: Optional[int], wide: bool=False) -> str:
    lines = []
    if wide:
        header = "| Run | Fwd | Img | Acc@1 (last/best) | Acc@5 | Step (med s) | imgs/s | Peak MB | Params (M) | GFLOPs | Steps/Epoch | TrainLoss(med) |"
        sep    = "|---|:--:|--:|:--:|--:|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = "| Run | Acc@1 | Acc@5 | Step (median s) | imgs/s | Peak MB | Steps/Epoch |"
        sep    = "|---|---:|---:|---:|---:|---:|---:|"
    lines += [header, sep]

    for name, m in rows:
        acc1 = "-" if m.get("acc1") is None else f"{m['acc1']:.2f}"
        acc5 = "-" if m.get("acc5") is None else f"{m['acc5']:.2f}"
        maxa = "-" if m.get("max_acc1") is None else f"{m['max_acc1']:.2f}"
        medt = "-" if m.get("step_time_median") is None else f"{m['step_time_median']:.3f}"
        imgs = "-"
        if batch_size and m.get("step_time_median"):
            imgs = f"{(batch_size / m['step_time_median']):.1f}"
        peak = "-" if m.get("peak_mem_mb") is None else str(m["peak_mem_mb"])
        spe  = "-" if m.get("steps_per_epoch") is None else str(m["steps_per_epoch"])
        fwd  = "-" if m.get("forward_type") is None else str(m["forward_type"])
        img  = "-" if m.get("img_size") is None else str(m["img_size"])
        pm   = "-" if m.get("params") is None else f"{m['params']/1e6:.2f}"
        gf   = "-" if m.get("gflops") is None else f"{m['gflops']:.2f}"
        lmd  = "-" if m.get("train_loss_median") is None else f"{m['train_loss_median']:.3f}"

        if wide:
            lines.append(f"| {name} | {fwd} | {img} | {acc1}/{maxa} | {acc5} | {medt} | {imgs} | {peak} | {pm} | {gf} | {spe} | {lmd} |")
        else:
            lines.append(f"| {name} | {acc1} | {acc5} | {medt} | {imgs} | {peak} | {spe} |")
    return "\n".join(lines)

def make_figures(outdir: str, rows: List[Tuple[str, Dict[str, Any]]], batch_size: Optional[int]) -> None:
    os.makedirs(outdir, exist_ok=True)

    labels, imgs_per_s, peak_mb, step_time_lists, val_loss_lists = [], [], [], [], []
    for name, r in rows:
        labels.append(name)
        st_med = r.get("step_time_median")
        imgs_per_s.append((batch_size / st_med) if (batch_size and st_med) else np.nan)
        peak_mb.append(r.get("peak_mem_mb") if r.get("peak_mem_mb") is not None else np.nan)
        step_time_lists.append(r.get("step_times", []))
        val_loss_lists.append(r.get("val_losses", []))

    # Figure 1
    plt.figure()
    for i, lbl in enumerate(labels):
        x, y = imgs_per_s[i], peak_mb[i]
        if np.isfinite(x) and np.isfinite(y):
            plt.scatter([x], [y])
            plt.text(x, y, lbl)
    plt.xlabel("Throughput (imgs/s, median step)")
    plt.ylabel("Peak GPU Memory (MB)")
    plt.title("Throughput vs Peak Memory")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "throughput_vs_mem.png"), dpi=180)

    # Figure 2
    plt.figure()
    plotted_any = False
    for i, lbl in enumerate(labels):
        st = np.array(step_time_lists[i], dtype=float)
        if st.size:
            plt.hist(st, bins=40, alpha=0.5, label=lbl)
            plotted_any = True
    plt.xlabel("Step time (s)")
    plt.ylabel("Count")
    plt.title("Step-time distribution")
    if plotted_any:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "step_time_hist.png"), dpi=180)

    # Figure 3
    plt.figure()
    drawn = False
    for i, lbl in enumerate(labels):
        vl = np.array(val_loss_lists[i], dtype=float)
        if vl.size:
            xs = np.arange(1, len(vl) + 1)
            plt.plot(xs, vl, marker="o", label=f"{lbl} val loss")
            drawn = True
    plt.xlabel("Eval # (≈ epoch)")
    plt.ylabel("Val loss")
    plt.title("Validation loss vs epoch")
    if drawn:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "val_loss_curve.png"), dpi=180)

def main():
    ap = argparse.ArgumentParser(description="Summarize VMamba logs into a compact table (with optional plots).")
    ap.add_argument("--run", action="append", default=[], help="name=/path/to/run_dir (repeatable)")
    ap.add_argument("--batch-size", type=int, default=None, help="Per-GPU batch size (for imgs/s)")
    ap.add_argument("--last-n-steps", type=int, default=20, help="Last N step times for median")
    ap.add_argument("--last-n-train-losses", type=int, default=200, help="Use last N train losses for median")
    ap.add_argument("--last-n-val-losses", type=int, default=200, help="Use last N validation loss points when plotting")
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    ap.add_argument("--wide", action="store_true", help="Show extra columns (fwd/img/params/GFLOPs/best acc/loss)")
    ap.add_argument("--outdir", type=str, default=None, help="If set, saves PNG figures in this directory")
    args = ap.parse_args()

    if not args.run:
        print("No --run provided. Example: --run base=/path/a --run mem=/path/b", file=sys.stderr)
        sys.exit(2)

    rows: List[Tuple[str, Dict[str, Any]]] = []
    out_json: Dict[str, Any] = {}
    for spec in args.run:
        if "=" not in spec:
            print(f"Bad --run spec (expected name=path): {spec}", file=sys.stderr)
            sys.exit(2)
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        summ = summarize_run(
            path,
            get_last_n_steps=args.last_n_steps,
            get_last_n_train_losses=args.last_n_train_losses,
            get_last_n_val_losses=args.last_n_val_losses
        )
        rows.append((name, summ))
        out_json[name] = summ

    print(json.dumps(out_json, indent=2))
    print("\n" + to_markdown_table(rows, args.batch_size, wide=args.wide) + "\n")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "run","acc1_last","acc5_last","acc1_best","step_time_median_s","imgs_per_s",
                "peak_mem_mb","steps_per_epoch","img_size","forward_type","params","gflops",
                "train_loss_median","log_path","config_path","run_dir"
            ])
            for name, m in rows:
                medt = m.get("step_time_median")
                imgs = (args.batch_size / medt) if (args.batch_size and medt) else None
                w.writerow([
                    name,
                    m.get("acc1"), m.get("acc5"), m.get("max_acc1"),
                    medt, imgs,
                    m.get("peak_mem_mb"), m.get("steps_per_epoch"),
                    m.get("img_size"), m.get("forward_type"),
                    m.get("params"), m.get("gflops"),
                    m.get("train_loss_median"),
                    m.get("log_path"), m.get("config_path"), m.get("run_dir"),
                ])
        print(f"CSV written: {args.csv}")

    if args.outdir:
        make_figures(args.outdir, rows, batch_size=args.batch_size)
        print(f"Figures written under: {args.outdir}")

if __name__ == "__main__":
    main()
