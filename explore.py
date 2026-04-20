# explore.py — Quick EDA: class distribution + sample visualization

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config as cfg
from src.dataset import build_samples

random.seed(42)


def main():
    samples = build_samples(cfg.TRAIN_DIR, cfg.CLASS2IDX)
    print(f"Total images: {len(samples)}\n")

    # ── Class counts ──────────────────────────────────────────────────────────
    counts = {c: 0 for c in cfg.CLASSES}
    for _, idx in samples:
        counts[cfg.IDX2CLASS[idx]] += 1

    print("Class distribution:")
    for cls, n in counts.items():
        bar = "█" * (n // max(1, max(counts.values()) // 40))
        print(f"  {cls:<20s} {n:>6d}  {bar}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    bars = ax.bar(counts.keys(), counts.values(),
                  color=plt.cm.Set2(np.linspace(0, 1, len(counts))))
    ax.set_title("Class Distribution", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_xticklabels(counts.keys(), rotation=30, ha="right")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                str(int(b.get_height())), ha="center", fontsize=9)

    # ── Sample images grid ────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Sample images per class", fontsize=14)

    fig2, grid_axes = plt.subplots(len(cfg.CLASSES), 4, figsize=(12, 3 * len(cfg.CLASSES)))
    for row, cls in enumerate(cfg.CLASSES):
        cls_samples = [p for p, l in samples if l == cfg.CLASS2IDX[cls]]
        chosen      = random.sample(cls_samples, min(4, len(cls_samples)))
        for col, path in enumerate(chosen):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            grid_axes[row][col].imshow(img)
            grid_axes[row][col].axis("off")
            if col == 0:
                grid_axes[row][col].set_title(cls, fontsize=10, loc="left")
        # blank remaining cols
        for col in range(len(chosen), 4):
            grid_axes[row][col].axis("off")

    fig.tight_layout()
    fig2.tight_layout()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "class_distribution.png"), dpi=150)
    fig2.savefig(os.path.join(cfg.OUTPUT_DIR, "sample_images.png"), dpi=150)
    print(f"\nPlots saved to {cfg.OUTPUT_DIR}/")
    plt.show()


if __name__ == "__main__":
    main()
