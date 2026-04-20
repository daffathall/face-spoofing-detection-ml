# pseudo_label.py — Pseudo Labeling dari high-confidence test predictions
# Jalankan SETELAH train_stacking.py selesai

import os, sys, shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config as cfg
from src.dataset import TestDataset, build_test_paths, get_val_transforms, build_samples
from src.model   import build_model


CONFIDENCE_THRESHOLD = 0.90   # hanya pakai prediksi dengan confidence >= 90%
PSEUDO_TRAIN_DIR     = os.path.join(cfg.DATA_DIR, "train_pseudo")


def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["model_name"], cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["img_size"]


@torch.no_grad()
def get_probs_all_models(test_paths, device):
    """Average softmax dari semua fold semua model."""
    all_ckpts = sorted(glob.glob(os.path.join(cfg.CHECKPOINT_DIR, "*_fold*_best.pth")))

    model_groups = {}
    for p in all_ckpts:
        ckpt  = torch.load(p, map_location="cpu")
        mname = ckpt["model_name"]
        if mname not in model_groups:
            model_groups[mname] = []
        model_groups[mname].append(p)

    all_model_probs = []
    for model_name, ckpt_paths in model_groups.items():
        print(f"  {model_name} ({len(ckpt_paths)} folds)")
        fold_probs = []
        for ckpt_path in ckpt_paths:
            model, img_size = load_model(ckpt_path, device)
            transform = get_val_transforms(img_size)
            ds     = TestDataset(test_paths, transform)
            loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE * 2,
                                shuffle=False, num_workers=cfg.NUM_WORKERS,
                                pin_memory=cfg.PIN_MEMORY)
            probs_list = []
            for imgs in tqdm(loader, desc=f"    Inference", leave=False):
                imgs = imgs.to(device)
                with autocast(enabled=cfg.MIXED_PRECISION):
                    logits = model(imgs)
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            fold_probs.append(np.concatenate(probs_list))
            del model
            torch.cuda.empty_cache()
        all_model_probs.append(np.mean(fold_probs, axis=0))

    # Average semua model
    return np.mean(all_model_probs, axis=0)   # (N_test, num_classes)


def copy_to_pseudo_dir(test_paths, pseudo_labels, confidences):
    """Copy gambar test confidence tinggi ke folder train_pseudo/"""

    # Buat struktur folder seperti train/
    for cls in cfg.CLASSES:
        os.makedirs(os.path.join(PSEUDO_TRAIN_DIR, cls), exist_ok=True)

    # Copy gambar asli train juga
    print("\nCopy train asli ke pseudo dir...")
    for cls in cfg.CLASSES:
        src = os.path.join(cfg.TRAIN_DIR, cls)
        dst = os.path.join(PSEUDO_TRAIN_DIR, cls)
        if os.path.isdir(src):
            for f in os.listdir(src):
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    # Copy test images dengan confidence tinggi
# Copy test images dengan confidence tinggi
    added = 0
    skipped = 0
    for img_path, label_idx, conf in zip(test_paths, pseudo_labels, confidences):
        # --- LOGIKA AMBANG BATAS AGRESIF DISINI ---
        label_name = cfg.IDX2CLASS[label_idx]
        if label_name == "fake_printed":
            current_threshold = 0.70  # Lebih rendah agar lebih banyak tertangkap
        else:
            current_threshold = CONFIDENCE_THRESHOLD # Default 0.90
        # ------------------------------------------

        if conf >= current_threshold:
            cls_name = cfg.IDX2CLASS[label_idx]
            fname    = os.path.basename(img_path)
            dst_path = os.path.join(PSEUDO_TRAIN_DIR, cls_name, f"pseudo_{fname}")
            shutil.copy2(img_path, dst_path)
            added += 1
        else:
            skipped += 1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}\n")

    test_data  = build_test_paths(cfg.TEST_DIR)
    test_paths = [p for p, _ in test_data]
    sample_ids = [s for _, s in test_data]
    print(f"Test images: {len(test_paths)}")

    # Kumpulkan probabilitas dari semua model
    print("\nCollecting ensemble probabilities...")
    probs = get_probs_all_models(test_paths, device)  # (404, 6)

    # Pseudo labels
    pred_indices  = probs.argmax(axis=1)
    confidences   = probs.max(axis=1)
    pred_labels   = [cfg.IDX2CLASS[i] for i in pred_indices]

    # Statistik
# Statistik dengan threshold berbeda per kelas
    high_conf_mask = []
    for i in range(len(confidences)):
        label_name = cfg.IDX2CLASS[pred_indices[i]]
        threshold = 0.70 if label_name == "fake_printed" else CONFIDENCE_THRESHOLD
        high_conf_mask.append(confidences[i] >= threshold)
    
    high_conf_mask = np.array(high_conf_mask)
    
    print(f"\n=== Pseudo Label Statistics ===")
    # ... sisanya biarkan tetap sama ...

    # Copy ke pseudo train dir
    print(f"\nMenyalin gambar ke {PSEUDO_TRAIN_DIR}...")
    added, skipped = copy_to_pseudo_dir(
        test_paths, pred_indices, confidences)
    print(f"✓ Pseudo samples ditambahkan : {added}")
    print(f"  Dibuang (low confidence)   : {skipped}")

    # Hitung total data baru
    total_orig = len(build_samples(cfg.TRAIN_DIR, cfg.CLASS2IDX))
    total_new  = len(build_samples(PSEUDO_TRAIN_DIR, cfg.CLASS2IDX))
    print(f"\nData asli  : {total_orig}")
    print(f"Data pseudo: {total_new} (+{total_new - total_orig} pseudo samples)")

    # Instruksi selanjutnya
    print(f"""
{'='*60}
SELANJUTNYA:
1. Edit configs/config.py — ubah TRAIN_DIR:
   TRAIN_DIR = "{PSEUDO_TRAIN_DIR}"

2. Jalankan training ulang:
   python train.py

3. Setelah selesai, jalankan stacking:
   python train_stacking.py

4. Submit outputs/submission_stacking.csv
{'='*60}
    """)


if __name__ == "__main__":
    main()
