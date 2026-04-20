# inference.py — TTA + K-Fold ensemble → submission CSV

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config as cfg
from src.dataset import TestDataset, build_test_paths, get_tta_transforms
from src.model   import build_model


def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["model_name"], cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["img_size"], ckpt.get("best_f1", 0.0)


@torch.no_grad()
def predict_tta(model, img_paths, img_size, device):
    all_probs = []
    for transform in get_tta_transforms(img_size):
        ds     = TestDataset(img_paths, transform)
        loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE * 2,
                            shuffle=False, num_workers=cfg.NUM_WORKERS,
                            pin_memory=cfg.PIN_MEMORY)
        probs_list = []
        for imgs in loader:
            imgs = imgs.to(device)
            with autocast(enabled=cfg.MIXED_PRECISION):
                logits = model(imgs)
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs_list, axis=0))
    return np.mean(all_probs, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    test_data  = build_test_paths(cfg.TEST_DIR)
    img_paths  = [p for p, _ in test_data]
    sample_ids = [s for _, s in test_data]
    print(f"Test images: {len(img_paths)}\n")

    # Cari SEMUA checkpoint (semua model × semua fold)
    all_ckpts = sorted(glob.glob(os.path.join(cfg.CHECKPOINT_DIR, "*_best.pth")))
    if not all_ckpts:
        raise RuntimeError("Tidak ada checkpoint! Jalankan train.py dulu.")

    print(f"Ditemukan {len(all_ckpts)} checkpoint untuk di-ensemble:\n")

    ensemble_probs = None
    total_weight   = 0.0

    for ckpt_path in all_ckpts:
        model, img_size, best_f1 = load_model(ckpt_path, device)
        name = os.path.basename(ckpt_path)
        print(f"  [{best_f1:.4f}] {name}")

        # Bobot proporsional terhadap val F1 (model lebih baik = kontribusi lebih besar)
        w     = best_f1 if best_f1 > 0 else 1.0
        probs = predict_tta(model, img_paths, img_size, device)

        if ensemble_probs is None:
            ensemble_probs = probs * w
        else:
            ensemble_probs += probs * w
        total_weight += w

        del model
        torch.cuda.empty_cache()

    ensemble_probs /= total_weight
    pred_labels = [cfg.IDX2CLASS[i] for i in ensemble_probs.argmax(axis=1)]

    sub = pd.DataFrame({"id": sample_ids, "label": pred_labels})
    if os.path.exists(cfg.SAMPLE_SUB):
        sample = pd.read_csv(cfg.SAMPLE_SUB)
        sub    = sample[["id"]].merge(sub, on="id", how="left")

    out_path = os.path.join(cfg.OUTPUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"\n✓ Submission → {out_path}  ({len(sub)} rows)")
    print(f"\nDistribusi prediksi:\n{sub['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()