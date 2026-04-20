# train_stacking.py — OOF Stacking dengan Meta-Learner

import os, sys, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config as cfg
from src.dataset import FaceDataset, build_samples, get_val_transforms, TestDataset, build_test_paths
from src.model   import build_model


# ── Load model dari checkpoint ────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["model_name"], cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["img_size"], ckpt.get("best_f1", 0.0)


# ── Get probabilities dari satu model ─────────────────────────────────────────

@torch.no_grad()
def get_probs(model, samples_or_paths, img_size, device, is_test=False):
    transform = get_val_transforms(img_size)
    if is_test:
        ds = TestDataset(samples_or_paths, transform)
    else:
        ds = FaceDataset(samples_or_paths, transform)

    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE * 2,
                        shuffle=False, num_workers=cfg.NUM_WORKERS,
                        pin_memory=cfg.PIN_MEMORY)
    all_probs = []
    for batch in tqdm(loader, desc="  Inference", leave=False):
        if is_test:
            imgs = batch.to(device)
        else:
            imgs = batch[0].to(device)
        with autocast(enabled=cfg.MIXED_PRECISION):
            logits = model(imgs)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)  # (N, num_classes)


# ── Step 1: Kumpulkan OOF predictions ────────────────────────────────────────

def collect_oof_predictions(all_samples, device):
    """
    Untuk setiap model × fold, load checkpoint dan predict val fold.
    Hasilnya: matrix (N_train, n_models × n_classes)
    """
    print("\n" + "="*60)
    print("  STEP 1: Collect OOF Predictions")
    print("="*60)

    n = len(all_samples)
    labels = np.array([s[1] for s in all_samples])

    # Temukan semua checkpoint yang tersedia
    all_ckpts = sorted(glob.glob(os.path.join(cfg.CHECKPOINT_DIR, "*_fold*_best.pth")))
    if not all_ckpts:
        raise RuntimeError("Tidak ada checkpoint fold! Jalankan train.py dulu.")

    # Kelompokkan per model_name
    model_groups = {}
    for ckpt_path in all_ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        mname = ckpt["model_name"]
        if mname not in model_groups:
            model_groups[mname] = []
        model_groups[mname].append(ckpt_path)

    print(f"\nModel ditemukan: {list(model_groups.keys())}")
    print(f"Total train samples: {n}")

    # OOF matrix: (N, n_models × n_classes)
    n_models    = len(model_groups)
    oof_matrix  = np.zeros((n, n_models * cfg.NUM_CLASSES))

    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)

    for m_idx, (model_name, ckpt_paths) in enumerate(model_groups.items()):
        print(f"\n  Model {m_idx+1}/{n_models}: {model_name}")
        col_start = m_idx * cfg.NUM_CLASSES
        col_end   = col_start + cfg.NUM_CLASSES

        for fold, (_, val_idx) in enumerate(skf.split(labels, labels)):
            # Cari checkpoint fold ini
            fold_ckpts = [p for p in ckpt_paths if f"_fold{fold+1}_" in p]
            if not fold_ckpts:
                print(f"    [SKIP] Fold {fold+1} checkpoint tidak ditemukan")
                continue

            ckpt_path = fold_ckpts[0]
            model, img_size, best_f1 = load_model(ckpt_path, device)
            print(f"    Fold {fold+1}: {os.path.basename(ckpt_path)} (F1={best_f1:.4f})")

            val_samples = [all_samples[i] for i in val_idx]
            probs = get_probs(model, val_samples, img_size, device, is_test=False)
            oof_matrix[val_idx, col_start:col_end] = probs

            del model
            torch.cuda.empty_cache()

    return oof_matrix, labels, list(model_groups.keys())


# ── Step 2: Kumpulkan Test predictions ───────────────────────────────────────

def collect_test_predictions(test_paths, device):
    """
    Untuk test set: average semua fold per model, lalu stack.
    Hasilnya: matrix (N_test, n_models × n_classes)
    """
    print("\n" + "="*60)
    print("  STEP 2: Collect Test Predictions")
    print("="*60)

    all_ckpts = sorted(glob.glob(os.path.join(cfg.CHECKPOINT_DIR, "*_fold*_best.pth")))

    model_groups = {}
    for ckpt_path in all_ckpts:
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        mname = ckpt["model_name"]
        if mname not in model_groups:
            model_groups[mname] = []
        model_groups[mname].append(ckpt_path)

    n_test      = len(test_paths)
    n_models    = len(model_groups)
    test_matrix = np.zeros((n_test, n_models * cfg.NUM_CLASSES))

    for m_idx, (model_name, ckpt_paths) in enumerate(model_groups.items()):
        print(f"\n  Model {m_idx+1}/{n_models}: {model_name}")
        col_start   = m_idx * cfg.NUM_CLASSES
        col_end     = col_start + cfg.NUM_CLASSES
        fold_probs  = []

        for ckpt_path in ckpt_paths:
            model, img_size, _ = load_model(ckpt_path, device)
            probs = get_probs(model, test_paths, img_size, device, is_test=True)
            fold_probs.append(probs)
            del model
            torch.cuda.empty_cache()

        # Average semua fold untuk model ini
        test_matrix[:, col_start:col_end] = np.mean(fold_probs, axis=0)

    return test_matrix


# ── Step 3: Train Meta-Learner ────────────────────────────────────────────────

def train_meta_learner(oof_matrix, labels, model_names):
    print("\n" + "="*60)
    print("  STEP 3: Train Meta-Learner")
    print("="*60)

    print(f"\nOOF matrix shape: {oof_matrix.shape}")
    print(f"Features per model: {cfg.NUM_CLASSES} probs × {len(model_names)} models = {oof_matrix.shape[1]} features")

    # Baseline: weighted averaging
    n_models = len(model_names)
    avg_probs = np.zeros((len(labels), cfg.NUM_CLASSES))
    for i in range(n_models):
        avg_probs += oof_matrix[:, i*cfg.NUM_CLASSES:(i+1)*cfg.NUM_CLASSES]
    avg_probs /= n_models
    baseline_f1 = f1_score(labels, avg_probs.argmax(axis=1),
                            average="macro", zero_division=0)
    print(f"\nBaseline (averaging) OOF F1: {baseline_f1:.4f}")

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(oof_matrix)

    # Meta-learner: Logistic Regression (simple = anti-overfit)
    meta = LogisticRegression(
        C=0.15,                    # strong regularization
        max_iter=1000,
        solver="lbfgs",
        random_state=cfg.SEED,
    )

    # Cross-val untuk estimasi score
    cv_scores = cross_val_score(
        meta, X, labels,
        cv=StratifiedKFold(5, shuffle=True, random_state=cfg.SEED),
        scoring="f1_macro",
    )
    print(f"Meta-LR OOF F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if cv_scores.mean() > baseline_f1:
        print(f"✓ Stacking lebih baik dari averaging! (+{cv_scores.mean()-baseline_f1:.4f})")
    else:
        print(f"⚠ Averaging masih lebih baik ({baseline_f1:.4f} vs {cv_scores.mean():.4f})")
        print(f"  Akan tetap pakai stacking jika selisihnya < 0.01")

    # Fit pada semua data
    meta.fit(X, labels)

    # Simpan
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    joblib.dump(meta,   os.path.join(cfg.OUTPUT_DIR, "meta_learner.pkl"))
    joblib.dump(scaler, os.path.join(cfg.OUTPUT_DIR, "meta_scaler.pkl"))
    print(f"\n✓ Meta-learner saved → outputs/meta_learner.pkl")

    return meta, scaler, baseline_f1, cv_scores.mean()


# ── Step 4: Generate submission ───────────────────────────────────────────────

def generate_submission(meta, scaler, test_matrix,
                        sample_ids, baseline_f1, meta_f1):
    print("\n" + "="*60)
    print("  STEP 4: Generate Submission")
    print("="*60)

    X_test = scaler.transform(test_matrix)

    # Gunakan stacking jika lebih baik, fallback ke averaging
    if meta_f1 >= baseline_f1 - 0.005:
        print(f"Menggunakan: STACKING (meta F1={meta_f1:.4f})")
        pred_indices = meta.predict(X_test)
    else:
        print(f"Menggunakan: AVERAGING (baseline F1={baseline_f1:.4f})")
        avg_probs    = test_matrix.reshape(len(test_matrix), -1, cfg.NUM_CLASSES).mean(axis=1)
        pred_indices = avg_probs.argmax(axis=1)

    pred_labels = [cfg.IDX2CLASS[i] for i in pred_indices]
    sub = pd.DataFrame({"id": sample_ids, "label": pred_labels})

    if os.path.exists(cfg.SAMPLE_SUB):
        sample = pd.read_csv(cfg.SAMPLE_SUB)
        sub    = sample[["id"]].merge(sub, on="id", how="left")

    out_path = os.path.join(cfg.OUTPUT_DIR, "submission_stacking.csv")
    sub.to_csv(out_path, index=False)
    print(f"\n✓ Submission → {out_path}")
    print(f"\nDistribusi prediksi:\n{sub['label'].value_counts().to_string()}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_samples = build_samples(cfg.TRAIN_DIR, cfg.CLASS2IDX)
    test_data   = build_test_paths(cfg.TEST_DIR)
    test_paths  = [p for p, _ in test_data]
    sample_ids  = [s for _, s in test_data]

    # Step 1: OOF predictions
    oof_matrix, labels, model_names = collect_oof_predictions(all_samples, device)

    # Step 2: Test predictions
    test_matrix = collect_test_predictions(test_paths, device)

    # Step 3: Train meta-learner
    meta, scaler, baseline_f1, meta_f1 = train_meta_learner(
        oof_matrix, labels, model_names)

    # Step 4: Generate submission
    generate_submission(meta, scaler, test_matrix,
                        sample_ids, baseline_f1, meta_f1)


if __name__ == "__main__":
    main()
