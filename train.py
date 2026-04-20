# train.py — Anti-overfit: lighter models + stronger regularization + 5-fold

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config as cfg
from src.dataset import (FaceDataset, build_samples, get_weighted_sampler,
                          get_train_transforms, get_val_transforms)
from src.model  import build_model
from src.losses import FocalLoss, compute_class_weights


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── MixUp / CutMix ────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cut_w, cut_h = int(W*np.sqrt(1-lam)), int(H*np.sqrt(1-lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1,x2 = np.clip(cx-cut_w//2,0,W), np.clip(cx+cut_w//2,0,W)
    y1,y2 = np.clip(cy-cut_h//2,0,H), np.clip(cy+cut_h//2,0,H)
    mixed = x.clone()
    mixed[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
    lam = 1 - (x2-x1)*(y2-y1)/(W*H)
    return mixed, y, y[idx], lam

def mix_criterion(criterion, pred, ya, yb, lam):
    return lam*criterion(pred,ya) + (1-lam)*criterion(pred,yb)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def get_scheduler(optimizer, num_epochs, warmup_epochs, steps_per_epoch):
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.05, end_factor=1.0,
                      total_iters=warmup_epochs*steps_per_epoch)
    cosine = CosineAnnealingLR(optimizer,
                                T_max=(num_epochs-warmup_epochs)*steps_per_epoch,
                                eta_min=cfg.MIN_LR)
    return SequentialLR(optimizer, [warmup, cosine],
                        milestones=[warmup_epochs*steps_per_epoch])


# ── Optimizer dengan layer-wise LR decay ──────────────────────────────────────

def build_optimizer(model, base_lr, weight_decay, lr_decay):
    head_p = list(model.head.parameters()) + list(model.pool.parameters())
    back_p = list(model.backbone.parameters())
    n, chunk = len(back_p), max(1, len(back_p)//4)
    groups = [{"params": head_p, "lr": base_lr}]
    for i, start in enumerate(range(0, n, chunk)):
        groups.append({"params": back_p[start:start+chunk],
                       "lr": base_lr * (lr_decay**(4-i))})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer,
                    scheduler, scaler, device, epoch):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    use_mix = epoch > cfg.WARMUP_EPOCHS

    for step, (imgs, labels) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)

        if use_mix and random.random() < cfg.MIXUP_PROB:
            fn = mixup_data if random.random() < 0.5 else cutmix_data
            imgs, ya, yb, lam = fn(imgs, labels)
            with autocast(enabled=cfg.MIXED_PRECISION):
                loss = mix_criterion(criterion, model(imgs), ya, yb, lam) / cfg.GRAD_ACCUM
        else:
            with autocast(enabled=cfg.MIXED_PRECISION):
                logits = model(imgs)
                loss   = criterion(logits, labels) / cfg.GRAD_ACCUM

        scaler.scale(loss).backward()
        if (step+1) % cfg.GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(); scheduler.step()

        total_loss += loss.item() * cfg.GRAD_ACCUM
        with torch.no_grad():
            all_preds.extend(model(imgs).argmax(1).cpu().numpy()
                             if use_mix else logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss/len(loader), f1_score(all_labels, all_preds, average="macro", zero_division=0)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(enabled=cfg.MIXED_PRECISION):
            logits = model(imgs)
            total_loss += criterion(logits, labels).item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    # Tampilkan per-class F1 untuk monitor kelas lemah
    report = classification_report(all_labels, all_preds,
                                    target_names=cfg.CLASSES, zero_division=0)
    return total_loss/len(loader), macro_f1, report


# ── Train one fold ────────────────────────────────────────────────────────────

def train_fold(model_name, img_size, ckpt_name, train_s, val_s, device):
    train_ds = FaceDataset(train_s, get_train_transforms(img_size))
    val_ds   = FaceDataset(val_s,   get_val_transforms(img_size))
    sampler  = get_weighted_sampler(train_s, cfg.NUM_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                               num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
                               drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE*2, shuffle=False,
                               num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)

    model     = build_model(model_name, cfg.NUM_CLASSES, drop_rate=cfg.DROP_RATE).to(device)
    weights   = compute_class_weights(train_s, cfg.NUM_CLASSES)
    criterion = FocalLoss(gamma=cfg.FOCAL_GAMMA, alpha=weights, label_smoothing=0.1)
    optimizer = build_optimizer(model, cfg.LR, cfg.WEIGHT_DECAY, cfg.LR_DECAY)
    scheduler = get_scheduler(optimizer, cfg.NUM_EPOCHS, cfg.WARMUP_EPOCHS, len(train_loader))
    scaler    = GradScaler(enabled=cfg.MIXED_PRECISION)

    best_f1, no_improve = 0.0, 0
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{ckpt_name}_best.pth")

    for epoch in range(1, cfg.NUM_EPOCHS+1):
        print(f"\n  Epoch {epoch}/{cfg.NUM_EPOCHS}")
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scheduler, scaler, device, epoch)
        vl_loss, vl_f1, report = validate(model, val_loader, criterion, device)

        print(f"    Train → loss:{tr_loss:.4f}  F1:{tr_f1:.4f}")
        print(f"    Val   → loss:{vl_loss:.4f}  F1:{vl_f1:.4f}")

        # Tampilkan per-class detail setiap 5 epoch
        if epoch % 5 == 0:
            print(report)

        if vl_f1 > best_f1:
            best_f1, no_improve = vl_f1, 0
            torch.save({"epoch": epoch, "model_name": model_name,
                        "img_size": img_size, "state_dict": model.state_dict(),
                        "best_f1": best_f1}, ckpt_path)
            print(f"    ✓ Saved  (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg.EARLY_STOP:
                print("    Early stopping.")
                break
    return best_f1


# ── K-Fold per model ──────────────────────────────────────────────────────────

def train_model_kfold(model_name, img_size, ckpt_name, all_samples, device):
    print(f"\n{'='*60}")
    print(f"  {model_name}  ({img_size}px)  —  {cfg.N_FOLDS}-Fold CV")
    print(f"{'='*60}")

    labels_arr  = np.array([s[1] for s in all_samples])
    skf         = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    fold_scores = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(labels_arr, labels_arr)):
        print(f"\n── Fold {fold+1}/{cfg.N_FOLDS} ──────────────────────")
        tr_s = [all_samples[i] for i in tr_idx]
        vl_s = [all_samples[i] for i in vl_idx]
        f1   = train_fold(model_name, img_size, f"{ckpt_name}_fold{fold+1}",
                          tr_s, vl_s, device)
        fold_scores.append(f1)
        print(f"  Fold {fold+1} best F1: {f1:.4f}")

    mean_f1 = float(np.mean(fold_scores))
    print(f"\n  {model_name} → Mean F1: {mean_f1:.4f} ± {np.std(fold_scores):.4f}")
    return mean_f1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("WARNING: GPU tidak terdeteksi! Training akan sangat lambat.")
        print("Pastikan PyTorch CUDA sudah terinstall dengan benar.")

    all_samples = build_samples(cfg.TRAIN_DIR, cfg.CLASS2IDX)
    print(f"\nTotal train images: {len(all_samples)}")
    for cls, idx in cfg.CLASS2IDX.items():
        n = sum(1 for _, l in all_samples if l == idx)
        print(f"  {cls:<20s}: {n}")

    results = {}
    for model_name, img_size, ckpt_name in cfg.MODELS:
        results[ckpt_name] = train_model_kfold(
            model_name, img_size, ckpt_name, all_samples, device)

    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    for name, f1 in results.items():
        print(f"  {name:<25s}  Mean Val F1: {f1:.4f}")
    print("\nJalankan: python inference.py")


if __name__ == "__main__":
    main()
