"""
Train UNet for navigation graph detection from floor plan images.

Saves:
    runs/<timestamp>/best_model.pth     - best model checkpoint
    runs/<timestamp>/training_log.csv   - per-epoch metrics
    runs/<timestamp>/training_curves.png
    runs/<timestamp>/test_predictions.png
    runs/<timestamp>/overlay_predictions.png
    runs/<timestamp>/config.json        - hyperparameters used
"""

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FloorPlanDataset(Dataset):
    """Loads floor-plan / nav-graph mask pairs. Optional geometric augmentation."""

    def __init__(self, img_dir, mask_dir, augment=False):
        self.images = sorted(Path(img_dir).glob('*.png'))
        self.masks = sorted(Path(mask_dir).glob('*.png'))
        self.augment = augment
        assert len(self.images) == len(self.masks), 'image / mask count mismatch'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert('L'), dtype=np.float32) / 255.0
        mask = np.array(Image.open(self.masks[idx]).convert('L'), dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            k = random.randint(0, 3)
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


# ---------------------------------------------------------------------------
# UNet Model
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_ch
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))

        self.head = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)


# ---------------------------------------------------------------------------
# Loss & Metrics
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """Combined BCE + Dice loss -- handles sparse mask class imbalance."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        dice = 1.0 - (2.0 * inter + 1.0) / (probs.sum() + targets.sum() + 1.0)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


def compute_metrics(logits, targets, threshold=0.5):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        intersection = tp
        union = preds.sum() + targets.sum() - intersection
        iou = intersection / (union + 1e-8)
        dice = (2.0 * intersection) / (preds.sum() + targets.sum() + 1e-8)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
    }


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    metrics_sum = defaultdict(float)
    n_samples = 0

    pbar = tqdm(loader, desc=f'  Train {epoch:2d}/{num_epochs}', leave=True)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        n_samples += bs
        batch_m = compute_metrics(logits, masks)
        for k, v in batch_m.items():
            metrics_sum[k] += v * bs

        pbar.set_postfix(loss=f'{running_loss / n_samples:.4f}',
                         dice=f'{metrics_sum["dice"] / n_samples:.4f}')

    avg_loss = running_loss / n_samples
    avg_metrics = {k: v / n_samples for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=None, num_epochs=None, label='Val'):
    model.eval()
    running_loss = 0.0
    metrics_sum = defaultdict(float)
    n_samples = 0

    desc = f'  {label:5s} {epoch:2d}/{num_epochs}' if epoch is not None else f'  {label}'
    pbar = tqdm(loader, desc=desc, leave=True)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        n_samples += bs
        for k, v in compute_metrics(logits, masks).items():
            metrics_sum[k] += v * bs

        pbar.set_postfix(loss=f'{running_loss / n_samples:.4f}')

    avg_loss = running_loss / n_samples
    avg_metrics = {k: v / n_samples for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_training_curves(history, out_dir):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train')
    ax.plot(epochs, history['val_loss'], label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Loss (BCE + Dice)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history['train_dice'], label='Train')
    ax.plot(epochs, history['val_dice'], label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Dice')
    ax.set_title('Dice Coefficient')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, history['train_iou'], label='Train')
    ax.plot(epochs, history['val_iou'], label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('IoU')
    ax.set_title('Intersection over Union')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history['val_precision'], label='Precision')
    ax.plot(epochs, history['val_recall'], label='Recall')
    ax.plot(epochs, history['val_f1'], label='F1', ls='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
    ax.set_title('Precision / Recall / F1 (Val)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history['lr'])
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    gap = [t - v for t, v in zip(history['train_dice'], history['val_dice'])]
    ax.plot(epochs, gap, color='red')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Train Dice - Val Dice')
    ax.set_title('Overfitting Gap')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_test_predictions(model, dataset, device, out_dir, n_vis=6):
    model.eval()
    indices = random.sample(range(len(dataset)), min(n_vis, len(dataset)))
    results = []
    with torch.no_grad():
        for idx in indices:
            img, mask = dataset[idx]
            logits = model(img.unsqueeze(0).to(device))
            pred = torch.sigmoid(logits).cpu().squeeze().numpy()
            results.append((img.squeeze().numpy(), mask.squeeze().numpy(), pred))

    # Predictions grid
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4 * len(results)))
    if len(results) == 1:
        axes = axes[np.newaxis, :]
    for i, (img, gt, pred) in enumerate(results):
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title('Floor Plan'); axes[i, 0].axis('off')
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('Ground Truth'); axes[i, 1].axis('off')
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction (prob)'); axes[i, 2].axis('off')
        axes[i, 3].imshow((pred > 0.5).astype(float), cmap='gray')
        axes[i, 3].set_title('Prediction (thresh=0.5)'); axes[i, 3].axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / 'test_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Overlay
    n_cols = max(len(results) // 2, 1)
    n_rows = (len(results) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes_flat = np.array(axes).flatten()
    for i, (img, gt, pred) in enumerate(results):
        rgb = np.stack([img, img, img], axis=-1)
        gt_mask = gt > 0.5
        rgb[gt_mask, 0] = np.clip(rgb[gt_mask, 0] - 0.2, 0, 1)
        rgb[gt_mask, 1] = 1.0
        rgb[gt_mask, 2] = np.clip(rgb[gt_mask, 2] - 0.2, 0, 1)
        pred_mask = pred > 0.5
        rgb[pred_mask, 0] = 1.0
        rgb[pred_mask, 1] = np.clip(rgb[pred_mask, 1] - 0.2, 0, 1)
        rgb[pred_mask, 2] = np.clip(rgb[pred_mask, 2] - 0.2, 0, 1)
        axes_flat[i].imshow(rgb)
        axes_flat[i].set_title(f'Sample {indices[i]}  (green=GT, red=pred)')
        axes_flat[i].axis('off')
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / 'overlay_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

def write_csv_header(path):
    with open(path, 'w') as f:
        f.write('epoch,train_loss,val_loss,train_dice,val_dice,train_iou,val_iou,'
                'train_precision,val_precision,train_recall,val_recall,'
                'train_f1,val_f1,lr,elapsed_s\n')


def append_csv_row(path, epoch, train_loss, val_loss, train_m, val_m, lr, elapsed):
    with open(path, 'a') as f:
        f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},'
                f'{train_m["dice"]:.6f},{val_m["dice"]:.6f},'
                f'{train_m["iou"]:.6f},{val_m["iou"]:.6f},'
                f'{train_m["precision"]:.6f},{val_m["precision"]:.6f},'
                f'{train_m["recall"]:.6f},{val_m["recall"]:.6f},'
                f'{train_m["f1"]:.6f},{val_m["f1"]:.6f},'
                f'{lr:.2e},{elapsed:.1f}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train UNet for nav-graph detection')
    parser.add_argument('-d', '--data-dir', type=str, default='data',
                        help='Root data directory with train/val/test splits')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory (default: runs/<timestamp>)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if device.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path('runs') / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output : {out_dir.resolve()}')

    # Save config
    config = vars(args)
    config['device'] = str(device)
    if device.type == 'cuda':
        config['gpu'] = torch.cuda.get_device_name(0)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Data
    data = Path(args.data_dir)
    train_ds = FloorPlanDataset(data / 'train' / 'images', data / 'train' / 'masks', augment=True)
    val_ds = FloorPlanDataset(data / 'val' / 'images', data / 'val' / 'masks', augment=False)
    test_ds = FloorPlanDataset(data / 'test' / 'images', data / 'test' / 'masks', augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

    # Model
    model = UNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'UNet parameters: {total_params:,}')

    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )

    # Training log
    csv_path = out_dir / 'training_log.csv'
    write_csv_header(csv_path)

    history = defaultdict(list)
    best_val_dice = 0.0
    patience_counter = 0
    total_batches = len(train_loader)

    print(f'\nTraining for up to {args.epochs} epochs  (early-stop patience={args.patience})')
    print(f'  {len(train_ds)} train samples, {total_batches} batches/epoch (bs={args.batch_size})')
    print('=' * 90)

    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
        )
        val_loss, val_m = validate(
            model, val_loader, criterion, device, epoch, args.epochs, label='Val',
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        # Log history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr_now)
        for k in train_m:
            history[f'train_{k}'].append(train_m[k])
            history[f'val_{k}'].append(val_m[k])

        # CSV
        append_csv_row(csv_path, epoch, train_loss, val_loss, train_m, val_m, lr_now, elapsed)

        # Checkpoint
        saved = ''
        if val_m['dice'] > best_val_dice:
            best_val_dice = val_m['dice']
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / 'best_model.pth')
            saved = '  ** saved best **'
        else:
            patience_counter += 1

        # Epoch summary
        print(f'  ---- Epoch {epoch:2d} summary ({elapsed:.1f}s, lr={lr_now:.1e}) ----')
        print(f'       {"":12s}  {"Train":>8s}  {"Val":>8s}')
        print(f'       {"Loss":12s}  {train_loss:8.4f}  {val_loss:8.4f}')
        print(f'       {"Dice":12s}  {train_m["dice"]:8.4f}  {val_m["dice"]:8.4f}')
        print(f'       {"IoU":12s}  {train_m["iou"]:8.4f}  {val_m["iou"]:8.4f}')
        print(f'       {"Precision":12s}  {train_m["precision"]:8.4f}  {val_m["precision"]:8.4f}')
        print(f'       {"Recall":12s}  {train_m["recall"]:8.4f}  {val_m["recall"]:8.4f}')
        print(f'       {"F1":12s}  {train_m["f1"]:8.4f}  {val_m["f1"]:8.4f}')
        if saved:
            print(saved)
        print(f'       patience: {patience_counter}/{args.patience}')
        print()

        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    total_time = time.time() - train_start
    print('=' * 90)
    print(f'Training complete in {total_time:.0f}s.  Best val Dice: {best_val_dice:.4f}')

    # ── Save training curves ──
    print('\nSaving training curves...')
    save_training_curves(history, out_dir)

    # ── Test evaluation ──
    print('Evaluating on test set...')
    model.load_state_dict(torch.load(out_dir / 'best_model.pth', map_location=device, weights_only=True))
    test_loss, test_m = validate(model, test_loader, criterion, device)

    print('=' * 50)
    print('        TEST SET RESULTS')
    print('=' * 50)
    for name, val in [('Loss', test_loss), *test_m.items()]:
        label = name.capitalize() if isinstance(name, str) else name
        print(f'  {label:12s}: {val:.4f}')
    print('=' * 50)

    # Save test metrics
    test_results = {'test_loss': test_loss, **{f'test_{k}': v for k, v in test_m.items()}}
    with open(out_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    # ── Final summary (train/val/test) ──
    print('\nComputing final train/val metrics for summary...')
    _, final_train_m = validate(model, train_loader, criterion, device)
    _, final_val_m = validate(model, val_loader, criterion, device)

    header = f'{"Metric":>12s} | {"Train":>8s} | {"Val":>8s} | {"Test":>8s}'
    print(header)
    print('-' * len(header))
    for k in ['dice', 'iou', 'precision', 'recall', 'f1']:
        print(f'{k:>12s} | {final_train_m[k]:8.4f} | {final_val_m[k]:8.4f} | {test_m[k]:8.4f}')

    summary = {
        'train': final_train_m,
        'val': final_val_m,
        'test': test_m,
        'best_val_dice': best_val_dice,
        'total_epochs': len(history['train_loss']),
        'total_time_s': round(total_time, 1),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Visual predictions ──
    print('\nSaving test predictions...')
    save_test_predictions(model, test_ds, device, out_dir, n_vis=6)

    print(f'\nAll outputs saved to: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
