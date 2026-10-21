"""
Chest X-ray binary classifier (e.g. Normal vs Pneumonia) - starter pipeline.

Expected CSV (one row per image):
    image_path,label,patient_id
    data/img_001.png,0,P001
    data/img_002.png,1,P002

label: 0 = negative, 1 = positive. patient_id is REQUIRED so the split is done
per patient (prevents leakage between train/val/test).

Usage:
    pip install torch torchvision scikit-learn pandas pillow
    python cxr_classifier.py --csv labels.csv --epochs 15
    python cxr_classifier.py --csv labels.csv --scratch   # random init baseline
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    if train:
        # No horizontal flip: heart position / laterality carries meaning in CXR.
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.RandomResizedCrop(224, scale=(0.85, 1.0)),
            T.RandomRotation(8),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class CXRDataset(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.image_path).convert("L")
        return self.tf(img), torch.tensor(row.label, dtype=torch.float32)


def patient_split(df, val_frac=0.15, test_frac=0.15, seed=42):
    """Split by patient_id so no patient appears in more than one set."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, groups=df.patient_id))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
    train_idx, val_idx = next(gss.split(trainval, groups=trainval.patient_id))
    return trainval.iloc[train_idx], trainval.iloc[val_idx], test


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)  # single logit for binary task
    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def report(probs, labels, threshold=0.5):
    auc = roc_auc_score(labels, probs)
    tn, fp, fn, tp = confusion_matrix(labels, probs >= threshold).ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    return auc, sens, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scratch", action="store_true", help="random init instead of ImageNet weights")
    ap.add_argument("--out", default="best_model.pt", help="path to save the best checkpoint")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    df = pd.read_csv(args.csv)
    train_df, val_df, test_df = patient_split(df)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} | device={device}")

    loader = lambda d, tr: DataLoader(CXRDataset(d, tr), batch_size=args.batch_size,
                                      shuffle=tr, num_workers=4, pin_memory=True)
    train_dl, val_dl, test_dl = loader(train_df, True), loader(val_df, False), loader(test_df, False)

    model = build_model(pretrained=not args.scratch).to(device)

    # Handle class imbalance: weight positives by neg/pos ratio.
    pos = train_df.label.sum()
    pos_weight = torch.tensor([(len(train_df) - pos) / max(pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc, patience, bad = 0.0, 5, 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()

        val_probs, val_labels = predict(model, val_dl, device)
        auc, sens, spec = report(val_probs, val_labels)
        print(f"epoch {epoch+1:02d} | loss {running/len(train_df):.4f} | "
              f"val AUROC {auc:.3f} sens {sens:.3f} spec {spec:.3f}")

        if auc > best_auc:
            best_auc, bad = auc, 0
            torch.save(model.state_dict(), args.out)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # Evaluate on the held-out test set ONCE, using the best validation checkpoint.
    model.load_state_dict(torch.load(args.out, map_location=device))
    test_probs, test_labels = predict(model, test_dl, device)
    auc, sens, spec = report(test_probs, test_labels)
    print(f"\nTEST | AUROC {auc:.3f} | sensitivity {sens:.3f} | specificity {spec:.3f}")


if __name__ == "__main__":
    main()
