# -*- coding: utf-8 -*-
"""
state_classifier 학습 스크립트
- 폴더 구조: data/state_frames/{menu,battle,field}/*.png
- 출력: models/state_cnn.pt

예)
  python scripts/train_state_classifier.py --data-dir data/state_frames --epochs 8 --lr 3e-4
""" 
from __future__ import annotations
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.state_classifier import _StateCNN, CLASS_NAMES

class FrameFolderDataset(Dataset):
    def __init__(self, root: str, size=(160,120)):
        self.items = []
        self.size = size
        root_p = Path(root)
        for idx, cls in enumerate(CLASS_NAMES):
            for p in (root_p/cls).glob('*.png'):
                self.items.append((str(p), idx))
        random.shuffle(self.items)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert('RGB').resize(self.size)
        arr = np.asarray(img).astype(np.float32)/255.0
        t = torch.from_numpy(arr).permute(2,0,1)
        return t, torch.tensor(label, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, default='data/state_frames')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', type=str, default='models/state_cnn.pt')
    args = ap.parse_args()

    ds = FrameFolderDataset(args.data_dir)
    if len(ds) < 30:
        print('[train] 데이터가 부족합니다. 최소 30장 이상 라벨링 해주세요.')
        return
    n_val = max(10, int(0.1*len(ds)))
    val_items = ds.items[:n_val]
    train_items = ds.items[n_val:]
    train_ds = FrameFolderDataset(args.data_dir)
    train_ds.items = train_items
    val_ds = FrameFolderDataset(args.data_dir)
    val_ds.items = val_items

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _StateCNN(num_classes=len(CLASS_NAMES)).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    Path(os.path.dirname(args.out) or '.').mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x,y in train_loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); opt.step()
            loss_sum += float(loss.item())*x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred==y).sum().item())
            total += int(x.size(0))
        train_acc = correct/total if total else 0.0
        train_loss = loss_sum/total if total else 0.0

        model.eval()
        v_total, v_correct = 0,0
        with torch.no_grad():
            for x,y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                v_correct += int((pred==y).sum().item())
                v_total += int(x.size(0))
        val_acc = v_correct/v_total if v_total else 0.0

        print(f"[epoch {epoch}] train_acc={train_acc:.3f} val_acc={val_acc:.3f} loss={train_loss:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"[save] {args.out} (val_acc={best_acc:.3f})")

if __name__ == '__main__':
    main()
