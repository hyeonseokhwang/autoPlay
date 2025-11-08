# -*- coding: utf-8 -*-
"""
라벨 데이터 증가를 감지해 자동으로 분류기를 재학습합니다.
- 모니터링: data/state_frames/{menu,battle,field}
- 변화량이 min-growth 이상이면 train_state_classifier.py 호출

예)
  python scripts/auto_retrain.py --data-dir data/state_frames --min-growth 60 --interval 90 --epochs 8
"""
from __future__ import annotations
import os
import time
import argparse
from pathlib import Path
import subprocess

CLASSES = ["menu", "battle", "field"]


def count_images(root: Path) -> int:
    total = 0
    for cls in CLASSES:
        p = root / cls
        if p.exists():
            total += len(list(p.glob('*.png')))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, default='data/state_frames')
    ap.add_argument('--interval', type=float, default=60.0, help='폴더 모니터링 간격(초)')
    ap.add_argument('--min-growth', type=int, default=50, help='재학습 트리거 최소 증가 수')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--model-out', type=str, default='models/state_cnn.pt')
    args = ap.parse_args()

    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    baseline = count_images(root)
    print(f"[auto-train] start. baseline={baseline} dir={root}")

    while True:
        time.sleep(args.interval)
        now = count_images(root)
        growth = now - baseline
        if growth >= args.min_growth:
            print(f"[auto-train] growth detected: +{growth}. training...")
            # 학습 실행
            cmd = [
                'python', 'scripts/train_state_classifier.py',
                '--data-dir', str(root),
                '--epochs', str(args.epochs),
                '--lr', str(args.lr),
                '--batch', str(args.batch),
                '--out', args.model_out,
            ]
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:
                print('[auto-train] train failed:', e)
            # 기준 업데이트
            baseline = count_images(root)
            print(f"[auto-train] training done. new baseline={baseline}")
        else:
            print(f"[auto-train] tick. total={now} (+{growth}) waiting...")


if __name__ == '__main__':
    main()
