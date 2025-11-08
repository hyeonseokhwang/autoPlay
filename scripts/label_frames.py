# -*- coding: utf-8 -*-
"""
실시간 프레임 라벨링 스크립트
- 화면 캡처 파이프라인으로 프레임 수집
- 키 입력 (1=menu, 2=battle, 3=field, s=skip, q=quit)
- 저장 디렉터리: data/state_frames/<label>/timestamp_counter.png

사용 예 (PowerShell):
  $env:HERO4_WIN_TITLE="DOSBox"
  $env:HERO4_PROC_EXE="dosbox"
  $env:HERO4_CAPTURE_MODE="window"
  python scripts/label_frames.py --limit 300

라벨 충분히 모으면 train_state_classifier.py 실행.
"""
from __future__ import annotations
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
from src.recognition import ScreenRecognizer

LABEL_MAP = {
    '1': 'menu',
    '2': 'battle',
    '3': 'field'
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='최대 저장 프레임 수 (0=무한)')
    ap.add_argument('--interval', type=float, default=0.2, help='캡처 간격초')
    ap.add_argument('--out-dir', type=str, default='data/state_frames', help='라벨 저장 루트')
    args = ap.parse_args()

    root = Path(args.out_dir)
    for lbl in LABEL_MAP.values():
        (root / lbl).mkdir(parents=True, exist_ok=True)

    recog = ScreenRecognizer()

    print('[label] 시작. 키: 1=menu,2=battle,3=field,s=skip,q=quit')
    count = 0
    skipped = 0
    while True:
        res = recog.read()
        img = res.image
        if img is None:
            print('[label] no frame meta=', res.meta)
            time.sleep(args.interval)
            continue
        # 간단 텍스트 출력
        feats = res.features
        print(f"[frame] br={feats.get('brightness',0):.1f} mv={feats.get('movement',0):.2f} mode={res.meta.get('mode')} count={count}")
        # 임시 뷰: 작은 버전 출력(선택)
        # 라벨 입력 대기
        key = input('라벨 입력 (1/2/3/s/q): ').strip()
        if key == 'q':
            break
        if key == 's':
            skipped += 1
            continue
        label = LABEL_MAP.get(key)
        if not label:
            print('[label] 잘못된 키')
            continue
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{ts}_{count:06d}.png"
        out_path = root / label / fname
        img.save(out_path)
        count += 1
        if args.limit and count >= args.limit:
            break
        time.sleep(args.interval)
    print(f"[label] 종료. saved={count} skipped={skipped} root={root}")

if __name__ == '__main__':
    main()
