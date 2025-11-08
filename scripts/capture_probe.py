# -*- coding: utf-8 -*-
"""
단계 1: 화면 캡처/인식만 검증하는 프로브 스크립트
- 입력/포커스 변경 없음
- 일정 주기로 프레임을 받아 간단 특징값과 함께 저장

실행(Windows cmd):

  set HERO4_WIN_TITLE=DOSBox
  set HERO4_PROC_EXE=dosbox
  set HERO4_CAPTURE_MODE=wgc   # 또는 obs | client | window | frame
  python scripts/capture_probe.py

종료: Ctrl+C
"""
import os
import time
from datetime import datetime
import argparse
from pathlib import Path

from PIL import Image

from src.screen_pipeline import build_capture_pipeline

SAVE_ROOT = Path(os.environ.get('HERO4_SNAPSHOT_DIR', 'snapshots')) / f"probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0=run forever)")
args = parser.parse_args()

locator, capture, analyzer = build_capture_pipeline()

print("[probe] start. mode=", os.environ.get('HERO4_CAPTURE_MODE','window'))

step = 0
try:
    while True:
        step += 1
        res = capture.get_frame()
        if res.image is None:
            if step % 30 == 0:
                print("[probe] no frame (", res.meta, ")")
            time.sleep(0.05)
            continue
        meta = analyzer.analyze(res.image)
        if step % 10 == 0:
            print(f"[probe] step {step} | mode={res.meta.get('mode')} | bbox={res.bbox} | br={meta.get('brightness'):.1f} mv={meta.get('movement',0):.1f}")
        if step % 20 == 0:
            raw_path = SAVE_ROOT / f"probe_{step:06d}.png"
            res.image.save(raw_path)
        time.sleep(0.02)
        if args.max_steps and step >= args.max_steps:
            break
except KeyboardInterrupt:
    print("[probe] stop")
finally:
    print("[probe] done. saved to:", SAVE_ROOT)
