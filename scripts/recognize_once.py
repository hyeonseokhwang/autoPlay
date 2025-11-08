# -*- coding: utf-8 -*-
"""
컨트롤 없이 화면 인식만 수행하는 원샷 스크립트
- 환경 변수로 타깃 창 / 캡처 모드 설정
- 한 프레임 캡처 후, 특징값 출력 + 스냅샷 저장

Windows cmd 예:
  set HERO4_WIN_TITLE=DOSBox
  set HERO4_PROC_EXE=dosbox
  set HERO4_CAPTURE_MODE=window  # 또는 client | wgc | obs
  python scripts\recognize_once.py
"""
from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from src.recognition import ScreenRecognizer
from src.state_classifier import StateClassifier

if __name__ == "__main__":
    out_dir = Path(os.environ.get('HERO4_SNAPSHOT_DIR', 'snapshots'))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f"recognize_{ts}.png"

    recog = ScreenRecognizer()
    result = recog.read()
    clf = None
    try:
        clf = StateClassifier.from_env()
    except Exception:
        clf = None

    mode = os.environ.get('HERO4_CAPTURE_MODE', 'window')
    if result.image is None:
        print(f"[recognize] no frame. mode={mode} meta={result.meta}")
    else:
        recog.save(result.image, str(out_path))
        feats = result.features
        print("[recognize] saved:", out_path)
        print("  mode:", result.meta.get('mode'))
        if result.meta.get('error'):
            print("  backend-error:", result.meta.get('error'))
        if clf is not None:
            pred = clf.predict(result.image)
            print("  scene:", pred.label, f"(conf {pred.confidence:.2f})")
            print("  scene-probs:", {k: round(v, 3) for k, v in pred.probs.items()})
        print("  features:")
        print("    brightness:", f"{feats.get('brightness',0):.1f}")
        print("    movement:", f"{feats.get('movement',0):.2f}")
        print("    edge_h:", f"{feats.get('edge_h',0):.1f}")
        print("    edge_v:", f"{feats.get('edge_v',0):.1f}")
        print("    red_ratio:", f"{feats.get('red_ratio',0):.3f}")
        print("    blue_ratio:", f"{feats.get('blue_ratio',0):.3f}")
