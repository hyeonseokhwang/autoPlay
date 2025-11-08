# -*- coding: utf-8 -*-
"""
스크린샷 도구
- DOSBox(혹은 지정 윈도우) 화면을 캡처하여 PNG로 저장합니다.

사용 예:
  python screenshot.py --count 1
  python screenshot.py --count 10 --interval 0.5 --out screenshots --prefix hero4 --window-title DOSBox
  python screenshot.py --region 100,100,320,240 --open

기본 동작:
- 윈도우 제목 기본값은 config/settings.json의 screen_capture.window_title 또는 'DOSBox'
- 저장 경로 기본값은 ./screenshots/
- 파일명: {prefix}_YYYYMMDD_HHMMSS_mmm.png
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import json

# 내부 모듈
from src.screen_capture import ScreenCapture


def load_default_window_title(config_path: str = "config/settings.json", fallback: str = "DOSBox") -> str:
    """설정 파일에서 기본 윈도우 타이틀을 가져옵니다."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("screen_capture", {}).get("window_title", fallback)
    except Exception:
        return fallback


def parse_region(region_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """"x,y,w,h" 형식의 문자열을 파싱합니다."""
    if not region_str:
        return None
    try:
        parts = [int(p.strip()) for p in region_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        x, y, w, h = parts
        if w <= 0 or h <= 0:
            raise ValueError
        return x, y, w, h
    except Exception:
        raise argparse.ArgumentTypeError("--region 은 'x,y,w,h' 형식이어야 하며 w,h>0 이어야 합니다.")


def ts_now() -> str:
    """밀리초 포함 타임스탬프 문자열"""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_") + f"{int(now.microsecond/1000):03d}"


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def clamp_region(w: int, h: int, region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """화면 크기에 맞춰 영역을 클램프합니다."""
    x, y, rw, rh = region
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return x, y, rw, rh


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DOSBox 화면 스크린샷 저장기")
    parser.add_argument("--window-title", dest="window_title", default=load_default_window_title(), help="대상 윈도우 제목(기본: 설정 또는 DOSBox)")
    parser.add_argument("--out", dest="out", default="screenshot", help="출력 디렉토리 (기본: screenshot)")
    parser.add_argument("--prefix", dest="prefix", default="shot", help="파일명 접두 (기본: shot)")
    parser.add_argument("--count", dest="count", type=int, default=1, help="캡처 횟수 (0=무한, 기본:1)")
    parser.add_argument("--interval", dest="interval", type=float, default=1.0, help="캡처 간격 초 (기본:1.0)")
    parser.add_argument("--region", dest="region", type=str, default=None, help="캡처 영역 'x,y,w,h' (선택)")
    parser.add_argument("--open", dest="show", action="store_true", help="마지막 이미지를 창으로 표시")
    parser.add_argument("--delay", dest="delay", type=float, default=0.0, help="시작 전 대기 초")
    parser.add_argument("--quiet", dest="quiet", action="store_true", help="로그 최소화")

    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    ensure_out_dir(out_dir)

    cap = ScreenCapture(window_title=args.window_title)

    # 시작 대기
    if args.delay > 0:
        if not args.quiet:
            print(f"시작 대기 {args.delay:.2f}s...")
        time.sleep(args.delay)

    # 윈도우 찾기 재시도
    if not cap.find_window():
        retries = 5
        for i in range(retries):
            if not args.quiet:
                print(f"윈도우를 찾지 못했습니다. 재시도 {i+1}/{retries}...", flush=True)
            time.sleep(2.0)
            if cap.find_window():
                break
        else:
            print(f"대상 윈도우를 찾을 수 없습니다: '{args.window_title}'")
            return 1

    # region 파싱
    region_tuple: Optional[Tuple[int, int, int, int]] = None
    if args.region:
        region_tuple = parse_region(args.region)

    saved_paths = []

    def capture_once() -> Optional[str]:
        img = cap.capture_screen()
        if img is None:
            return None
        # region 적용
        if region_tuple is not None:
            x, y, rw, rh = region_tuple
            # 화면 크기 클램프
            h, w = img.shape[:2]
            x, y, rw, rh = clamp_region(w, h, (x, y, rw, rh))
            img = img[y:y+rh, x:x+rw]
        # 저장
        fname = f"{args.prefix}_{ts_now()}.png"
        fpath = str(out_dir / fname)
        ok = cv2.imwrite(fpath, img)
        if ok:
            if not args.quiet:
                print(f"스크린샷 저장: {fpath}")
            return fpath
        else:
            print("이미지 저장 실패")
            return None

    # 루프
    n = 0
    last_path: Optional[str] = None
    while True:
        # 캡처 재시도(최대 3회)
        got = None
        for _ in range(3):
            got = capture_once()
            if got:
                break
            time.sleep(0.2)
        if got:
            saved_paths.append(got)
            last_path = got
            n += 1
        else:
            if not args.quiet:
                print("캡처 실패 (3회 재시도) → 건너뜀")

        if args.count != 0 and n >= args.count:
            break
        time.sleep(max(0.0, args.interval))

    # 표시
    if args.show and last_path is not None:
        img = cv2.imread(last_path)
        if img is not None:
            cv2.imshow("screenshot", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # 결과 요약
    if not args.quiet:
        print(f"총 저장 {len(saved_paths)}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
