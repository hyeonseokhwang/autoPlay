# -*- coding: utf-8 -*-
"""
화면 인식 모듈 (컨트롤 분리)
- GameWindowLocator + GameCapture + FrameAnalyzer 래핑
- 한 번의 호출로 이미지와 특징값을 반환
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict
from PIL import Image
from .screen_pipeline import build_capture_pipeline

@dataclass
class RecognitionResult:
    image: Optional[Image.Image]
    features: Dict
    meta: Dict

class ScreenRecognizer:
    def __init__(self):
        self.locator, self.capture, self.analyzer = build_capture_pipeline()
    def read(self) -> RecognitionResult:
        res = self.capture.get_frame()
        img = res.image
        feats: Dict = {}
        if img is not None:
            feats = self.analyzer.analyze(img)
        return RecognitionResult(img, feats, res.meta or {})
    def save(self, img: Image.Image, path: str) -> None:
        if img is None:
            return
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        try:
            # 이미지가 비정상 크기면 예외 발생할 수 있어 안전한 복사본 사용
            safe = img.copy()
            safe.save(path)
        except Exception as e:
            # PNG 저장 실패 시 JPEG로 재시도
            try:
                alt = path.replace('.png', '_fallback.jpg')
                img.convert('RGB').save(alt, quality=90)
            except Exception:
                pass
