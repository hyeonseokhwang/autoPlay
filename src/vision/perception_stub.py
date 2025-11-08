# -*- coding: utf-8 -*-
"""
화면 인지 스텁
- 입력: BGR ndarray
- 출력: 상태 요약 dict (LLM 프롬프트에 들어갈 가벼운 요약)
"""
from __future__ import annotations
from typing import Dict, Any
import time

class Perception:
    def summarize(self, frame) -> Dict[str, Any]:
        if frame is None:
            return {"scene": "unknown", "objects": [], "cursor": None, "ts": time.time()}
        h, w = frame.shape[:2]
        return {
            "scene": "unknown",  # 추후 분류기 교체
            "objects": [],
            "cursor": None,
            "resolution": [w, h],
            "ts": time.time()
        }
