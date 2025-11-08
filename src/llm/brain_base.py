# -*- coding: utf-8 -*-
"""
두뇌(LLM 정책) 베이스 인터페이스
- 입력: 요약된 상태(JSON 호환 dict)
- 출력: 액션 JSON 스키마
"""
from __future__ import annotations
from typing import Dict, Any, List

Action = Dict[str, Any]

class BrainBase:
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """상태를 받아 행동 JSON을 반환해야 합니다.
        출력 스키마:
        {
          "actions": [
            {"type": "tap", "key": "enter"},
            {"type": "key_hold", "key": "up", "ms": 200}
          ]
        }
        """
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
