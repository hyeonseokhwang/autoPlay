# -*- coding: utf-8 -*-
"""
행동 가중치 정책(ActionPolicy)
- reasoning_patterns 테이블을 읽어 상황별 행동 성공률/사용빈도 기반 확률 분포 생성
- 최근 성공 편향 + 사용 다양성 + 탐험(엡실론) 혼합

사용 흐름:
    policy = ActionPolicy(db_path="hero4_rag.db")
    action = policy.choose_action(situation="battle_scene", candidates=["left","right","x","z"])

가중치 계산:
    base_score = success_rate * 0.7 + norm_usage * 0.2
    최근성(last_used 시간 가중) + 소량 탐험(ε=0.05 랜덤 선택)

DB 스키마 기대:
    reasoning_patterns(situation_type TEXT, action_chosen TEXT, success_rate REAL,
                       usage_count INTEGER, last_used TEXT)
"""
from __future__ import annotations
import os
import sqlite3
import time
from typing import List, Optional
from math import exp

class ActionPolicy:
    def __init__(self, db_path: str = "hero4_rag.db"):
        self.db_path = db_path
        self._cache = {}
        self._last_load = 0.0
        self.reload_interval = 5.0  # 초
    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=5.0)
    def _load(self):
        now = time.time()
        if (now - self._last_load) < self.reload_interval:
            return
        self._cache.clear()
        if not os.path.exists(self.db_path):
            return
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT situation_type, action_chosen, success_rate, usage_count, last_used FROM reasoning_patterns"
                )
                for row in cur:
                    sit, act, sr, uc, lu = row
                    self._cache.setdefault(sit, []).append({
                        'action': act,
                        'success_rate': float(sr or 0.0),
                        'usage_count': int(uc or 0),
                        'last_used': lu or ''
                    })
        except Exception:
            pass
        self._last_load = now
    def _score(self, item: dict) -> float:
        sr = item['success_rate']
        uc = item['usage_count']
        # usage 정규화(로그 스케일)
        norm_usage = 1.0 - exp(-uc/10.0)
        base = sr * 0.7 + norm_usage * 0.2
        # 최근성 보정(최근 호출이면 +0.1) 간단 처리: last_used 존재만 가산
        if item.get('last_used'):
            base += 0.1
        return base
    def choose_action(self, situation: str, candidates: List[str]) -> Optional[str]:
        self._load()
        # 데이터 없으면 탐험 패턴(단순 순환)
        items = self._cache.get(situation) or self._cache.get('general') or []
        filtered = [i for i in items if i['action'] in candidates]
        if not filtered:
            # 작은 탐험: 상황별 순환
            if not candidates:
                return None
            idx = int(time.time()) % len(candidates)
            return candidates[idx]
        # 점수 계산 및 softmax 선택
        scored = [(self._score(it), it['action']) for it in filtered]
        # 엡실론 탐험
        import random
        if random.random() < 0.05:
            return random.choice(candidates)
        # 최고 점수 우선(추가로 동일 점수 시 랜덤)
        max_score = max(s for s, _ in scored)
        best = [a for s, a in scored if abs(s - max_score) < 1e-6]
        return random.choice(best)

__all__ = ["ActionPolicy"]
