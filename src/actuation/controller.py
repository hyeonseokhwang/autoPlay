# -*- coding: utf-8 -*-
"""
키/액션 실행 컨트롤러
- Win32 메시지 기반 (향후) / 현재는 print로 시뮬레이션
"""
from __future__ import annotations
from typing import Dict, Any, List
import time

class Controller:
    def __init__(self, hwnd: int | None = None, dry_run: bool = True):
        self.hwnd = hwnd
        self.dry_run = dry_run

    def execute_actions(self, actions: List[Dict[str, Any]]):
        for act in actions:
            atype = act.get("type")
            if atype == "tap":
                key = act.get("key", "")
                self._send_key(key, down=True)
                self._send_key(key, down=False)
            elif atype == "key_hold":
                key = act.get("key", "")
                ms = int(act.get("ms", 150))
                self._send_key(key, down=True)
                time.sleep(ms/1000.0)
                self._send_key(key, down=False)
            else:
                print(f"[Controller] 알 수 없는 액션 타입: {atype}")

    def _send_key(self, key: str, down: bool):
        # TODO: win32api key 이벤트 구현 예정
        direction = "DOWN" if down else "UP"
        if self.dry_run:
            print(f"[Controller] {direction} {key}")
        else:
            # 실제 구현: PostMessage(self.hwnd, WM_KEYDOWN/UP, vk, 0)
            pass
