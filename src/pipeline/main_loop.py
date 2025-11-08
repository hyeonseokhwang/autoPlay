# -*- coding: utf-8 -*-
"""MVP 메인 루프
캡처 -> 인지 요약 -> LLM(DeepSeek 스텁) -> 컨트롤러 실행
"""
from __future__ import annotations
import time
from typing import Optional
import cv2

from src.screen_capture import ScreenCapture
from src.vision.perception_stub import Perception
from src.llm.brain_deepseek import DeepSeekBrain
from src.actuation.controller import Controller

class MainLoop:
    def __init__(self, window_title: str = "DOSBox", dry_run: bool = True):
        self.capture = ScreenCapture(window_title=window_title)
        self.perception = Perception()
        self.brain = DeepSeekBrain()  # 모델 로드 실패시 자동 더미 동작
        self.controller = Controller(dry_run=dry_run)
        self.running = False
        self.target_fps = 5  # 초기 저속

    def start(self, duration_sec: int = 10):
        if not self.capture.find_window():
            print("[MainLoop] DOSBox 윈도우를 찾지 못했습니다.")
            return
        self.running = True
        end_time = time.time() + duration_sec
        frame_interval = 1.0 / self.target_fps
        while self.running and time.time() < end_time:
            frame = self.capture.capture_screen()
            state = self.perception.summarize(frame)
            actions_json = self.brain.decide(state)
            actions = actions_json.get("actions", [])
            self.controller.execute_actions(actions)
            time.sleep(frame_interval)
        print("[MainLoop] 종료")

if __name__ == "__main__":
    loop = MainLoop(dry_run=True)
    loop.start(duration_sec=5)
