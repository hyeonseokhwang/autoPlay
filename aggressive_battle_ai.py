#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 적극적 전투 AI
실제 전투를 찾기 위한 더 공격적인 전략
"""

import asyncio
import time
import random
import numpy as np
import cv2
from typing import Optional, Dict, Any
import win32gui
import win32con
import win32api
from PIL import ImageGrab, Image

class AggressiveBattleAI:
    """영웅전설4 적극적 전투 찾기 AI"""
    
    def __init__(self):
        """초기화"""
        self.battle_count = 0
        self.total_actions = 0
        self.start_time = time.time()
        self.last_screenshot = None
        self.hero4_window = None
        self.capture_region = None
        
        # 전투 찾기 전략 설정
        self.exploration_patterns = [
            # 맵 전체 탐색 패턴
            ['right', 'right', 'right', 'down', 'left', 'left', 'left', 'up'],     # 사각형
            ['up', 'up', 'right', 'right', 'down', 'down', 'left', 'left'],       # 큰 사각형
            ['right', 'down', 'right', 'up', 'left', 'down', 'left', 'up'],       # 지그재그
            ['up', 'right', 'down', 'left'] * 3,                                   # 작은 원형 반복
        ]
        
        # 상호작용 키들 (NPC, 문, 아이템 등)
        self.interaction_keys = ['space', 'enter', 'z', 'x', 'a', 's', '1', '2', '3']
        
        # 메뉴/전투 키들
        self.battle_keys = ['z', 'x', 'a', 's', 'enter', 'space', '1', '2', '3', '4']
        
        self.current_pattern = 0
        self.pattern_step = 0
        
    def find_hero4_window(self) -> bool:
        """영웅전설4 창 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                
                # 영웅전설4 관련 창 찾기
                if ('ED4' in window_text or 
                    'dosbox' in window_text.lower() or
                    'DOS' in window_text):
                    windows.append((hwnd, window_text, class_name))
            return True

        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            self.hero4_window = windows[0][0]  # 첫 번째 창 사용
            
            # 창 정보 및 캡처 영역 설정
            rect = win32gui.GetWindowRect(self.hero4_window)
            self.capture_region = rect
            print(f"🎮 영웅전설4 창 발견: {windows[0][1]} (영역: {rect})")
            return True
        
        print("❌ 영웅전설4 창을 찾을 수 없습니다!")
        return False
    
    def send_key(self, key: str) -> bool:
        """키 입력 전송"""
        if not self.hero4_window:
            return False
            
        # 창을 활성화
        try:
            win32gui.SetForegroundWindow(self.hero4_window)
        except:
            pass
            
        time.sleep(0.05)  # 매우 짧은 대기
        
        # 키 매핑
        key_map = {
            'left': 0x25, 'right': 0x27, 'up': 0x26, 'down': 0x28,
            'space': 0x20, 'enter': 0x0D, 'esc': 0x1B,
            'z': 0x5A, 'x': 0x58, 'a': 0x41, 's': 0x53,
            '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34
        }
        
        if key in key_map:
            vk_code = key_map[key]
            # 키 누르기 + 떼기
            win32api.keybd_event(vk_code, 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            print(f"🎮 키 입력: {key.upper()}")
            return True
        
        return False
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """화면 캡처"""
        if not self.capture_region:
            return None
            
        try:
            screenshot = ImageGrab.grab(self.capture_region)
            self.last_screenshot = screenshot
            return np.array(screenshot)
        except Exception as e:
            print(f"❌ 화면 캡처 실패: {e}")
            return None
    
    def analyze_for_battle(self, image: np.ndarray) -> Dict[str, Any]:
        """전투 상황 분석"""
        if image is None:
            return {"battle_likely": False, "confidence": 0}
            
        # HSV 색상 공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 빨간색 (HP, 데미지, 적 표시) 감지
        red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_pixels = np.sum(red_mask1) + np.sum(red_mask2)
        
        # 파란색 (마나, UI) 감지  
        blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
        blue_pixels = np.sum(blue_mask)
        
        # 녹색 (상태창, HP 풀) 감지
        green_mask = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))
        green_pixels = np.sum(green_mask)
        
        # 노란색 (경험치, 골드) 감지
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
        yellow_pixels = np.sum(yellow_mask)
        
        total_pixels = image.shape[0] * image.shape[1] * 3
        
        # 색상 비율 계산
        red_ratio = red_pixels / total_pixels
        blue_ratio = blue_pixels / total_pixels
        green_ratio = green_pixels / total_pixels
        yellow_ratio = yellow_pixels / total_pixels
        
        # 화면 밝기
        brightness = np.mean(image)
        
        # 전투 가능성 판단
        battle_indicators = 0
        confidence = 0
        
        if red_ratio > 0.05:    # 빨간색 5% 이상
            battle_indicators += 3
            confidence += 30
            
        if blue_ratio > 0.1:    # 파란색 10% 이상
            battle_indicators += 2
            confidence += 20
            
        if green_ratio > 0.08:  # 녹색 8% 이상
            battle_indicators += 2
            confidence += 20
            
        if yellow_ratio > 0.03: # 노란색 3% 이상
            battle_indicators += 1
            confidence += 10
            
        if brightness > 100:    # 밝은 화면 (UI 활성)
            battle_indicators += 1
            confidence += 10
        
        # 급격한 화면 변화 감지
        change_detected = False
        if hasattr(self, 'last_analysis'):
            last_brightness = self.last_analysis.get('brightness', brightness)
            if abs(brightness - last_brightness) > 20:
                change_detected = True
                battle_indicators += 2
                confidence += 20
        
        battle_likely = battle_indicators >= 3 or confidence >= 50
        
        result = {
            "battle_likely": battle_likely,
            "confidence": min(confidence, 100),
            "indicators": battle_indicators,
            "red_ratio": red_ratio,
            "blue_ratio": blue_ratio, 
            "green_ratio": green_ratio,
            "yellow_ratio": yellow_ratio,
            "brightness": brightness,
            "change_detected": change_detected
        }
        
        self.last_analysis = result
        return result
    
    def get_next_action(self) -> str:
        """다음 행동 결정"""
        # 현재 패턴에서 다음 행동
        pattern = self.exploration_patterns[self.current_pattern]
        action = pattern[self.pattern_step % len(pattern)]
        
        # 패턴 진행
        self.pattern_step += 1
        if self.pattern_step >= len(pattern) * 2:  # 패턴 2회 반복 후 변경
            self.current_pattern = (self.current_pattern + 1) % len(self.exploration_patterns)
            self.pattern_step = 0
            print(f"🔄 탐색 패턴 변경: {self.current_pattern}")
        
        # 가끔 상호작용 시도
        if self.total_actions % 7 == 0:
            action = random.choice(self.interaction_keys)
            print(f"🤝 상호작용 시도: {action}")
        
        return action
    
    async def battle_response(self, analysis: Dict[str, Any]) -> None:
        """전투 대응"""
        print(f"⚔️ 전투 감지! 신뢰도: {analysis['confidence']}%")
        
        # 전투 액션 시퀀스
        battle_sequence = [
            'z',      # 공격
            'enter',  # 확인
            'a',      # 아이템/스킬
            '1',      # 선택 1
            'space',  # 스페이스
            'x',      # 취소/뒤로
            '2',      # 선택 2
            's'       # S키
        ]
        
        # 빠른 전투 액션
        for i in range(min(4, len(battle_sequence))):
            action = battle_sequence[i % len(battle_sequence)]
            self.send_key(action)
            await asyncio.sleep(0.1)
            
        self.battle_count += 1
        print(f"🏆 전투 완료! 총 {self.battle_count}회")
    
    async def run(self, max_actions: int = 200, target_battles: int = 10) -> None:
        """메인 실행"""
        print("🚀 영웅전설4 적극적 전투 AI 시작!")
        print(f"🎯 목표: {target_battles}회 전투, 최대 {max_actions}회 행동")
        
        if not self.find_hero4_window():
            print("❌ 영웅전설4를 실행한 후 다시 시도하세요!")
            return
        
        while self.total_actions < max_actions and self.battle_count < target_battles:
            self.total_actions += 1
            
            # 1. 화면 분석
            current_screen = self.capture_screen()
            if current_screen is not None:
                analysis = self.analyze_for_battle(current_screen)
                
                # 2. 전투 감지 시 대응
                if analysis["battle_likely"]:
                    await self.battle_response(analysis)
                    await asyncio.sleep(0.3)  # 전투 후 잠시 대기
                    continue
            
            # 3. 일반 탐색 행동
            action = self.get_next_action()
            self.send_key(action)
            
            # 4. 짧은 대기
            await asyncio.sleep(0.12)  # 빠른 행동
            
            # 5. 진행 상황 출력
            if self.total_actions % 10 == 0:
                elapsed = time.time() - self.start_time
                aps = self.total_actions / elapsed if elapsed > 0 else 0
                print(f"📊 진행: {self.total_actions}/{max_actions} | 전투:{self.battle_count}/{target_battles} | 속도:{aps:.1f}aps")
        
        # 결과 출력
        elapsed = time.time() - self.start_time
        efficiency = self.battle_count / self.total_actions if self.total_actions > 0 else 0
        
        print(f"\n🏁 완료!")
        print(f"⏱️  시간: {elapsed:.1f}초")
        print(f"⚔️  전투: {self.battle_count}/{target_battles}회")
        print(f"🎮 행동: {self.total_actions}회")
        print(f"📈 효율: {efficiency:.3f} (전투/행동)")
        
        if self.battle_count >= target_battles:
            print("🎉 목표 달성!")
        else:
            print("😅 목표 미달성")

# 실행
if __name__ == "__main__":
    async def main():
        ai = AggressiveBattleAI()
        await ai.run(max_actions=200, target_battles=10)
    
    print("🎮 영웅전설4 적극적 전투 AI")
    print("=" * 50)
    asyncio.run(main())