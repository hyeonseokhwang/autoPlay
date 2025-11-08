#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고속 적응형 영웅전설4 AI - 빠른 분석과 능동적 탐험
"""

import asyncio
import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import traceback

# 게임 제어
import cv2
import numpy as np
import pyautogui
import win32gui
import win32con
import win32api

@dataclass
class GameAction:
    """게임 액션"""
    name: str
    keys: List[str] 
    description: str
    success_count: int = 0
    total_count: int = 0
    last_success_time: float = 0
    
    @property
    def success_rate(self):
        if self.total_count == 0:
            return 0.5
        return self.success_count / self.total_count
    
    @property
    def recent_success(self):
        return time.time() - self.last_success_time < 10  # 최근 10초 내 성공

class FastGameVision:
    """고속 게임 화면 분석"""
    
    def __init__(self):
        self.window_title_keywords = ["DOSBox", "dosbox", "ED4"]
        self.last_screenshot = None
        self.last_analysis = None
        self.analysis_cache_time = 0
        
    def get_game_window(self):
        """DOSBox 윈도우 찾기 - 캐시 사용"""
        if not hasattr(self, '_cached_hwnd'):
            def enum_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if any(keyword in title for keyword in self.window_title_keywords):
                        windows.append(hwnd)
                return True
                
            windows = []
            win32gui.EnumWindows(enum_callback, windows)
            self._cached_hwnd = windows[0] if windows else None
        
        return self._cached_hwnd
    
    def quick_capture(self) -> Optional[np.ndarray]:
        """빠른 화면 캡처"""
        try:
            hwnd = self.get_game_window()
            if not hwnd:
                return None
                
            # 윈도우 영역 캐싱
            if not hasattr(self, '_cached_rect'):
                self._cached_rect = win32gui.GetWindowRect(hwnd)
            
            x, y, x2, y2 = self._cached_rect
            width = x2 - x
            height = y2 - y
            
            # 빠른 스크린샷
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            return np.array(screenshot)
            
        except Exception as e:
            # 캐시 무효화
            if hasattr(self, '_cached_rect'):
                delattr(self, '_cached_rect')
            return None
    
    def fast_analyze(self, image: np.ndarray) -> Dict:
        """초고속 화면 분석"""
        if image is None:
            return {'type': 'unknown', 'confidence': 0.0, 'action_hint': 'wait'}
        
        try:
            # 이미지 크기 축소로 속도 향상
            height, width = image.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # 빠른 색상 분석
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 기본 통계 (빠른 계산)
            brightness = np.mean(gray)
            
            # 주요 색상 비율 (샘플링으로 속도 향상)
            sample_hsv = hsv[::4, ::4]  # 1/16 샘플링
            
            # 파란색 (메뉴)
            blue_mask = cv2.inRange(sample_hsv, (100, 50, 50), (130, 255, 255))
            blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
            
            # 빨간색 (전투/경고)
            red_mask1 = cv2.inRange(sample_hsv, (0, 50, 50), (10, 255, 255))
            red_mask2 = cv2.inRange(sample_hsv, (170, 50, 50), (180, 255, 255))
            red_ratio = (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / blue_mask.size
            
            # 초록색 (HP/상태)
            green_mask = cv2.inRange(sample_hsv, (40, 50, 50), (80, 255, 255))
            green_ratio = np.sum(green_mask > 0) / blue_mask.size
            
            # 흰색/밝은색 (텍스트)
            white_mask = gray[::4, ::4] > 200
            white_ratio = np.sum(white_mask) / white_mask.size
            
            # 빠른 상황 판단
            screen_type = 'field'
            action_hint = 'explore'
            confidence = 0.6
            
            if blue_ratio > 0.15:
                screen_type = 'menu'
                action_hint = 'navigate'
                confidence = min(0.9, blue_ratio * 6)
                
            elif red_ratio > 0.08:
                screen_type = 'battle'
                action_hint = 'fight'
                confidence = min(0.9, red_ratio * 10)
                
            elif white_ratio > 0.3:
                screen_type = 'dialogue'
                action_hint = 'read'
                confidence = min(0.9, white_ratio * 2)
                
            elif brightness < 40:
                screen_type = 'dark'
                action_hint = 'search'
                confidence = 0.7
                
            elif green_ratio > 0.05:
                screen_type = 'status'
                action_hint = 'check'
                confidence = 0.8
            
            return {
                'type': screen_type,
                'confidence': confidence,
                'action_hint': action_hint,
                'stats': {
                    'brightness': brightness,
                    'blue': blue_ratio,
                    'red': red_ratio,
                    'green': green_ratio,
                    'white': white_ratio
                }
            }
            
        except Exception as e:
            return {'type': 'error', 'confidence': 0.0, 'action_hint': 'wait'}

class HighSpeedController:
    """고속 게임 컨트롤러"""
    
    def __init__(self):
        self.window_title_keywords = ["DOSBox", "dosbox", "ED4"]
        self.key_queue = []
        self.last_key_time = 0
        
    def get_game_window(self):
        """게임 윈도우 찾기 - 캐시 사용"""
        if not hasattr(self, '_cached_hwnd'):
            def enum_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if any(keyword in title for keyword in self.window_title_keywords):
                        windows.append(hwnd)
                return True
                
            windows = []
            win32gui.EnumWindows(enum_callback, windows)
            self._cached_hwnd = windows[0] if windows else None
        
        return self._cached_hwnd
    
    def rapid_key_input(self, key: str) -> bool:
        """빠른 키 입력"""
        try:
            current_time = time.time()
            
            # 키 입력 간격 제한 (0.15초)
            if current_time - self.last_key_time < 0.15:
                return False
            
            self.last_key_time = current_time
            
            # 윈도우 활성화 (캐싱된 핸들 사용)
            hwnd = self.get_game_window()
            if not hwnd:
                return False
            
            # 키 맵핑
            key_map = {
                'up': win32con.VK_UP,
                'down': win32con.VK_DOWN,
                'left': win32con.VK_LEFT,
                'right': win32con.VK_RIGHT,
                'enter': win32con.VK_RETURN,
                'space': win32con.VK_SPACE,
                'esc': win32con.VK_ESCAPE,
                'z': ord('Z'),
                'x': ord('X'),
                'c': ord('C'),
                'a': ord('A'),
                's': ord('S'),
                'd': ord('D')
            }
            
            if key.lower() not in key_map:
                return False
            
            vk_code = key_map[key.lower()]
            
            # 빠른 키 입력 (홀드 시간 단축)
            win32api.keybd_event(vk_code, 0, 0, 0)
            time.sleep(0.05)  # 50ms만 홀드
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            
            return True
            
        except Exception:
            return False

class AggressiveAI:
    """적극적이고 빠른 AI"""
    
    def __init__(self):
        self.vision = FastGameVision()
        self.controller = HighSpeedController()
        
        # 확장된 액션 세트
        self.actions = [
            # 기본 이동
            GameAction("move_up", ["up"], "위로 이동"),
            GameAction("move_down", ["down"], "아래로 이동"), 
            GameAction("move_left", ["left"], "왼쪽으로 이동"),
            GameAction("move_right", ["right"], "오른쪽으로 이동"),
            
            # 상호작용
            GameAction("confirm", ["enter"], "확인/선택"),
            GameAction("cancel", ["esc"], "취소/뒤로"),
            GameAction("action", ["space"], "조사/액션"),
            
            # 대안 키들
            GameAction("alt_confirm", ["z"], "Z키 확인"),
            GameAction("alt_cancel", ["x"], "X키 취소"),
            GameAction("special_a", ["a"], "A키 액션"),
            GameAction("special_s", ["s"], "S키 스킬"),
            GameAction("special_d", ["d"], "D키 방어"),
            
            # 조합 액션
            GameAction("double_enter", ["enter", "enter"], "연속 확인"),
            GameAction("explore_combo", ["space", "enter"], "조사 후 확인"),
            GameAction("menu_escape", ["esc", "esc"], "강제 메뉴 탈출")
        ]
        
        # AI 상태
        self.exploration_mode = True
        self.aggressive_level = 1.0
        self.screen_change_history = []
        self.action_sequence = []
        self.stuck_counter = 0
        self.success_streak = 0
        
    def adaptive_action_selection(self, screen_info: Dict) -> GameAction:
        """적응적 액션 선택"""
        screen_type = screen_info.get('type', 'field')
        action_hint = screen_info.get('action_hint', 'explore')
        confidence = screen_info.get('confidence', 0.5)
        
        # 화면 변화 추적
        self.screen_change_history.append(screen_type)
        if len(self.screen_change_history) > 8:
            self.screen_change_history.pop(0)
        
        # 변화 없음 감지
        recent_screens = self.screen_change_history[-5:]
        if len(set(recent_screens)) <= 1 and len(recent_screens) >= 5:
            self.stuck_counter += 1
            self.aggressive_level = min(2.0, self.aggressive_level + 0.1)
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
            self.aggressive_level = max(0.5, self.aggressive_level - 0.05)
        
        # 액션 후보 필터링
        candidates = []
        
        if action_hint == 'navigate' or screen_type == 'menu':
            # 메뉴 네비게이션
            if self.stuck_counter > 3:
                candidates = [a for a in self.actions if 'cancel' in a.name or 'escape' in a.name]
            else:
                candidates = [a for a in self.actions if a.name in [
                    'move_up', 'move_down', 'confirm', 'alt_confirm'
                ]]
                
        elif action_hint == 'fight' or screen_type == 'battle':
            # 전투 액션
            candidates = [a for a in self.actions if a.name in [
                'confirm', 'special_a', 'special_s', 'special_d', 
                'move_up', 'move_down', 'alt_confirm'
            ]]
            
        elif action_hint == 'read' or screen_type == 'dialogue':
            # 대화 진행
            candidates = [a for a in self.actions if a.name in [
                'confirm', 'double_enter', 'alt_confirm', 'space'
            ]]
            
        elif action_hint == 'search' or screen_type == 'dark':
            # 탐색 모드
            candidates = [a for a in self.actions if a.name in [
                'move_up', 'move_down', 'move_left', 'move_right',
                'action', 'explore_combo'
            ]]
            
        else:
            # 필드 탐험 (기본)
            if self.exploration_mode and self.stuck_counter < 5:
                # 정상 탐험
                candidates = [a for a in self.actions if a.name in [
                    'move_up', 'move_down', 'move_left', 'move_right', 
                    'action', 'explore_combo', 'confirm'
                ]]
            else:
                # 적극적 탐험
                candidates = self.actions  # 모든 액션 시도
        
        # 후보가 없으면 전체에서 선택
        if not candidates:
            candidates = self.actions
        
        # 성공률 기반 가중치 계산
        weights = []
        for action in candidates:
            base_weight = action.success_rate if action.total_count > 0 else 0.5
            
            # 적극성 레벨 적용
            if 'move' in action.name:
                base_weight *= self.aggressive_level
            
            # 최근 성공 보너스
            if action.recent_success:
                base_weight *= 1.3
            
            # 연속 성공 보너스
            if len(self.action_sequence) > 0 and self.action_sequence[-1] == action.name:
                if self.success_streak > 2:
                    base_weight *= 1.5
            
            # 막힘 상황에서는 새로운 액션 선호
            if self.stuck_counter > 5:
                if action.total_count == 0:  # 시도해보지 않은 액션
                    base_weight *= 2.0
            
            weights.append(max(0.1, base_weight))
        
        # 가중치 기반 선택
        selected = random.choices(candidates, weights=weights)[0]
        
        return selected
    
    def execute_rapid_action(self, action: GameAction) -> bool:
        """빠른 액션 실행"""
        try:
            success_count = 0
            
            for key in action.keys:
                if self.controller.rapid_key_input(key):
                    success_count += 1
                time.sleep(0.05)  # 키 간 최소 간격
            
            success = success_count == len(action.keys)
            
            # 통계 업데이트
            action.total_count += 1
            if success:
                action.success_count += 1
                action.last_success_time = time.time()
                self.success_streak += 1
            else:
                self.success_streak = 0
            
            # 액션 히스토리
            self.action_sequence.append(action.name)
            if len(self.action_sequence) > 15:
                self.action_sequence.pop(0)
            
            return success
            
        except Exception:
            return False
    
    async def rapid_play_cycle(self) -> Dict:
        """빠른 플레이 사이클"""
        try:
            # 빠른 화면 캡처 (0.1초 이내)
            screenshot = self.vision.quick_capture()
            if screenshot is None:
                return {'success': False, 'error': '화면 캡처 실패'}
            
            # 빠른 분석 (0.05초 이내)
            screen_info = self.vision.fast_analyze(screenshot)
            
            # 적응적 액션 선택 (즉시)
            action = self.adaptive_action_selection(screen_info)
            
            # 빠른 실행
            success = self.execute_rapid_action(action)
            
            return {
                'success': success,
                'action': action.name,
                'screen_type': screen_info.get('type'),
                'confidence': screen_info.get('confidence', 0),
                'aggressive_level': self.aggressive_level,
                'stuck_counter': self.stuck_counter,
                'success_streak': self.success_streak
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

async def main():
    """고속 메인 루프"""
    print("⚡ 고속 적응형 영웅전설4 AI")
    print("=" * 40)
    
    ai = AggressiveAI()
    
    # 게임 연결 확인
    screenshot = ai.vision.quick_capture()
    if screenshot is None:
        print("❌ 게임을 찾을 수 없습니다. DOSBox를 실행하세요.")
        return
    
    print("🎮 게임 감지! 고속 플레이 시작...")
    print("🚀 분석 주기: 0.3초, 적극적 탐험 모드")
    
    # 고속 플레이 루프
    total_cycles = 200  # 200사이클 = 약 1분
    success_count = 0
    start_time = time.time()
    
    for cycle in range(1, total_cycles + 1):
        cycle_start = time.time()
        
        # 빠른 플레이 사이클
        result = await ai.rapid_play_cycle()
        
        if result['success']:
            success_count += 1
            status = "✅"
        else:
            status = "❌"
        
        # 간단한 진행 상황 (5사이클마다)
        if cycle % 5 == 0:
            elapsed = time.time() - start_time
            cps = cycle / elapsed  # Cycles Per Second
            success_rate = success_count / cycle
            
            print(f"{status} #{cycle:3d} | {result.get('action', '?'):12s} | "
                  f"{result.get('screen_type', '?'):8s} | "
                  f"성공률:{success_rate:.2f} | "
                  f"속도:{cps:.1f}cps | "
                  f"적극성:{result.get('aggressive_level', 1):.1f}")
        
        # 0.3초 주기 유지
        cycle_time = time.time() - cycle_start
        if cycle_time < 0.3:
            await asyncio.sleep(0.3 - cycle_time)
    
    # 최종 통계
    total_time = time.time() - start_time
    print(f"\n⚡ 고속 플레이 완료!")
    print(f"총 사이클: {total_cycles}")
    print(f"성공: {success_count}")
    print(f"성공률: {success_count/total_cycles:.2f}")
    print(f"소요시간: {total_time:.1f}초")
    print(f"평균속도: {total_cycles/total_time:.1f} cps")
    
    # 액션별 성능
    print(f"\n📊 상위 액션 성능:")
    top_actions = sorted([a for a in ai.actions if a.total_count > 0], 
                        key=lambda x: x.success_rate, reverse=True)[:8]
    
    for action in top_actions:
        print(f"  {action.description:15s}: {action.success_rate:.2f} ({action.success_count}/{action.total_count})")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        traceback.print_exc()