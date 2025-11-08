#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 전용 스마트 AI - 게임 룰 기반 학습 + 독립적 키 입력
"""

import asyncio
import time
import random
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib

# 게임 제어 (DOSBox 전용)
import cv2
import pyautogui
import win32gui
import win32con
import win32api
import win32process

@dataclass
class GameState:
    """게임 상태"""
    screen_hash: str
    screen_type: str
    brightness: float
    color_profile: Dict[str, float]
    movement_possible: bool
    menu_detected: bool
    text_detected: bool
    timestamp: float

class DOSBoxController:
    """DOSBox 전용 컨트롤러"""
    
    def __init__(self):
        self.dosbox_hwnd = None
        self.dosbox_pid = None
        self.last_input_time = 0
        
    def find_dosbox(self):
        """DOSBox 윈도우 찾기 및 프로세스 확인"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if any(keyword in title.lower() for keyword in ["dosbox", "ed4", "영웅전설"]):
                    try:
                        # 프로세스 ID 확인
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        windows.append((hwnd, title, pid))
                    except:
                        pass
            return True
        
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            self.dosbox_hwnd, title, self.dosbox_pid = windows[0]
            print(f"✅ DOSBox 발견: {title} (PID: {self.dosbox_pid})")
            return True
        else:
            print("❌ DOSBox를 찾을 수 없습니다!")
            return False
    
    def send_key_to_dosbox(self, key: str, hold_time: float = 0.05) -> bool:
        """DOSBox에만 키 전송"""
        if not self.dosbox_hwnd:
            return False
        
        try:
            # 입력 간격 제한
            current_time = time.time()
            if current_time - self.last_input_time < 0.1:
                return False
            
            # DOSBox 활성화 (강제)
            win32gui.SetForegroundWindow(self.dosbox_hwnd)
            win32gui.SetActiveWindow(self.dosbox_hwnd)
            time.sleep(0.02)  # 활성화 대기
            
            # 키 매핑 (영웅전설4 표준)
            key_map = {
                # 이동키 (방향키)
                'up': win32con.VK_UP,
                'down': win32con.VK_DOWN,
                'left': win32con.VK_LEFT,
                'right': win32con.VK_RIGHT,
                
                # 선택키 (엔터, 스페이스)
                'enter': win32con.VK_RETURN,
                'space': win32con.VK_SPACE,
                
                # 기타
                'esc': win32con.VK_ESCAPE,
                'z': ord('Z'),
                'x': ord('X')
            }
            
            if key.lower() not in key_map:
                return False
            
            vk_code = key_map[key.lower()]
            
            # 키 입력 (DOSBox용 최적화)
            win32api.keybd_event(vk_code, 0, 0, 0)  # 키 다운
            time.sleep(hold_time)  # 홀드
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  # 키 업
            
            self.last_input_time = current_time
            print(f"🎮 키 전송: {key.upper()} → DOSBox")
            return True
            
        except Exception as e:
            print(f"❌ 키 입력 실패: {e}")
            return False

class SmartVision:
    """스마트 화면 분석"""
    
    def __init__(self):
        self.dosbox_hwnd = None
        self.capture_region = None
        
    def setup_capture(self, hwnd):
        """캡처 영역 설정"""
        self.dosbox_hwnd = hwnd
        try:
            rect = win32gui.GetWindowRect(hwnd)
            # 윈도우 테두리 제외하고 게임 화면만
            x, y, x2, y2 = rect
            margin = 8
            self.capture_region = (x + margin, y + 30, x2 - x - margin*2, y2 - y - 38)
            print(f"📸 캡처 영역 설정: {self.capture_region}")
        except Exception as e:
            print(f"❌ 캡처 설정 실패: {e}")
    
    def capture_game_screen(self) -> Optional[np.ndarray]:
        """게임 화면만 캡처"""
        if not self.capture_region:
            return None
        
        try:
            screenshot = pyautogui.screenshot(region=self.capture_region)
            return np.array(screenshot)
        except Exception as e:
            print(f"❌ 화면 캡처 실패: {e}")
            return None
    
    def analyze_game_screen(self, image: np.ndarray) -> GameState:
        """게임 화면 분석"""
        if image is None:
            return GameState("", "unknown", 0, {}, False, False, False, time.time())
        
        # 이미지 전처리 (속도 최적화)
        height, width = image.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # 색상 분석
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 화면 해시 생성
        small_gray = cv2.resize(gray, (32, 24))
        screen_hash = hashlib.md5(small_gray.tobytes()).hexdigest()[:12]
        
        # 밝기 분석
        brightness = np.mean(gray)
        
        # 색상 프로필 분석
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        # 주요 색상 비율 계산
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        
        total_pixels = image.shape[0] * image.shape[1]
        color_profile = {
            'blue': np.sum(blue_mask > 0) / total_pixels,
            'red': (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / total_pixels,
            'green': np.sum(green_mask > 0) / total_pixels,
        }
        
        # 텍스트 감지 (밝은 영역 + 에지)
        bright_mask = gray > 200
        bright_ratio = np.sum(bright_mask) / total_pixels
        
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / total_pixels
        
        text_detected = bright_ratio > 0.15 and edge_ratio > 0.05
        
        # 메뉴 감지 (파란색 계열 + 규칙적 패턴)
        menu_detected = color_profile['blue'] > 0.1 or (brightness > 100 and edge_ratio > 0.1)
        
        # 이동 가능 여부 (어두운 배경 + 캐릭터 있을 법한 상황)
        movement_possible = not menu_detected and not text_detected and brightness < 150
        
        # 화면 타입 결정
        screen_type = 'field'  # 기본값
        
        if text_detected:
            screen_type = 'dialogue'
        elif menu_detected:
            screen_type = 'menu'
        elif color_profile['red'] > 0.08:
            screen_type = 'battle'
        elif brightness < 50:
            screen_type = 'dark'
        elif movement_possible:
            screen_type = 'field'
        
        return GameState(
            screen_hash=screen_hash,
            screen_type=screen_type,
            brightness=brightness,
            color_profile=color_profile,
            movement_possible=movement_possible,
            menu_detected=menu_detected,
            text_detected=text_detected,
            timestamp=time.time()
        )

class GameRuleEngine:
    """게임 룰 엔진"""
    
    def __init__(self):
        # 영웅전설4 기본 룰 정의
        self.game_rules = {
            'movement_keys': ['up', 'down', 'left', 'right'],
            'action_keys': ['space', 'enter'],
            'cancel_keys': ['esc', 'x'],
            'special_keys': ['z']
        }
        
        # 상황별 권장 액션
        self.situation_actions = {
            'field': {
                'primary': ['up', 'down', 'left', 'right'],  # 이동 우선
                'secondary': ['space'],  # 조사
                'weights': {'up': 1.0, 'down': 1.0, 'left': 1.0, 'right': 1.0, 'space': 0.3}
            },
            'menu': {
                'primary': ['up', 'down'],  # 메뉴 네비게이션
                'secondary': ['enter', 'space'],  # 선택
                'cancel': ['esc'],
                'weights': {'up': 1.0, 'down': 1.0, 'enter': 0.8, 'space': 0.8, 'esc': 0.2}
            },
            'dialogue': {
                'primary': ['enter', 'space'],  # 대화 진행
                'secondary': [],
                'weights': {'enter': 1.0, 'space': 1.0}
            },
            'battle': {
                'primary': ['enter', 'space'],  # 공격/스킬
                'secondary': ['up', 'down'],  # 메뉴 선택
                'weights': {'enter': 1.0, 'space': 0.8, 'up': 0.6, 'down': 0.6}
            },
            'dark': {
                'primary': ['up', 'down', 'left', 'right'],  # 탐색
                'secondary': ['space'],  # 조사
                'weights': {'up': 0.8, 'down': 0.8, 'left': 0.8, 'right': 0.8, 'space': 0.6}
            }
        }
    
    def get_recommended_actions(self, screen_type: str) -> List[str]:
        """상황별 권장 액션"""
        if screen_type in self.situation_actions:
            actions = self.situation_actions[screen_type]
            return actions['primary'] + actions['secondary']
        return ['up', 'down', 'left', 'right', 'space', 'enter']
    
    def get_action_weight(self, screen_type: str, action: str) -> float:
        """액션 가중치"""
        if screen_type in self.situation_actions:
            weights = self.situation_actions[screen_type].get('weights', {})
            return weights.get(action, 0.1)
        return 0.5

class IntelligentAI:
    """지능형 게임 AI"""
    
    def __init__(self):
        self.controller = DOSBoxController()
        self.vision = SmartVision()
        self.rules = GameRuleEngine()
        
        # 학습 메모리
        self.state_history = deque(maxlen=50)
        self.action_results = defaultdict(list)
        self.exploration_map = set()  # 방문한 화면들
        self.stuck_counter = 0
        
        # 탐험 전략
        self.exploration_mode = True
        self.current_direction = None
        self.direction_steps = 0
        self.max_direction_steps = random.randint(3, 8)
        
    def initialize(self):
        """초기화"""
        if not self.controller.find_dosbox():
            return False
        
        self.vision.setup_capture(self.controller.dosbox_hwnd)
        
        print("🎮 영웅전설4 AI 초기화 완료")
        print("📋 게임 룰:")
        print("  - 이동: 방향키 (↑↓←→)")  
        print("  - 선택: 엔터, 스페이스")
        print("  - 취소: ESC")
        return True
    
    def calculate_exploration_reward(self, state: GameState) -> float:
        """탐험 보상 계산"""
        reward = 0.0
        
        # 새로운 화면 발견 보상
        if state.screen_hash not in self.exploration_map:
            self.exploration_map.add(state.screen_hash)
            reward += 5.0
            print(f"🗺️ 새 지역 발견! 총 {len(self.exploration_map)}곳 탐험")
        
        # 화면 타입 변화 보상
        if len(self.state_history) > 0:
            prev_state = self.state_history[-1]
            if prev_state.screen_type != state.screen_type:
                reward += 3.0
                print(f"🔄 상황 변화: {prev_state.screen_type} → {state.screen_type}")
        
        # 밝기 변화 보상 (뭔가 일어남)
        if len(self.state_history) > 0:
            prev_state = self.state_history[-1]
            brightness_change = abs(state.brightness - prev_state.brightness)
            if brightness_change > 20:
                reward += brightness_change * 0.1
        
        return reward
    
    def choose_smart_action(self, state: GameState) -> str:
        """지능적 액션 선택"""
        # 상황별 권장 액션
        recommended_actions = self.rules.get_recommended_actions(state.screen_type)
        
        # 막힘 상황 감지
        if len(self.state_history) >= 5:
            recent_hashes = [s.screen_hash for s in list(self.state_history)[-5:]]
            if len(set(recent_hashes)) <= 2:  # 같은 화면만 반복
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        # 탐험 전략 적용
        if state.screen_type == 'field' and self.exploration_mode:
            return self.choose_exploration_action(state, recommended_actions)
        
        # 상황별 액션 선택
        if state.screen_type == 'dialogue' or state.text_detected:
            # 대화는 무조건 진행
            return random.choice(['enter', 'space'])
        
        elif state.screen_type == 'menu' or state.menu_detected:
            # 메뉴에서는 네비게이션 후 선택
            if self.stuck_counter > 3:
                return 'esc'  # 메뉴 탈출
            return random.choices(
                ['up', 'down', 'enter', 'space'], 
                weights=[1.0, 1.0, 0.8, 0.6]
            )[0]
        
        elif state.screen_type == 'battle':
            # 전투에서는 공격 우선
            return random.choices(
                ['enter', 'space', 'up', 'down'],
                weights=[1.0, 0.8, 0.4, 0.4]
            )[0]
        
        else:
            # 필드에서는 탐험
            return self.choose_exploration_action(state, recommended_actions)
    
    def choose_exploration_action(self, state: GameState, recommended_actions: List[str]) -> str:
        """탐험 액션 선택"""
        movement_actions = ['up', 'down', 'left', 'right']
        
        # 막혔을 때 방향 전환
        if self.stuck_counter > 5:
            self.current_direction = None
            self.direction_steps = 0
            self.stuck_counter = 0
            print("🔄 탐험 전략 변경")
        
        # 방향 유지 전략
        if self.current_direction and self.direction_steps < self.max_direction_steps:
            if self.current_direction in recommended_actions:
                self.direction_steps += 1
                return self.current_direction
        
        # 새 방향 선택
        available_movements = [a for a in movement_actions if a in recommended_actions]
        
        if available_movements:
            # 이전에 적게 사용한 방향 우선
            action_counts = {}
            for action in available_movements:
                action_counts[action] = len(self.action_results.get(action, []))
            
            # 가장 적게 사용한 방향들
            min_count = min(action_counts.values()) if action_counts else 0
            preferred_actions = [a for a, c in action_counts.items() if c <= min_count + 2]
            
            self.current_direction = random.choice(preferred_actions)
            self.direction_steps = 1
            self.max_direction_steps = random.randint(3, 8)
            
            print(f"🎯 새 탐험 방향: {self.current_direction} ({self.max_direction_steps}스텝)")
            return self.current_direction
        
        # 이동할 수 없으면 조사
        return 'space'
    
    def learn_from_action(self, prev_state: GameState, action: str, new_state: GameState, success: bool):
        """액션 결과로부터 학습"""
        reward = self.calculate_exploration_reward(new_state) if success else -0.5
        
        # 액션 결과 저장
        self.action_results[action].append({
            'reward': reward,
            'success': success,
            'context': prev_state.screen_type,
            'timestamp': time.time()
        })
        
        # 최근 결과만 유지
        if len(self.action_results[action]) > 30:
            self.action_results[action].pop(0)
    
    async def play_cycle(self) -> Dict:
        """게임 플레이 사이클"""
        try:
            # 1. 화면 분석
            image = self.vision.capture_game_screen()
            current_state = self.vision.analyze_game_screen(image)
            
            # 2. 액션 선택
            action = self.choose_smart_action(current_state)
            
            # 3. 액션 실행
            success = self.controller.send_key_to_dosbox(action, hold_time=0.08)
            
            # 4. 결과 대기
            await asyncio.sleep(0.25)  # 게임 반응 시간
            
            # 5. 학습
            if len(self.state_history) > 0:
                prev_state = self.state_history[-1]
                self.learn_from_action(prev_state, action, current_state, success)
            
            # 6. 상태 기록
            self.state_history.append(current_state)
            
            return {
                'success': success,
                'action': action,
                'state_type': current_state.screen_type,
                'hash': current_state.screen_hash,
                'explored_areas': len(self.exploration_map),
                'stuck_counter': self.stuck_counter
            }
            
        except Exception as e:
            print(f"❌ 플레이 사이클 오류: {e}")
            return {'success': False, 'error': str(e)}

async def main():
    """메인 실행"""
    print("🎮 영웅전설4 스마트 AI")
    print("=" * 40)
    
    ai = IntelligentAI()
    
    # 초기화
    if not ai.initialize():
        return
    
    print("\n🚀 게임 플레이 시작!")
    print("🧠 규칙 기반 학습 + 지능형 탐험")
    
    total_cycles = 200
    success_count = 0
    
    for cycle in range(1, total_cycles + 1):
        result = await ai.play_cycle()
        
        if result['success']:
            success_count += 1
            status = "✅"
        else:
            status = "❌"
        
        # 5사이클마다 리포트
        if cycle % 5 == 0:
            success_rate = success_count / cycle
            print(f"{status} #{cycle:3d} | {result.get('action', '?'):5s} | "
                  f"{result.get('state_type', '?'):9s} | "
                  f"성공률:{success_rate:.2f} | "
                  f"탐험:{result.get('explored_areas', 0):2d}곳")
        
        # 20사이클마다 상세 리포트
        if cycle % 20 == 0:
            print(f"📊 탐험 현황: {len(ai.exploration_map)}곳 발견")
            
            # 액션별 성공률
            action_stats = {}
            for action, results in ai.action_results.items():
                if results:
                    recent_results = results[-10:]
                    success_rate = sum(1 for r in recent_results if r['success']) / len(recent_results)
                    action_stats[action] = success_rate
            
            if action_stats:
                best_action = max(action_stats.items(), key=lambda x: x[1])
                print(f"🎯 최고 성능 액션: {best_action[0]} ({best_action[1]:.2f})")
    
    # 최종 결과
    print(f"\n🏁 게임 플레이 완료!")
    print(f"총 사이클: {total_cycles}")
    print(f"성공률: {success_count/total_cycles:.2f}")
    print(f"탐험 지역: {len(ai.exploration_map)}곳")
    
    # 액션별 최종 통계
    print(f"\n📈 액션 성능 순위:")
    final_stats = {}
    for action, results in ai.action_results.items():
        if results:
            success_rate = sum(1 for r in results if r['success']) / len(results)
            avg_reward = sum(r['reward'] for r in results) / len(results)
            final_stats[action] = {'success_rate': success_rate, 'avg_reward': avg_reward, 'count': len(results)}
    
    for action, stats in sorted(final_stats.items(), key=lambda x: x[1]['avg_reward'], reverse=True):
        print(f"  {action:5s}: 성공률 {stats['success_rate']:.2f}, 평균보상 {stats['avg_reward']:.1f} ({stats['count']}회)")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()