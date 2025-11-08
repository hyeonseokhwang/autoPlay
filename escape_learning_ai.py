#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
메뉴 탈출 + 실제 학습하는 영웅전설4 AI
"""

import asyncio
import time
import random
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib

# 게임 제어
import cv2
import pyautogui
import win32gui
import win32con
import win32api

@dataclass
class ScreenState:
    """화면 상태 정보"""
    hash_id: str
    screen_type: str
    brightness: float
    color_ratios: Dict[str, float]
    timestamp: float
    is_menu: bool
    is_stuck: bool

class SmartScreenAnalyzer:
    """스마트 화면 분석기"""
    
    def __init__(self):
        self.dosbox_hwnd = None
        self.capture_region = None
        self.screen_memory = {}  # 해시별 방문 횟수
        self.last_analysis = None
        
    def setup(self):
        """초기 설정"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if any(keyword in title.lower() for keyword in ["dosbox", "ed4"]):
                    windows.append((hwnd, title))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if not windows:
            return False
        
        self.dosbox_hwnd, title = windows[0]
        print(f"✅ DOSBox 연결: {title}")
        
        # 캡처 영역 최적화
        rect = win32gui.GetWindowRect(self.dosbox_hwnd)
        x, y, x2, y2 = rect
        self.capture_region = (x + 8, y + 30, x2 - x - 16, y2 - y - 38)
        return True
    
    def analyze_screen(self) -> Optional[ScreenState]:
        """화면 분석"""
        try:
            # 캡처
            screenshot = pyautogui.screenshot(region=self.capture_region)
            image = np.array(screenshot)
            
            # 전처리 (속도 최적화)
            if image.shape[1] > 500:
                scale = 500 / image.shape[1]
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # 색상 분석
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 해시 생성
            tiny = cv2.resize(gray, (20, 15))
            hash_id = hashlib.md5(tiny.tobytes()).hexdigest()[:10]
            
            # 밝기
            brightness = np.mean(gray)
            
            # 색상 비율
            total_pixels = image.shape[0] * image.shape[1]
            
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
            green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            
            color_ratios = {
                'blue': np.sum(blue_mask > 0) / total_pixels,
                'red': (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / total_pixels,
                'green': np.sum(green_mask > 0) / total_pixels,
            }
            
            # 텍스트/에지 분석
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / total_pixels
            
            bright_mask = gray > 180
            bright_ratio = np.sum(bright_mask) / total_pixels
            
            # 화면 타입 분류 (더 정확하게)
            screen_type = self._classify_screen_type(
                brightness, color_ratios, bright_ratio, edge_ratio
            )
            
            # 메뉴 감지 (중요!)
            is_menu = self._detect_menu(brightness, color_ratios, edge_ratio)
            
            # 방문 기록
            self.screen_memory[hash_id] = self.screen_memory.get(hash_id, 0) + 1
            
            # 막힘 감지
            is_stuck = self.screen_memory[hash_id] > 5  # 같은 화면 5번 이상
            
            state = ScreenState(
                hash_id=hash_id,
                screen_type=screen_type,
                brightness=brightness,
                color_ratios=color_ratios,
                timestamp=time.time(),
                is_menu=is_menu,
                is_stuck=is_stuck
            )
            
            self.last_analysis = state
            return state
            
        except Exception as e:
            print(f"❌ 화면 분석 실패: {e}")
            return None
    
    def _classify_screen_type(self, brightness, color_ratios, bright_ratio, edge_ratio):
        """화면 타입 분류"""
        # 더 정확한 분류 로직
        if bright_ratio > 0.15 and edge_ratio > 0.05:
            return 'dialogue'
        elif color_ratios['blue'] > 0.08 or (brightness > 60 and brightness < 90):
            return 'menu'
        elif color_ratios['red'] > 0.06:
            return 'battle'
        elif brightness < 40:
            return 'dark_field'
        elif brightness > 100:
            return 'bright_field'
        else:
            return 'normal_field'
    
    def _detect_menu(self, brightness, color_ratios, edge_ratio):
        """메뉴 상태 감지"""
        # 메뉴 특성: 중간 밝기 + 파란색 요소 + 많은 에지
        return (60 < brightness < 90 and 
                (color_ratios['blue'] > 0.03 or edge_ratio > 0.25))

class EscapeController:
    """탈출 전용 컨트롤러"""
    
    def __init__(self):
        self.dosbox_hwnd = None
        self.last_input = 0
        
    def setup(self, hwnd):
        """설정"""
        self.dosbox_hwnd = hwnd
    
    def send_key(self, key: str, force: bool = False) -> bool:
        """키 전송 (안전하게)"""
        current_time = time.time()
        if not force and current_time - self.last_input < 0.12:
            return False
        
        try:
            if not self.dosbox_hwnd:
                return False
            
            win32gui.SetForegroundWindow(self.dosbox_hwnd)
            time.sleep(0.02)
            
            key_map = {
                'up': win32con.VK_UP, 'down': win32con.VK_DOWN,
                'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT,
                'enter': win32con.VK_RETURN, 'space': win32con.VK_SPACE,
                'esc': win32con.VK_ESCAPE, 'z': ord('Z'), 'x': ord('X')
            }
            
            if key.lower() not in key_map:
                return False
            
            vk = key_map[key.lower()]
            
            # 키 입력
            win32api.keybd_event(vk, 0, 0, 0)
            time.sleep(0.06)
            win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
            
            self.last_input = current_time
            return True
            
        except Exception as e:
            print(f"❌ 키 전송 실패: {e}")
            return False

class LearningBrain:
    """학습 뇌 - 실제 패턴 인식"""
    
    def __init__(self):
        # 메모리 초기화
        self.conn = sqlite3.connect(':memory:')
        self._init_db()
        
        # 상태-액션 경험
        self.state_action_results = defaultdict(list)
        self.successful_sequences = []
        self.escape_strategies = []
        
        # 학습 파라미터
        self.exploration_rate = 0.4
        self.escape_attempts = 0
        
    def _init_db(self):
        """DB 초기화"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE experiences (
                id INTEGER PRIMARY KEY,
                screen_hash TEXT,
                screen_type TEXT,
                action TEXT,
                reward REAL,
                success INTEGER,
                timestamp REAL
            )
        ''')
        self.conn.commit()
    
    def learn_from_action(self, before_state: ScreenState, action: str, 
                         after_state: ScreenState, success: bool):
        """액션 결과 학습"""
        # 보상 계산
        reward = self._calculate_reward(before_state, after_state, action, success)
        
        # 경험 저장
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO experiences 
            (screen_hash, screen_type, action, reward, success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (before_state.hash_id, before_state.screen_type, action, 
              reward, 1 if success else 0, time.time()))
        
        # 메모리에도 저장
        key = f"{before_state.screen_type}_{action}"
        self.state_action_results[key].append({
            'reward': reward,
            'success': success,
            'before_hash': before_state.hash_id,
            'after_hash': after_state.hash_id if after_state else None,
            'timestamp': time.time()
        })
        
        # 성공적 탈출 시퀀스 기록
        if before_state.is_menu and after_state and not after_state.is_menu:
            self.escape_strategies.append(action)
            print(f"🎯 메뉴 탈출 성공: {action}")
        
        self.conn.commit()
    
    def _calculate_reward(self, before: ScreenState, after: Optional[ScreenState], 
                         action: str, success: bool) -> float:
        """보상 계산"""
        if not success or not after:
            return -1.0
        
        reward = 0.0
        
        # 새로운 화면 발견 보상
        if after.hash_id != before.hash_id:
            reward += 3.0
        
        # 메뉴 탈출 보상 (중요!)
        if before.is_menu and not after.is_menu:
            reward += 10.0
            print(f"🚀 메뉴 탈출! +10점")
        
        # 필드 진입 보상
        if 'field' in after.screen_type and 'field' not in before.screen_type:
            reward += 5.0
        
        # 화면 타입 변화 보상
        if before.screen_type != after.screen_type:
            reward += 2.0
        
        # 막힌 상태 탈출 보상
        if before.is_stuck and not after.is_stuck:
            reward += 4.0
        
        # 탐험 보상 (새로운 곳)
        if after.hash_id not in [before.hash_id]:
            reward += 1.0
        
        return reward
    
    def choose_smart_action(self, current_state: ScreenState) -> str:
        """지능적 액션 선택"""
        # 메뉴에서 탈출 전략 (최우선!)
        if current_state.is_menu:
            return self._choose_escape_action(current_state)
        
        # 막힌 상황 감지
        if current_state.is_stuck:
            return self._choose_unstuck_action(current_state)
        
        # 일반 탐험
        return self._choose_exploration_action(current_state)
    
    def _choose_escape_action(self, state: ScreenState) -> str:
        """메뉴 탈출 액션"""
        self.escape_attempts += 1
        
        # 이전 성공 전략 사용
        if self.escape_strategies:
            best_escape = max(set(self.escape_strategies), 
                             key=self.escape_strategies.count)
            if random.random() < 0.7:  # 70% 확률로 검증된 전략 사용
                print(f"🔄 검증된 탈출 전략: {best_escape}")
                return best_escape
        
        # ESC 키 우선 시도
        if self.escape_attempts % 3 == 1:
            return 'esc'
        
        # 메뉴 네비게이션
        return random.choice(['esc', 'x', 'enter', 'space'])
    
    def _choose_unstuck_action(self, state: ScreenState) -> str:
        """막힘 해결 액션"""
        # 다양한 키 시도
        return random.choice(['esc', 'x', 'space', 'enter', 'up', 'down'])
    
    def _choose_exploration_action(self, state: ScreenState) -> str:
        """탐험 액션"""
        # 화면 타입별 최적 액션
        if 'dialogue' in state.screen_type:
            return random.choice(['enter', 'space'])
        
        elif 'battle' in state.screen_type:
            return random.choice(['enter', 'space', 'up', 'down'])
        
        elif 'field' in state.screen_type:
            # 필드에서는 이동 위주
            return random.choice(['up', 'down', 'left', 'right', 'space'])
        
        else:
            # 기본 탐험
            return random.choice(['up', 'down', 'left', 'right', 'space', 'enter'])
    
    def get_learning_stats(self) -> Dict:
        """학습 통계"""
        cursor = self.conn.cursor()
        
        # 총 경험
        cursor.execute('SELECT COUNT(*) FROM experiences')
        total_exp = cursor.fetchone()[0]
        
        # 평균 보상
        cursor.execute('SELECT AVG(reward) FROM experiences WHERE reward > 0')
        avg_reward = cursor.fetchone()[0] or 0
        
        # 성공률
        cursor.execute('SELECT AVG(success) FROM experiences')
        success_rate = cursor.fetchone()[0] or 0
        
        # 탈출 성공 횟수
        escape_count = len(self.escape_strategies)
        
        return {
            'total_experiences': total_exp,
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'escape_successes': escape_count,
            'best_escape_action': max(set(self.escape_strategies), 
                                    key=self.escape_strategies.count) if self.escape_strategies else None
        }

class SmartHeroAI:
    """스마트한 영웅전설4 AI"""
    
    def __init__(self):
        self.analyzer = SmartScreenAnalyzer()
        self.controller = EscapeController()
        self.brain = LearningBrain()
        
        self.last_state = None
        self.cycle_count = 0
        
    def initialize(self) -> bool:
        """초기화"""
        if not self.analyzer.setup():
            return False
        
        self.controller.setup(self.analyzer.dosbox_hwnd)
        
        print("🧠 스마트 영웅전설4 AI 초기화 완료")
        print("🎯 특별 기능:")
        print("  - 메뉴 자동 탈출")
        print("  - 실시간 패턴 학습")
        print("  - 막힘 상황 자동 해결")
        return True
    
    async def smart_cycle(self) -> Dict:
        """스마트 플레이 사이클"""
        try:
            # 1. 화면 분석
            current_state = self.analyzer.analyze_screen()
            if not current_state:
                return {'success': False, 'error': '화면 분석 실패'}
            
            # 2. 지능적 액션 선택
            action = self.brain.choose_smart_action(current_state)
            
            # 3. 액션 실행
            success = self.controller.send_key(action)
            
            # 4. 결과 대기
            await asyncio.sleep(0.3)
            
            # 5. 결과 분석
            result_state = self.analyzer.analyze_screen()
            
            # 6. 학습
            if self.last_state and result_state:
                self.brain.learn_from_action(self.last_state, action, result_state, success)
            
            # 7. 상태 업데이트
            self.last_state = current_state
            self.cycle_count += 1
            
            return {
                'success': success,
                'action': action,
                'before_type': current_state.screen_type,
                'after_type': result_state.screen_type if result_state else '?',
                'before_hash': current_state.hash_id,
                'after_hash': result_state.hash_id if result_state else '?',
                'is_menu': current_state.is_menu,
                'is_stuck': current_state.is_stuck,
                'cycle': self.cycle_count
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

async def main():
    """메인 실행"""
    print("🧠 스마트 학습 영웅전설4 AI")
    print("=" * 45)
    
    ai = SmartHeroAI()
    
    if not ai.initialize():
        print("❌ 초기화 실패!")
        return
    
    print("\n🚀 스마트 플레이 시작!")
    
    total_cycles = 100
    success_count = 0
    escape_count = 0
    
    for cycle in range(1, total_cycles + 1):
        result = await ai.smart_cycle()
        
        if result['success']:
            success_count += 1
            status = "✅"
        else:
            status = "❌"
        
        # 탈출 성공 감지
        if result.get('is_menu') and result.get('after_type', '') != 'menu':
            escape_count += 1
        
        # 3사이클마다 리포트
        if cycle % 3 == 0:
            success_rate = success_count / cycle
            print(f"{status} #{cycle:3d} | {result.get('action', '?'):5s} | "
                  f"{result.get('before_type', '?'):12s} → {result.get('after_type', '?'):12s} | "
                  f"성공률:{success_rate:.2f}")
            
            # 상태 정보
            if result.get('is_menu'):
                print(f"      🔴 메뉴 상태 감지!")
            if result.get('is_stuck'):
                print(f"      ⚠️ 막힌 상태 감지!")
        
        # 10사이클마다 학습 통계
        if cycle % 10 == 0:
            stats = ai.brain.get_learning_stats()
            print(f"📊 학습 통계:")
            print(f"    경험: {stats['total_experiences']}개")
            print(f"    평균 보상: {stats['avg_reward']:.1f}")
            print(f"    성공률: {stats['success_rate']:.2f}")
            print(f"    탈출 성공: {stats['escape_successes']}회")
            
            if stats['best_escape_action']:
                print(f"    🎯 최고 탈출법: {stats['best_escape_action']}")
            print()
    
    # 최종 결과
    final_stats = ai.brain.get_learning_stats()
    print(f"\n🏁 스마트 플레이 완료!")
    print(f"총 사이클: {total_cycles}")
    print(f"성공률: {success_count/total_cycles:.2f}")
    print(f"총 경험: {final_stats['total_experiences']}개")
    print(f"평균 보상: {final_stats['avg_reward']:.2f}")
    print(f"메뉴 탈출: {final_stats['escape_successes']}회")
    
    if final_stats['best_escape_action']:
        print(f"🏆 최고 탈출 전략: {final_stats['best_escape_action']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()