#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초고속 자율학습 영웅전설4 AI - 패턴 인식 및 자가 진화
"""

import asyncio
import time
import random
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import pickle

# 게임 제어
import cv2
import pyautogui
import win32gui
import win32con
import win32api

@dataclass
class GameState:
    """게임 상태 정보"""
    screen_hash: str
    screen_type: str
    brightness: float
    color_ratios: Dict[str, float]
    timestamp: float
    
@dataclass
class ActionResult:
    """액션 결과"""
    action_name: str
    before_state: GameState
    after_state: GameState
    success: bool
    reward: float
    time_taken: float

@dataclass 
class LearnedPattern:
    """학습된 패턴"""
    state_pattern: str
    best_action: str
    success_rate: float
    total_tries: int
    last_updated: float
    confidence: float

class HyperSpeedVision:
    """초고속 비전 시스템"""
    
    def __init__(self):
        self.hwnd_cache = None
        self.rect_cache = None
        self.last_hash = None
        
    def get_game_window(self):
        """게임 윈도우 찾기 (캐시됨)"""
        if self.hwnd_cache is None:
            def enum_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if any(keyword in title for keyword in ["DOSBox", "dosbox", "ED4"]):
                        windows.append(hwnd)
                return True
                
            windows = []
            win32gui.EnumWindows(enum_callback, windows)
            self.hwnd_cache = windows[0] if windows else None
            
        return self.hwnd_cache
    
    def ultra_fast_capture(self) -> Optional[np.ndarray]:
        """초고속 화면 캡처 (0.05초 이내)"""
        try:
            if self.rect_cache is None:
                hwnd = self.get_game_window()
                if not hwnd:
                    return None
                self.rect_cache = win32gui.GetWindowRect(hwnd)
            
            x, y, x2, y2 = self.rect_cache
            # 더 작은 영역만 캡처 (속도 향상)
            w, h = x2-x, y2-y
            capture_w, capture_h = min(600, w), min(400, h)
            
            screenshot = pyautogui.screenshot(region=(x, y, capture_w, capture_h))
            return np.array(screenshot)
            
        except Exception:
            self.rect_cache = None
            return None
    
    def lightning_analyze(self, image: np.ndarray) -> GameState:
        """번개 속도 분석 (0.02초 이내)"""
        if image is None:
            return GameState("", "unknown", 0, {}, time.time())
        
        # 극도로 축소 (16x16 픽셀로!)
        tiny = cv2.resize(image, (16, 16))
        gray_tiny = cv2.cvtColor(tiny, cv2.COLOR_RGB2GRAY)
        
        # 해시 생성 (상태 식별용)
        screen_hash = hashlib.md5(gray_tiny.tobytes()).hexdigest()[:8]
        
        # 초고속 색상 분석
        hsv_tiny = cv2.cvtColor(tiny, cv2.COLOR_RGB2HSV)
        
        # 평균값 기반 빠른 분석
        brightness = float(np.mean(gray_tiny))
        
        # 색상 비율 (4x4로 더 축소)
        micro = hsv_tiny[::4, ::4]  # 4x4 픽셀만 사용
        
        blue_count = np.sum((micro[:,:,0] >= 100) & (micro[:,:,0] <= 130))
        red_count = np.sum((micro[:,:,0] >= 170) | (micro[:,:,0] <= 10))
        green_count = np.sum((micro[:,:,0] >= 40) & (micro[:,:,0] <= 80))
        
        total_pixels = micro.shape[0] * micro.shape[1]
        
        color_ratios = {
            'blue': blue_count / total_pixels,
            'red': red_count / total_pixels, 
            'green': green_count / total_pixels,
        }
        
        # 화면 타입 추론
        screen_type = 'field'
        if color_ratios['blue'] > 0.25:
            screen_type = 'menu'
        elif color_ratios['red'] > 0.2:
            screen_type = 'battle'
        elif brightness > 180:
            screen_type = 'dialogue'
        elif brightness < 50:
            screen_type = 'dark'
        
        return GameState(
            screen_hash=screen_hash,
            screen_type=screen_type,
            brightness=brightness,
            color_ratios=color_ratios,
            timestamp=time.time()
        )

class LightningController:
    """번개 속도 컨트롤러"""
    
    def __init__(self):
        self.hwnd_cache = None
        self.last_input_time = 0
        
    def instant_key(self, key: str) -> bool:
        """즉시 키 입력 (0.02초 이내)"""
        current_time = time.time()
        if current_time - self.last_input_time < 0.08:  # 80ms 제한
            return False
            
        try:
            if self.hwnd_cache is None:
                def enum_callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if any(keyword in title for keyword in ["DOSBox", "dosbox", "ED4"]):
                            windows.append(hwnd)
                    return True
                    
                windows = []
                win32gui.EnumWindows(enum_callback, windows)
                self.hwnd_cache = windows[0] if windows else None
            
            if not self.hwnd_cache:
                return False
            
            # 키 코드 매핑
            key_codes = {
                'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
                'enter': 0x0D, 'space': 0x20, 'esc': 0x1B,
                'z': 0x5A, 'x': 0x58, 'a': 0x41, 's': 0x53, 'c': 0x43
            }
            
            if key.lower() not in key_codes:
                return False
            
            vk = key_codes[key.lower()]
            
            # 초고속 키 입력 (25ms만 홀드)
            win32api.keybd_event(vk, 0, 0, 0)
            time.sleep(0.025)
            win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
            
            self.last_input_time = current_time
            return True
            
        except Exception:
            return False

class SelfLearningBrain:
    """자기학습 뇌"""
    
    def __init__(self):
        # 메모리 데이터베이스 (SQLite)
        self.conn = sqlite3.connect(':memory:')
        self.init_memory()
        
        # 실시간 학습 데이터
        self.state_action_history = deque(maxlen=1000)  # 최근 1000개 기억
        self.pattern_memory = {}  # 패턴 -> 최적 액션
        self.success_memory = defaultdict(list)  # 액션별 성공 기록
        self.reward_system = {}  # 보상 시스템
        
        # 학습 파라미터
        self.exploration_rate = 0.3  # 탐험 vs 활용
        self.learning_rate = 0.1
        self.memory_decay = 0.95
        
        # 가능한 액션들
        self.action_space = [
            'up', 'down', 'left', 'right', 
            'enter', 'space', 'esc', 'z', 'x', 'a', 's', 'c'
        ]
        
    def init_memory(self):
        """메모리 초기화"""
        cursor = self.conn.cursor()
        
        # 상태-액션-보상 테이블
        cursor.execute('''
            CREATE TABLE experiences (
                id INTEGER PRIMARY KEY,
                state_hash TEXT,
                action TEXT,
                reward REAL,
                next_state_hash TEXT,
                timestamp REAL
            )
        ''')
        
        # 패턴 학습 테이블
        cursor.execute('''
            CREATE TABLE patterns (
                state_pattern TEXT PRIMARY KEY,
                best_action TEXT,
                success_rate REAL,
                total_tries INTEGER,
                confidence REAL,
                last_updated REAL
            )
        ''')
        
        self.conn.commit()
    
    def calculate_reward(self, before_state: GameState, after_state: GameState, action: str) -> float:
        """보상 계산"""
        reward = 0.0
        
        # 화면 변화 보상 (새로운 상황 발견)
        if before_state.screen_hash != after_state.screen_hash:
            reward += 2.0
        
        # 화면 타입 변화 보상
        if before_state.screen_type != after_state.screen_type:
            reward += 5.0
        
        # 밝기 변화 보상 (뭔가 일어남)
        brightness_change = abs(after_state.brightness - before_state.brightness)
        if brightness_change > 10:
            reward += brightness_change * 0.1
        
        # 색상 변화 보상
        for color in ['blue', 'red', 'green']:
            color_change = abs(after_state.color_ratios.get(color, 0) - 
                              before_state.color_ratios.get(color, 0))
            if color_change > 0.1:
                reward += color_change * 3
        
        # 특정 상황별 보상
        if after_state.screen_type == 'menu' and action in ['up', 'down', 'enter']:
            reward += 1.0
        elif after_state.screen_type == 'battle' and action in ['enter', 'z', 'a']:
            reward += 1.5
        elif after_state.screen_type == 'dialogue' and action in ['enter', 'space']:
            reward += 1.0
        
        # 탐험 보상 (새로운 상태)
        if after_state.screen_hash not in [exp[1].screen_hash for exp in self.state_action_history]:
            reward += 3.0
        
        return reward
    
    def learn_from_experience(self, before_state: GameState, action: str, after_state: GameState):
        """경험으로부터 학습"""
        # 보상 계산
        reward = self.calculate_reward(before_state, after_state, action)
        
        # 경험 저장
        experience = (before_state, after_state, action, reward)
        self.state_action_history.append(experience)
        
        # 데이터베이스에 저장
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO experiences (state_hash, action, reward, next_state_hash, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (before_state.screen_hash, action, reward, after_state.screen_hash, time.time()))
        
        # 패턴 학습
        self.update_patterns(before_state, action, reward)
        
        # 성공 기록 업데이트
        success = reward > 1.0
        self.success_memory[action].append(success)
        if len(self.success_memory[action]) > 50:
            self.success_memory[action].pop(0)
        
        self.conn.commit()
    
    def update_patterns(self, state: GameState, action: str, reward: float):
        """패턴 업데이트"""
        # 상태 패턴 생성 (단순화된 상태 표현)
        pattern = f"{state.screen_type}_{int(state.brightness/50)}_{int(state.color_ratios.get('blue', 0)*10)}"
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM patterns WHERE state_pattern = ?', (pattern,))
        existing = cursor.fetchone()
        
        if existing:
            # 기존 패턴 업데이트
            _, best_action, success_rate, total_tries, confidence, _ = existing
            
            new_total = total_tries + 1
            new_success_rate = (success_rate * total_tries + (1 if reward > 1 else 0)) / new_total
            
            # 더 좋은 액션이면 업데이트
            if reward > 1 and (new_success_rate > success_rate or best_action != action):
                best_action = action
            
            new_confidence = min(0.99, confidence + 0.01)
            
            cursor.execute('''
                UPDATE patterns 
                SET best_action=?, success_rate=?, total_tries=?, confidence=?, last_updated=?
                WHERE state_pattern=?
            ''', (best_action, new_success_rate, new_total, new_confidence, time.time(), pattern))
            
        else:
            # 새 패턴 추가
            cursor.execute('''
                INSERT INTO patterns (state_pattern, best_action, success_rate, total_tries, confidence, last_updated)
                VALUES (?, ?, ?, 1, 0.1, ?)
            ''', (pattern, action, 1 if reward > 1 else 0, time.time()))
    
    def choose_intelligent_action(self, current_state: GameState) -> str:
        """지능적 액션 선택"""
        # 패턴 매칭으로 최적 액션 찾기
        pattern = f"{current_state.screen_type}_{int(current_state.brightness/50)}_{int(current_state.color_ratios.get('blue', 0)*10)}"
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT best_action, confidence FROM patterns WHERE state_pattern = ?', (pattern,))
        learned_action = cursor.fetchone()
        
        # 학습된 패턴이 있고 신뢰도가 높으면 사용
        if learned_action and learned_action[1] > 0.6:
            if random.random() > self.exploration_rate:  # 활용
                return learned_action[0]
        
        # 탐험 모드 - 성공률 기반 선택
        action_scores = {}
        for action in self.action_space:
            if action in self.success_memory and self.success_memory[action]:
                recent_successes = self.success_memory[action][-10:]  # 최근 10개
                success_rate = sum(recent_successes) / len(recent_successes)
                action_scores[action] = success_rate
            else:
                action_scores[action] = 0.5  # 기본값
        
        # 상황별 보정
        if current_state.screen_type == 'menu':
            for action in ['up', 'down', 'enter']:
                action_scores[action] *= 1.5
        elif current_state.screen_type == 'battle':
            for action in ['enter', 'z', 'a', 's']:
                action_scores[action] *= 1.3
        elif current_state.screen_type == 'dialogue':
            for action in ['enter', 'space']:
                action_scores[action] *= 2.0
        elif current_state.screen_type == 'field':
            for action in ['up', 'down', 'left', 'right', 'space']:
                action_scores[action] *= 1.2
        
        # 가중치 기반 선택
        actions = list(action_scores.keys())
        weights = list(action_scores.values())
        
        return random.choices(actions, weights=weights)[0]
    
    def get_learning_stats(self) -> Dict:
        """학습 통계"""
        cursor = self.conn.cursor()
        
        # 총 경험 수
        cursor.execute('SELECT COUNT(*) FROM experiences')
        total_experiences = cursor.fetchone()[0]
        
        # 학습된 패턴 수
        cursor.execute('SELECT COUNT(*) FROM patterns')
        learned_patterns = cursor.fetchone()[0]
        
        # 평균 보상
        cursor.execute('SELECT AVG(reward) FROM experiences')
        avg_reward = cursor.fetchone()[0] or 0
        
        # 액션별 성공률
        action_success_rates = {}
        for action in self.action_space:
            if action in self.success_memory and self.success_memory[action]:
                success_rate = sum(self.success_memory[action]) / len(self.success_memory[action])
                action_success_rates[action] = success_rate
        
        return {
            'total_experiences': total_experiences,
            'learned_patterns': learned_patterns,
            'avg_reward': avg_reward,
            'exploration_rate': self.exploration_rate,
            'best_actions': sorted(action_success_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        }

class HyperIntelligentAI:
    """초지능 AI"""
    
    def __init__(self):
        self.vision = HyperSpeedVision()
        self.controller = LightningController()
        self.brain = SelfLearningBrain()
        
        # 상태 추적
        self.last_state = None
        self.action_count = 0
        self.learning_enabled = True
        
    async def hyper_fast_cycle(self) -> Dict:
        """초고속 사이클 (0.15초 목표)"""
        cycle_start = time.time()
        
        try:
            # 1. 초고속 상태 인식 (0.05초)
            image = self.vision.ultra_fast_capture()
            current_state = self.vision.lightning_analyze(image)
            
            # 2. 이전 액션으로부터 학습 (0.02초)
            if self.last_state and self.learning_enabled:
                last_action = getattr(self, '_last_action', None)
                if last_action:
                    self.brain.learn_from_experience(self.last_state, last_action, current_state)
            
            # 3. 지능적 액션 선택 (0.03초)
            action = self.brain.choose_intelligent_action(current_state)
            
            # 4. 즉시 실행 (0.03초)
            success = self.controller.instant_key(action)
            
            # 5. 상태 업데이트
            self.last_state = current_state
            self._last_action = action
            self.action_count += 1
            
            cycle_time = time.time() - cycle_start
            
            return {
                'success': success,
                'action': action,
                'state': current_state.screen_type,
                'hash': current_state.screen_hash,
                'cycle_time': cycle_time,
                'learning_progress': self.action_count
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

async def main():
    """초고속 자율학습 메인"""
    print("🧠⚡ 초고속 자율학습 영웅전설4 AI")
    print("=" * 50)
    
    ai = HyperIntelligentAI()
    
    # 게임 연결 확인
    if ai.vision.ultra_fast_capture() is None:
        print("❌ 게임을 찾을 수 없습니다!")
        return
    
    print("🎮 게임 연결 완료! 자율학습 시작...")
    print("⚡ 목표 사이클 시간: 0.15초 (6.7 cps)")
    print("🧠 실시간 패턴 학습 활성화")
    
    # 초고속 학습 루프
    total_cycles = 400  # 1분 = 400사이클
    start_time = time.time()
    success_count = 0
    
    last_stats_time = start_time
    
    for cycle in range(1, total_cycles + 1):
        cycle_start_time = time.time()
        
        # 초고속 사이클 실행
        result = await ai.hyper_fast_cycle()
        
        if result['success']:
            success_count += 1
        
        # 10사이클마다 간단 리포트
        if cycle % 10 == 0:
            elapsed = time.time() - start_time
            cps = cycle / elapsed
            success_rate = success_count / cycle
            
            print(f"⚡ #{cycle:3d} | {result.get('action', '?'):5s} | "
                  f"{result.get('state', '?'):8s} | "
                  f"성공:{success_rate:.2f} | "
                  f"속도:{cps:.1f}cps | "
                  f"학습진행:{result.get('learning_progress', 0)}")
        
        # 50사이클마다 학습 통계
        if cycle % 50 == 0:
            stats = ai.brain.get_learning_stats()
            print(f"📊 학습통계: 경험{stats['total_experiences']} | "
                  f"패턴{stats['learned_patterns']} | "
                  f"평균보상{stats['avg_reward']:.1f}")
            
            if stats['best_actions']:
                best_action, best_rate = stats['best_actions'][0]
                print(f"🎯 최고액션: {best_action} ({best_rate:.2f})")
        
        # 0.15초 사이클 유지 (가능한 경우)
        cycle_elapsed = time.time() - cycle_start_time
        if cycle_elapsed < 0.15:
            await asyncio.sleep(0.15 - cycle_elapsed)
    
    # 최종 결과
    total_time = time.time() - start_time
    final_stats = ai.brain.get_learning_stats()
    
    print(f"\n🧠 자율학습 완료!")
    print(f"총 사이클: {total_cycles}")
    print(f"성공률: {success_count/total_cycles:.2f}")
    print(f"평균속도: {total_cycles/total_time:.1f} cps")
    print(f"학습된 패턴: {final_stats['learned_patterns']}개")
    print(f"총 경험: {final_stats['total_experiences']}개")
    print(f"평균 보상: {final_stats['avg_reward']:.2f}")
    
    print(f"\n🏆 학습된 최고 액션들:")
    for action, rate in final_stats['best_actions']:
        print(f"  {action}: {rate:.2f}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 학습 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()