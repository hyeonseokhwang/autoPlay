#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 진정한 AI 학습 시스템
- 실제 강화학습 기반
- 경험 데이터 축적
- 자율적 판단과 추론
- 스스로 발전하는 AI
"""

import asyncio
import time
import random
import numpy as np
import cv2
import json
import sqlite3
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Any
from PIL import ImageGrab
import win32gui
import win32con
import win32api

class GameExperience:
    """게임 경험 데이터 클래스"""
    
    def __init__(self, state: Dict, action: str, reward: float, next_state: Dict, info: Dict = None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.timestamp = datetime.now()
        self.info = info or {}
        
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'timestamp': self.timestamp.isoformat(),
            'info': self.info
        }

class GameStateAnalyzer:
    """게임 상태 분석기"""
    
    def __init__(self):
        self.previous_states = deque(maxlen=10)
        self.state_features = {}
        
    def extract_features(self, screenshot: np.ndarray) -> Dict[str, float]:
        """스크린샷에서 특성 추출"""
        if screenshot is None:
            return {}
            
        try:
            # 기본 통계
            brightness = np.mean(screenshot)
            contrast = np.std(screenshot)
            
            # HSV 변환
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV)
            
            # 색상 히스토그램
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # 색상 분포
            red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            yellow_mask = cv2.inRange(hsv, (20, 50, 50), (40, 255, 255))
            
            total_pixels = screenshot.shape[0] * screenshot.shape[1]
            
            # 엣지 검출
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_pixels
            
            # 텍스처 분석
            texture = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            features = {
                'brightness': brightness,
                'contrast': contrast,
                'red_ratio': (np.sum(red_mask1) + np.sum(red_mask2)) / total_pixels,
                'blue_ratio': np.sum(blue_mask) / total_pixels,
                'green_ratio': np.sum(green_mask) / total_pixels,
                'yellow_ratio': np.sum(yellow_mask) / total_pixels,
                'edge_density': edge_density,
                'texture': texture,
                'hue_entropy': self._calculate_entropy(hist_h),
                'saturation_entropy': self._calculate_entropy(hist_s),
                'value_entropy': self._calculate_entropy(hist_v)
            }
            
            return features
            
        except Exception as e:
            print(f"❌ 특성 추출 오류: {e}")
            return {}
    
    def _calculate_entropy(self, histogram: np.ndarray) -> float:
        """히스토그램 엔트로피 계산"""
        histogram = histogram.flatten()
        histogram = histogram[histogram > 0]
        if len(histogram) == 0:
            return 0.0
        
        probabilities = histogram / np.sum(histogram)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def detect_state_change(self, current_features: Dict[str, float]) -> Tuple[bool, float]:
        """상태 변화 감지"""
        if not self.previous_states:
            self.previous_states.append(current_features)
            return False, 0.0
            
        prev_features = self.previous_states[-1]
        
        # 주요 특성들의 변화 계산
        change_score = 0.0
        important_features = ['brightness', 'red_ratio', 'blue_ratio', 'yellow_ratio', 'edge_density']
        
        for feature in important_features:
            if feature in prev_features and feature in current_features:
                diff = abs(current_features[feature] - prev_features[feature])
                change_score += diff
        
        self.previous_states.append(current_features)
        
        # 임계값 이상이면 상태 변화로 판정
        significant_change = change_score > 0.1
        
        return significant_change, change_score

class QLearningAgent:
    """Q-Learning 기반 게임 에이전트"""
    
    def __init__(self, actions: List[str], learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.3):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # 탐험 확률
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-테이블 (상태-행동 가치 함수)
        self.q_table = {}
        
        # 학습 통계
        self.total_episodes = 0
        self.total_rewards = 0
        self.episode_rewards = []
        
    def state_to_key(self, state: Dict[str, float]) -> str:
        """상태를 Q-테이블 키로 변환"""
        # 연속값을 이산화
        key_parts = []
        
        for feature, value in sorted(state.items()):
            if isinstance(value, (int, float)):
                # 값을 구간으로 나누어 이산화
                if feature == 'brightness':
                    bucket = int(value // 20)  # 0-19, 20-39, ...
                elif feature.endswith('_ratio'):
                    bucket = int(value * 10)   # 0.0-0.1 -> 0, 0.1-0.2 -> 1, ...
                else:
                    bucket = int(value // 10)
                
                key_parts.append(f"{feature}:{bucket}")
        
        return "|".join(key_parts[:6])  # 처음 6개 특성만 사용
    
    def get_action(self, state: Dict[str, float]) -> str:
        """상태에 대한 최적 행동 선택 (ε-greedy)"""
        state_key = self.state_to_key(state)
        
        # 새로운 상태면 Q값 초기화
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
        
        # ε-greedy 정책
        if random.random() < self.epsilon:
            # 탐험: 랜덤 행동
            action = random.choice(self.actions)
            exploration = True
        else:
            # 활용: 최대 Q값 행동
            q_values = self.q_table[state_key]
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)
            exploration = False
        
        return action
    
    def update_q_value(self, state: Dict[str, float], action: str, reward: float, 
                      next_state: Dict[str, float]) -> None:
        """Q값 업데이트 (Q-Learning)"""
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Q-테이블 초기화 (필요시)
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.actions}
        
        # 현재 Q값
        current_q = self.q_table[state_key][action]
        
        # 다음 상태의 최대 Q값
        max_next_q = max(self.q_table[next_state_key].values())
        
        # Q-Learning 업데이트 공식
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
        print(f"📚 Q학습: {action} | 보상:{reward:.2f} | Q:{current_q:.3f}→{new_q:.3f}")
    
    def decay_epsilon(self) -> None:
        """탐험 확률 감소"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        return {
            'total_states': len(self.q_table),
            'epsilon': self.epsilon,
            'total_episodes': self.total_episodes,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'q_table_size': len(self.q_table)
        }

class ExperienceDatabase:
    """경험 데이터베이스"""
    
    def __init__(self, db_path: str = "hero4_experience.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reward REAL NOT NULL,
                    next_state TEXT NOT NULL,
                    info TEXT,
                    session_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_actions INTEGER DEFAULT 0,
                    total_reward REAL DEFAULT 0.0,
                    battles_found INTEGER DEFAULT 0
                )
            """)
    
    def save_experience(self, experience: GameExperience, session_id: str) -> None:
        """경험 저장"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiences 
                (timestamp, state, action, reward, next_state, info, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                experience.timestamp.isoformat(),
                json.dumps(experience.state),
                experience.action,
                experience.reward,
                json.dumps(experience.next_state),
                json.dumps(experience.info),
                session_id
            ))
    
    def get_recent_experiences(self, limit: int = 1000) -> List[GameExperience]:
        """최근 경험들 가져오기"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT state, action, reward, next_state, timestamp, info
                FROM experiences 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            experiences = []
            for row in cursor:
                exp = GameExperience(
                    state=json.loads(row[0]),
                    action=row[1],
                    reward=row[2],
                    next_state=json.loads(row[3]),
                    info=json.loads(row[5]) if row[5] else {}
                )
                experiences.append(exp)
            
            return experiences

class LearningHero4AI:
    """학습하는 영웅전설4 AI"""
    
    def __init__(self):
        # 게임 연결
        self.dosbox_window = None
        self.game_region = None
        
        # AI 컴포넌트들
        self.state_analyzer = GameStateAnalyzer()
        self.actions = ['left', 'right', 'up', 'down', 'space', 'enter', 'z', 'x', 'a', 's', '1', '2']
        self.q_agent = QLearningAgent(self.actions)
        self.experience_db = ExperienceDatabase()
        
        # 학습 상태
        self.session_id = f"session_{int(time.time())}"
        self.current_state = {}
        self.last_action = None
        self.last_screenshot = None
        self.step_count = 0
        self.battle_count = 0
        self.total_reward = 0.0
        
        # 보상 시스템
        self.reward_calculator = RewardCalculator()
        
        print("🧠 학습하는 영웅전설4 AI 초기화 완료")
        print(f"🆔 세션 ID: {self.session_id}")
    
    def find_game_window(self) -> bool:
        """게임 창 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if 'dosbox' in window_text.lower() or 'ED4' in window_text:
                    windows.append(hwnd)
            return True

        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            self.dosbox_window = windows[0]
            self.game_region = win32gui.GetWindowRect(self.dosbox_window)
            print(f"🎮 게임 연결: {self.game_region}")
            return True
        
        return False
    
    def capture_game_screen(self) -> np.ndarray:
        """게임 화면 캡처"""
        try:
            screenshot = ImageGrab.grab(self.game_region)
            self.last_screenshot = screenshot
            return np.array(screenshot)
        except Exception as e:
            print(f"❌ 화면 캡처 실패: {e}")
            return None
    
    def send_action(self, action: str) -> bool:
        """게임에 행동 전송"""
        if not self.dosbox_window:
            return False
        
        try:
            win32gui.SetForegroundWindow(self.dosbox_window)
            time.sleep(0.05)
            
            key_map = {
                'left': 0x25, 'right': 0x27, 'up': 0x26, 'down': 0x28,
                'space': 0x20, 'enter': 0x0D, 'z': 0x5A, 'x': 0x58,
                'a': 0x41, 's': 0x53, '1': 0x31, '2': 0x32
            }
            
            if action in key_map:
                vk_code = key_map[action]
                win32api.keybd_event(vk_code, 0, 0, 0)
                time.sleep(0.08)
                win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
                return True
                
        except Exception as e:
            print(f"❌ 행동 전송 실패: {e}")
        
        return False
    
    async def learning_step(self) -> None:
        """한 번의 학습 스텝"""
        self.step_count += 1
        
        # 1. 현재 상태 관찰
        screenshot = self.capture_game_screen()
        if screenshot is None:
            return
            
        current_features = self.state_analyzer.extract_features(screenshot)
        if not current_features:
            return
        
        # 2. 이전 경험이 있다면 학습 업데이트
        if self.current_state and self.last_action:
            # 보상 계산
            reward = self.reward_calculator.calculate_reward(
                self.current_state, current_features, self.last_action
            )
            self.total_reward += reward
            
            # Q-Learning 업데이트
            self.q_agent.update_q_value(
                self.current_state, self.last_action, reward, current_features
            )
            
            # 경험 저장
            experience = GameExperience(
                self.current_state, self.last_action, reward, current_features,
                {'step': self.step_count, 'battle_count': self.battle_count}
            )
            self.experience_db.save_experience(experience, self.session_id)
        
        # 3. 다음 행동 선택
        action = self.q_agent.get_action(current_features)
        
        # 4. 행동 실행
        success = self.send_action(action)
        
        if success:
            print(f"🎮 스텝 {self.step_count}: {action} | ε={self.q_agent.epsilon:.3f}")
            
            # 5. 상태 업데이트
            self.current_state = current_features.copy()
            self.last_action = action
            
            # 6. 전투 감지
            if self._detect_battle(current_features):
                self.battle_count += 1
                print(f"⚔️ 전투 감지! 총 {self.battle_count}회")
        
        # 7. 탐험 확률 감소
        if self.step_count % 10 == 0:
            self.q_agent.decay_epsilon()
    
    def _detect_battle(self, features: Dict[str, float]) -> bool:
        """전투 감지 (학습 가능한 방식)"""
        # 여러 조건을 조합해서 전투 가능성 판단
        battle_indicators = 0
        
        if features.get('red_ratio', 0) > 0.05:
            battle_indicators += 1
        if features.get('blue_ratio', 0) > 0.08:
            battle_indicators += 1
        if features.get('yellow_ratio', 0) > 0.03:
            battle_indicators += 1
        if features.get('brightness', 0) > 80:
            battle_indicators += 1
        if features.get('contrast', 0) > 50:
            battle_indicators += 1
        
        return battle_indicators >= 3
    
    async def run_learning_session(self, max_steps: int = 500, target_battles: int = 15) -> None:
        """학습 세션 실행"""
        print("🚀 학습 세션 시작!")
        print(f"🎯 목표: {max_steps}스텝 내에 {target_battles}회 전투")
        
        if not self.find_game_window():
            print("❌ 게임 창을 찾을 수 없습니다!")
            return
        
        start_time = time.time()
        
        while self.step_count < max_steps and self.battle_count < target_battles:
            await self.learning_step()
            await asyncio.sleep(0.15)  # 적당한 속도
            
            # 진행 상황 출력
            if self.step_count % 50 == 0:
                stats = self.q_agent.get_learning_stats()
                elapsed = time.time() - start_time
                sps = self.step_count / elapsed if elapsed > 0 else 0
                
                print(f"📊 진행: {self.step_count}/{max_steps} | 전투:{self.battle_count}/{target_battles}")
                print(f"   🧠 상태:{stats['total_states']} | ε:{stats['epsilon']:.3f} | 보상:{self.total_reward:.2f} | {sps:.1f}sps")
        
        # 결과 요약
        elapsed = time.time() - start_time
        print(f"\n🏁 학습 세션 완료!")
        print(f"⏱️ 시간: {elapsed:.1f}초")
        print(f"🎮 스텝: {self.step_count}")
        print(f"⚔️ 전투: {self.battle_count}")
        print(f"💰 총 보상: {self.total_reward:.2f}")
        
        final_stats = self.q_agent.get_learning_stats()
        print(f"🧠 학습된 상태: {final_stats['total_states']}개")
        print(f"🔍 최종 탐험률: {final_stats['epsilon']:.3f}")
        
        if self.battle_count >= target_battles:
            print("🎉 목표 달성! AI가 성공적으로 학습했습니다!")
        else:
            print("📈 부분 성공. AI가 경험을 쌓았습니다.")

class RewardCalculator:
    """보상 계산기"""
    
    def __init__(self):
        self.previous_features = {}
    
    def calculate_reward(self, prev_state: Dict[str, float], 
                        current_state: Dict[str, float], action: str) -> float:
        """보상 계산"""
        reward = 0.0
        
        # 1. 기본 행동 보상 (생존)
        reward += 0.01
        
        # 2. 화면 변화 보상 (탐험)
        change_reward = self._calculate_change_reward(prev_state, current_state)
        reward += change_reward
        
        # 3. 전투 관련 보상
        battle_reward = self._calculate_battle_reward(current_state)
        reward += battle_reward
        
        # 4. 탐험 보상
        exploration_reward = self._calculate_exploration_reward(current_state, action)
        reward += exploration_reward
        
        # 5. 패널티
        penalty = self._calculate_penalty(prev_state, current_state)
        reward -= penalty
        
        return reward
    
    def _calculate_change_reward(self, prev_state: Dict, current_state: Dict) -> float:
        """화면 변화 보상"""
        if not prev_state:
            return 0.0
        
        change_score = 0.0
        for key in ['brightness', 'red_ratio', 'blue_ratio', 'edge_density']:
            if key in prev_state and key in current_state:
                diff = abs(current_state[key] - prev_state[key])
                change_score += diff
        
        return min(change_score * 2.0, 0.5)  # 최대 0.5점
    
    def _calculate_battle_reward(self, current_state: Dict) -> float:
        """전투 상황 보상"""
        battle_score = 0.0
        
        # 전투 관련 색상들에 대한 보상
        red_ratio = current_state.get('red_ratio', 0)
        blue_ratio = current_state.get('blue_ratio', 0) 
        yellow_ratio = current_state.get('yellow_ratio', 0)
        
        if red_ratio > 0.05:
            battle_score += 2.0
        if blue_ratio > 0.08:
            battle_score += 1.5
        if yellow_ratio > 0.03:
            battle_score += 1.0
            
        return battle_score
    
    def _calculate_exploration_reward(self, current_state: Dict, action: str) -> float:
        """탐험 보상"""
        # 다양한 행동에 대한 작은 보상
        action_rewards = {
            'left': 0.1, 'right': 0.1, 'up': 0.05, 'down': 0.05,
            'space': 0.2, 'enter': 0.15, 'z': 0.1, 'a': 0.1
        }
        
        return action_rewards.get(action, 0.0)
    
    def _calculate_penalty(self, prev_state: Dict, current_state: Dict) -> float:
        """패널티 계산"""
        penalty = 0.0
        
        # 너무 어두운 화면 패널티
        brightness = current_state.get('brightness', 0)
        if brightness < 10:
            penalty += 0.5
        
        # 변화 없음 패널티 (정체)
        if prev_state:
            total_change = sum(abs(current_state.get(k, 0) - prev_state.get(k, 0)) 
                             for k in ['brightness', 'red_ratio', 'blue_ratio'])
            if total_change < 0.01:
                penalty += 0.2
        
        return penalty

# 실행
if __name__ == "__main__":
    async def main():
        ai = LearningHero4AI()
        await ai.run_learning_session(max_steps=500, target_battles=15)
    
    print("🧠 진정한 AI 학습 시스템")
    print("=" * 60)
    print("🎯 특징: 강화학습, 경험 축적, 자율 판단")
    asyncio.run(main())