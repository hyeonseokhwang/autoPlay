#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
진정한 자율학습 영웅전설4 AI - 걸음마부터 시작하는 학습 시스템
"""

import asyncio
import time
import random
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
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
class Experience:
    """경험 데이터"""
    situation: str          # 상황 (화면 해시)
    action: str            # 취한 액션
    outcome: str           # 결과 (다음 화면 해시)
    reward: float          # 보상
    timestamp: float       # 시간
    success: bool          # 성공 여부
    notes: str = ""        # 메모

@dataclass
class Pattern:
    """발견된 패턴"""
    trigger: str           # 트리거 상황
    action: str            # 권장 액션
    confidence: float      # 신뢰도
    success_count: int     # 성공 횟수
    total_count: int       # 총 시도 횟수
    last_used: float       # 마지막 사용 시간

class BabyStepsController:
    """걸음마 단계 컨트롤러 - 아주 기본적인 것부터"""
    
    def __init__(self):
        self.dosbox_hwnd = None
        self.last_input_time = 0
        
        # 아기 AI가 시도해볼 수 있는 모든 액션
        self.all_possible_actions = [
            # 방향키들
            'up', 'down', 'left', 'right',
            # 기본 키들
            'enter', 'space', 'esc',
            # 문자 키들
            'z', 'x', 'c', 'a', 's', 'd',
            # 숫자 키들  
            '1', '2', '3', '4', '5',
            # 기능키들
            'f1', 'f2', 'f10',
            # 기타
            'tab', 'shift', 'ctrl'
        ]
        
        self.tried_actions = set()
        
    def find_game(self):
        """게임 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if any(keyword in title.lower() for keyword in ["dosbox", "ed4"]):
                    windows.append((hwnd, title))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            self.dosbox_hwnd, title = windows[0]
            return True
        return False
    
    def try_random_action(self) -> str:
        """랜덤 액션 시도 (탐험)"""
        return random.choice(self.all_possible_actions)
    
    def send_action(self, action: str) -> bool:
        """액션 실행"""
        current_time = time.time()
        if current_time - self.last_input_time < 0.2:
            return False
        
        if not self.dosbox_hwnd:
            return False
        
        try:
            # 키 매핑
            key_map = {
                'up': win32con.VK_UP, 'down': win32con.VK_DOWN,
                'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT,
                'enter': win32con.VK_RETURN, 'space': win32con.VK_SPACE,
                'esc': win32con.VK_ESCAPE, 'tab': win32con.VK_TAB,
                'shift': win32con.VK_SHIFT, 'ctrl': win32con.VK_CONTROL,
                'z': ord('Z'), 'x': ord('X'), 'c': ord('C'),
                'a': ord('A'), 's': ord('S'), 'd': ord('D'),
                '1': ord('1'), '2': ord('2'), '3': ord('3'),
                '4': ord('4'), '5': ord('5'),
                'f1': win32con.VK_F1, 'f2': win32con.VK_F2,
                'f10': win32con.VK_F10
            }
            
            if action.lower() not in key_map:
                return False
            
            # 윈도우 활성화 시도
            try:
                win32gui.SetForegroundWindow(self.dosbox_hwnd)
                time.sleep(0.05)
            except:
                pass  # 실패해도 계속 진행
            
            # 키 입력
            vk = key_map[action.lower()]
            win32api.keybd_event(vk, 0, 0, 0)
            time.sleep(0.1)
            win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
            
            self.last_input_time = current_time
            self.tried_actions.add(action)
            
            return True
            
        except Exception as e:
            return False

class CuriousVision:
    """호기심 많은 시각 시스템"""
    
    def __init__(self):
        self.capture_region = None
        self.seen_screens = {}  # 해시 -> 정보
        self.screen_transitions = []  # (from, action, to) 기록
        
    def setup_eyes(self, hwnd):
        """눈 설정"""
        try:
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            self.capture_region = (x + 8, y + 30, x2 - x - 16, y2 - y - 38)
            return True
        except:
            return False
    
    def look_around(self) -> Optional[Dict]:
        """주변 관찰"""
        if not self.capture_region:
            return None
        
        try:
            # 화면 캡처
            screenshot = pyautogui.screenshot(region=self.capture_region)
            image = np.array(screenshot)
            
            # 간단한 분석
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 화면 특징 추출
            tiny = cv2.resize(gray, (20, 15))
            screen_hash = hashlib.md5(tiny.tobytes()).hexdigest()[:10]
            
            brightness = np.mean(gray)
            
            # 색상 분포 (간단하게)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            color_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180])
            dominant_colors = np.argsort(color_hist.flatten())[-3:]
            
            # 에지 (변화 감지)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 화면 정보
            screen_info = {
                'hash': screen_hash,
                'brightness': float(brightness),
                'edge_density': float(edge_density),
                'dominant_colors': dominant_colors.tolist(),
                'timestamp': time.time(),
                'size': image.shape[:2]
            }
            
            # 새로운 화면인지 확인
            is_new = screen_hash not in self.seen_screens
            if is_new:
                self.seen_screens[screen_hash] = screen_info
                print(f"👀 새로운 화면 발견! #{len(self.seen_screens)}: {screen_hash}")
            
            screen_info['is_new'] = is_new
            screen_info['visit_count'] = self.seen_screens[screen_hash].get('visit_count', 0) + 1
            self.seen_screens[screen_hash]['visit_count'] = screen_info['visit_count']
            
            return screen_info
            
        except Exception as e:
            return None

class LearningBrain:
    """학습하는 뇌"""
    
    def __init__(self):
        # SQLite 기반 RAG 메모리
        self.conn = sqlite3.connect('baby_ai_memory.db')
        self.init_memory()
        
        # 실시간 학습 데이터
        self.experiences = deque(maxlen=1000)
        self.patterns = {}
        self.curiosity_score = 1.0  # 호기심 점수
        
        # 학습 상태
        self.total_actions = 0
        self.successful_actions = 0
        self.discovered_screens = 0
        
        # 가설들 (AI가 스스로 세우는)
        self.hypotheses = [
            "방향키는 이동에 사용될 것이다",
            "enter나 space는 선택/확인일 것이다", 
            "esc는 취소나 뒤로가기일 것이다",
            "같은 화면에서 다른 액션을 하면 다른 결과가 나올 것이다",
            "밝은 화면과 어두운 화면은 다른 상황일 것이다"
        ]
        
    def init_memory(self):
        """기억 저장소 초기화"""
        cursor = self.conn.cursor()
        
        # 경험 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                situation TEXT,
                action TEXT,
                outcome TEXT,
                reward REAL,
                success INTEGER,
                timestamp REAL,
                notes TEXT
            )
        ''')
        
        # 패턴 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger TEXT,
                action TEXT,
                confidence REAL,
                success_count INTEGER,
                total_count INTEGER,
                last_used REAL
            )
        ''')
        
        # 가설 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis TEXT,
                evidence_count INTEGER,
                confidence REAL,
                created_at REAL
            )
        ''')
        
        self.conn.commit()
    
    def calculate_reward(self, before_screen: Dict, action: str, after_screen: Dict) -> float:
        """보상 계산 - AI가 스스로 무엇이 좋은지 학습"""
        reward = 0.0
        
        if not after_screen:
            return -1.0  # 실패
        
        # 새로운 화면 발견 보상 (탐험 욕구)
        if after_screen.get('is_new', False):
            reward += 5.0
            print(f"🌟 탐험 보상! 새로운 곳 발견: +5.0")
        
        # 화면 변화 보상
        if before_screen['hash'] != after_screen['hash']:
            reward += 2.0
            print(f"🔄 변화 감지: +2.0")
        
        # 밝기 변화 보상 (뭔가 일어남)
        brightness_change = abs(before_screen['brightness'] - after_screen['brightness'])
        if brightness_change > 20:
            reward += brightness_change * 0.1
            print(f"💡 화면 변화: +{brightness_change * 0.1:.1f}")
        
        # 호기심 보상 (적게 방문한 화면)
        visit_count = after_screen.get('visit_count', 1)
        curiosity_bonus = max(0, 3.0 - visit_count * 0.5)
        reward += curiosity_bonus
        
        # 패턴 확인 보상
        if self.check_hypothesis_evidence(before_screen, action, after_screen):
            reward += 1.0
            print(f"🧠 가설 확인: +1.0")
        
        return reward
    
    def check_hypothesis_evidence(self, before: Dict, action: str, after: Dict) -> bool:
        """가설에 대한 증거 확인"""
        # 간단한 패턴 확인
        if action in ['up', 'down', 'left', 'right'] and before['hash'] != after['hash']:
            return True  # 방향키로 화면이 바뀜
        
        if action in ['enter', 'space'] and before['brightness'] != after['brightness']:
            return True  # 확인키로 뭔가 변함
        
        return False
    
    def learn_from_experience(self, before_screen: Dict, action: str, 
                            after_screen: Dict, success: bool):
        """경험으로부터 학습"""
        
        reward = self.calculate_reward(before_screen, action, after_screen) if success else -0.5
        
        # 경험 생성
        experience = Experience(
            situation=before_screen['hash'],
            action=action,
            outcome=after_screen['hash'] if after_screen else 'FAILED',
            reward=reward,
            timestamp=time.time(),
            success=success,
            notes=f"brightness_change: {abs(before_screen['brightness'] - after_screen.get('brightness', 0)):.1f}"
        )
        
        # 메모리에 저장
        self.experiences.append(experience)
        
        # DB에 영구 저장
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO experiences 
            (situation, action, outcome, reward, success, timestamp, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (experience.situation, experience.action, experience.outcome,
              experience.reward, 1 if experience.success else 0,
              experience.timestamp, experience.notes))
        
        # 패턴 학습
        self.update_patterns(before_screen, action, reward > 0)
        
        # 통계 업데이트
        self.total_actions += 1
        if success and reward > 0:
            self.successful_actions += 1
        
        self.conn.commit()
    
    def update_patterns(self, situation: Dict, action: str, was_good: bool):
        """패턴 업데이트"""
        # 상황을 간단한 키로 변환
        situation_key = f"bright_{int(situation['brightness']/30)}"
        pattern_key = f"{situation_key}_{action}"
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM patterns WHERE trigger = ? AND action = ?', 
                      (situation_key, action))
        existing = cursor.fetchone()
        
        if existing:
            # 기존 패턴 업데이트
            pattern_id, _, _, confidence, success_count, total_count, _ = existing
            new_total = total_count + 1
            new_success = success_count + (1 if was_good else 0)
            new_confidence = new_success / new_total
            
            cursor.execute('''
                UPDATE patterns 
                SET confidence = ?, success_count = ?, total_count = ?, last_used = ?
                WHERE id = ?
            ''', (new_confidence, new_success, new_total, time.time(), pattern_id))
        else:
            # 새 패턴 생성
            cursor.execute('''
                INSERT INTO patterns (trigger, action, confidence, success_count, total_count, last_used)
                VALUES (?, ?, ?, ?, 1, ?)
            ''', (situation_key, action, 1.0 if was_good else 0.0, 
                  1 if was_good else 0, time.time()))
    
    def choose_next_action(self, current_situation: Dict) -> str:
        """다음 액션 선택 - 학습된 패턴 vs 탐험"""
        
        # 상황 분석
        situation_key = f"bright_{int(current_situation['brightness']/30)}"
        
        # 학습된 패턴 찾기
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT action, confidence FROM patterns 
            WHERE trigger = ? AND confidence > 0.3 
            ORDER BY confidence DESC, last_used DESC
        ''', (situation_key,))
        
        learned_actions = cursor.fetchall()
        
        # 탐험 vs 활용 결정
        exploration_chance = max(0.1, self.curiosity_score - (self.successful_actions / max(1, self.total_actions)))
        
        if learned_actions and random.random() > exploration_chance:
            # 학습된 것 활용
            best_action, confidence = learned_actions[0]
            print(f"🧠 학습된 행동: {best_action} (신뢰도: {confidence:.2f})")
            return best_action
        else:
            # 탐험
            all_actions = ['up', 'down', 'left', 'right', 'enter', 'space', 'esc', 
                          'z', 'x', 'c', '1', '2', 'f1', 'tab']
            action = random.choice(all_actions)
            print(f"🔍 탐험 행동: {action}")
            return action
    
    def get_learning_stats(self) -> Dict:
        """학습 통계"""
        cursor = self.conn.cursor()
        
        # 총 경험
        cursor.execute('SELECT COUNT(*) FROM experiences')
        total_exp = cursor.fetchone()[0]
        
        # 성공률
        cursor.execute('SELECT AVG(success) FROM experiences')
        success_rate = cursor.fetchone()[0] or 0
        
        # 평균 보상
        cursor.execute('SELECT AVG(reward) FROM experiences WHERE reward > 0')
        avg_reward = cursor.fetchone()[0] or 0
        
        # 학습된 패턴
        cursor.execute('SELECT COUNT(*) FROM patterns WHERE confidence > 0.5')
        good_patterns = cursor.fetchone()[0]
        
        # 발견한 화면 수
        cursor.execute('SELECT COUNT(DISTINCT situation) FROM experiences')
        discovered_screens = cursor.fetchone()[0]
        
        return {
            'total_experiences': total_exp,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'learned_patterns': good_patterns,
            'discovered_screens': discovered_screens,
            'curiosity_score': self.curiosity_score
        }

class BabyAI:
    """걸음마 단계 AI - 스스로 학습하며 성장"""
    
    def __init__(self):
        self.controller = BabyStepsController()
        self.vision = CuriousVision()
        self.brain = LearningBrain()
        
        self.current_screen = None
        self.step_count = 0
        
        print("👶 Baby AI 탄생!")
        print("🧠 AI가 스스로 배울 것들:")
        for hypothesis in self.brain.hypotheses:
            print(f"   💭 {hypothesis}")
    
    def initialize(self):
        """초기화"""
        if not self.controller.find_game():
            print("❌ 게임을 찾을 수 없어요!")
            return False
        
        if not self.vision.setup_eyes(self.controller.dosbox_hwnd):
            print("❌ 눈을 뜰 수 없어요!")
            return False
        
        print("👀 Baby AI가 눈을 떴어요!")
        return True
    
    async def take_baby_step(self) -> Dict:
        """아기 한 걸음"""
        try:
            # 1. 주변 관찰
            current_screen = self.vision.look_around()
            if not current_screen:
                return {'success': False, 'error': '시각 장애'}
            
            # 2. 행동 결정 (학습 vs 탐험)
            if self.current_screen:
                action = self.brain.choose_next_action(current_screen)
            else:
                # 첫 번째 행동은 랜덤
                action = self.controller.try_random_action()
                print(f"🍼 첫 번째 랜덤 행동: {action}")
            
            # 3. 행동 실행
            success = self.controller.send_action(action)
            
            # 4. 결과 관찰
            await asyncio.sleep(0.5)  # 게임 반응 대기
            result_screen = self.vision.look_around()
            
            # 5. 경험으로부터 학습
            if self.current_screen and result_screen:
                self.brain.learn_from_experience(
                    self.current_screen, action, result_screen, success
                )
            
            # 6. 기억 업데이트
            self.current_screen = current_screen
            self.step_count += 1
            
            return {
                'success': success,
                'action': action,
                'step': self.step_count,
                'screens_discovered': len(self.vision.seen_screens),
                'is_new_screen': current_screen.get('is_new', False)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

async def watch_baby_grow():
    """아기 AI가 성장하는 모습 관찰"""
    print("👶 Baby AI 성장 일기")
    print("=" * 50)
    
    baby = BabyAI()
    
    if not baby.initialize():
        return
    
    print("\n🍼 Baby AI가 게임을 배우기 시작합니다!")
    print("(AI가 스스로 시행착오하며 학습하는 과정을 지켜보세요)\n")
    
    growth_phases = [
        (50, "🍼 신생아 단계 - 무작위 시도"),
        (100, "👶 유아 단계 - 패턴 인식 시작"), 
        (200, "🧒 아동 단계 - 학습된 행동 활용"),
        (300, "🎓 청소년 단계 - 전략적 사고")
    ]
    
    current_phase = 0
    
    for step in range(1, 301):
        # 성장 단계 체크
        if current_phase < len(growth_phases) and step > growth_phases[current_phase][0]:
            print(f"\n🌟 {growth_phases[current_phase][1]}")
            current_phase += 1
        
        # 한 걸음 내딛기
        result = await baby.take_baby_step()
        
        # 5걸음마다 성장 상황 보고
        if step % 5 == 0:
            stats = baby.brain.get_learning_stats()
            
            if result['success']:
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} 걸음 #{step:3d} | {result.get('action', '?'):5s} | "
                  f"발견:{result.get('screens_discovered', 0):2d}곳 | "
                  f"성공률:{stats['success_rate']:.2f}")
            
            if result.get('is_new_screen'):
                print(f"        🌟 새 세계 발견!")
        
        # 20걸음마다 학습 보고서
        if step % 20 == 0:
            stats = baby.brain.get_learning_stats()
            print(f"\n📊 성장 보고서 (걸음 {step}):")
            print(f"    🧠 총 경험: {stats['total_experiences']}개")
            print(f"    📈 성공률: {stats['success_rate']:.2f}")
            print(f"    🎯 평균 보상: {stats['avg_reward']:.1f}")
            print(f"    🔍 학습된 패턴: {stats['learned_patterns']}개")
            print(f"    🗺️ 발견한 세계: {stats['discovered_screens']}곳")
            print(f"    👀 호기심 지수: {stats['curiosity_score']:.2f}")
            print()
        
        # 잠깐 쉬기
        await asyncio.sleep(0.1)
    
    # 성장 완료 보고
    final_stats = baby.brain.get_learning_stats()
    print(f"\n🎓 Baby AI 성장 완료!")
    print(f"총 300걸음을 통해 학습한 내용:")
    print(f"  🧠 축적된 경험: {final_stats['total_experiences']}개")
    print(f"  🎯 최종 성공률: {final_stats['success_rate']:.2f}")
    print(f"  📚 학습된 패턴: {final_stats['learned_patterns']}개")
    print(f"  🗺️ 탐험한 세계: {final_stats['discovered_screens']}곳")
    
    if final_stats['success_rate'] > 0.5:
        print("🏆 훌륭한 성장! AI가 게임을 어느 정도 이해했어요!")
    elif final_stats['success_rate'] > 0.3:
        print("👍 괜찮은 성장! 계속 학습하면 더 나아질 거예요!")
    else:
        print("🌱 아직 어린 AI! 더 많은 경험이 필요해요!")

if __name__ == "__main__":
    try:
        asyncio.run(watch_baby_grow())
    except KeyboardInterrupt:
        print("\n👋 Baby AI가 잠들었어요...")
    except Exception as e:
        print(f"\n❌ Baby AI에게 문제가 생겼어요: {e}")
        import traceback
        traceback.print_exc()