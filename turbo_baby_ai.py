#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
터보 Baby AI - 초고속 자율학습 시스템
완전 제로베이스 강화학습, 외부 ML 모델 없음
"""

import asyncio
import time
import random
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import hashlib
import ctypes
from ctypes import wintypes
import threading

# 게임 제어
import cv2
import pyautogui
import win32gui
import win32con
import win32api
import win32process

@dataclass
class Experience:
    """학습 경험"""
    state_hash: str      # 상태 (화면 해시)
    action: str         # 행동
    next_state: str     # 다음 상태
    reward: float       # 보상
    timestamp: float    # 시간

class TurboController:
    """초고속 DOSBox 전용 컨트롤러"""
    
    def __init__(self):
        self.dosbox_hwnd = None
        self.input_lock = threading.Lock()
        
        # 모든 가능한 액션 (확장됨)
        self.action_space = [
            # 핵심 이동
            'up', 'down', 'left', 'right',
            # 핵심 액션  
            'enter', 'space', 'esc',
            # 게임 특화
            'z', 'x', 'c', 'a', 's', 'd',
            # 숫자 (메뉴 선택)
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            # 기능키 (게임 기능)
            'f1', 'f2', 'f3', 'f4', 'f5', 'f10',
            # 기타
            'tab', 'shift+tab', 'ctrl', 'alt',
            # 조합키 (자주 사용)
            'alt+f4', 'ctrl+s', 'shift+enter'
        ]
        
    def find_dosbox_precise(self) -> bool:
        """DOSBox 정확히 찾기"""
        def enum_callback(hwnd, windows):
            if not win32gui.IsWindowVisible(hwnd):
                return True
                
            try:
                title = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                
                # DOSBox 식별
                dosbox_indicators = [
                    'dosbox' in title.lower(),
                    'ed4' in title.lower(),
                    'legend' in title.lower(),
                    'hero' in title.lower(),
                    'SDL_app' in class_name  # DOSBox 클래스명
                ]
                
                if any(dosbox_indicators):
                    # 프로세스 확인
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        process_name = self.get_process_name(pid)
                        if 'dosbox' in process_name.lower():
                            windows.append((hwnd, title, class_name))
                    except:
                        windows.append((hwnd, title, class_name))
                        
            except:
                pass
            return True
        
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            # 가장 적합한 윈도우 선택
            for hwnd, title, class_name in windows:
                self.dosbox_hwnd = hwnd
                print(f"🎯 DOSBox 발견: {title} (클래스: {class_name})")
                return True
        
        return False
    
    def get_process_name(self, pid):
        """프로세스 이름 획득"""
        try:
            import psutil
            return psutil.Process(pid).name()
        except:
            return "unknown"
    
    def send_key_direct(self, action: str) -> bool:
        """DOSBox에 직접 키 전송"""
        if not self.dosbox_hwnd:
            return False
            
        with self.input_lock:
            try:
                # 윈도우 활성화 (강제)
                try:
                    win32gui.ShowWindow(self.dosbox_hwnd, win32con.SW_RESTORE)
                    win32gui.BringWindowToTop(self.dosbox_hwnd)
                    win32gui.SetForegroundWindow(self.dosbox_hwnd)
                except:
                    pass
                
                # 키 매핑
                key_mappings = {
                    # 방향키
                    'up': win32con.VK_UP, 'down': win32con.VK_DOWN,
                    'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT,
                    # 기본키
                    'enter': win32con.VK_RETURN, 'space': win32con.VK_SPACE,
                    'esc': win32con.VK_ESCAPE, 'tab': win32con.VK_TAB,
                    # 문자키
                    'z': ord('Z'), 'x': ord('X'), 'c': ord('C'),
                    'a': ord('A'), 's': ord('S'), 'd': ord('D'),
                    # 숫자키
                    '1': ord('1'), '2': ord('2'), '3': ord('3'),
                    '4': ord('4'), '5': ord('5'), '6': ord('6'),
                    '7': ord('7'), '8': ord('8'), '9': ord('9'), '0': ord('0'),
                    # 기능키
                    'f1': win32con.VK_F1, 'f2': win32con.VK_F2,
                    'f3': win32con.VK_F3, 'f4': win32con.VK_F4,
                    'f5': win32con.VK_F5, 'f10': win32con.VK_F10,
                    # 수정키
                    'shift': win32con.VK_SHIFT, 'ctrl': win32con.VK_CONTROL,
                    'alt': win32con.VK_MENU
                }
                
                # 조합키 처리
                if '+' in action:
                    keys = action.split('+')
                    vk_keys = []
                    
                    for key in keys:
                        if key in key_mappings:
                            vk_keys.append(key_mappings[key])
                    
                    if len(vk_keys) >= 2:
                        # 수정키 누르기
                        for vk in vk_keys[:-1]:
                            win32api.keybd_event(vk, 0, 0, 0)
                        
                        time.sleep(0.01)
                        
                        # 메인키 누르기/떼기
                        main_key = vk_keys[-1]
                        win32api.keybd_event(main_key, 0, 0, 0)
                        time.sleep(0.02)
                        win32api.keybd_event(main_key, 0, win32con.KEYEVENTF_KEYUP, 0)
                        
                        time.sleep(0.01)
                        
                        # 수정키 떼기 (역순)
                        for vk in reversed(vk_keys[:-1]):
                            win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
                
                else:
                    # 단일키 처리
                    if action in key_mappings:
                        vk = key_mappings[action]
                        win32api.keybd_event(vk, 0, 0, 0)
                        time.sleep(0.02)
                        win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
                
                return True
                
            except Exception as e:
                return False

class HyperVision:
    """초고속 시각 시스템"""
    
    def __init__(self):
        self.capture_region = None
        self.screen_cache = {}
        self.last_capture_time = 0
        
    def setup_vision(self, hwnd) -> bool:
        """시각 시스템 설정"""
        try:
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            
            # DOSBox 내부 게임 영역만 캡처 (테두리 제외)
            margin_x, margin_y = 8, 30
            self.capture_region = (
                x + margin_x, 
                y + margin_y, 
                x2 - x - margin_x * 2, 
                y2 - y - margin_y - 8
            )
            
            print(f"📸 시각 영역: {self.capture_region}")
            return True
            
        except Exception as e:
            print(f"❌ 시각 설정 실패: {e}")
            return False
    
    def capture_state(self) -> Optional[Dict]:
        """초고속 화면 캡처 및 상태 분석"""
        current_time = time.time()
        
        # 캡처 빈도 제한 (너무 빠르면 의미 없음)
        if current_time - self.last_capture_time < 0.03:
            return None
            
        try:
            # 화면 캡처
            screenshot = pyautogui.screenshot(region=self.capture_region)
            image = np.array(screenshot)
            
            # 고속 특징 추출
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 해시 생성 (상태 식별용)
            # 다양한 크기로 해시 생성하여 세밀도 조절
            tiny = cv2.resize(gray, (16, 12))  # 매우 작게
            medium = cv2.resize(gray, (32, 24))  # 중간
            
            state_hash = hashlib.md5(tiny.tobytes()).hexdigest()[:8]
            detail_hash = hashlib.md5(medium.tobytes()).hexdigest()[:12]
            
            # 빠른 특징들
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # HSV 기반 색상 정보
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 주요 색상 비율
            hue_hist = cv2.calcHist([hsv], [0], None, [6], [0, 180])
            dominant_hue = int(np.argmax(hue_hist))
            
            # 화면 영역별 분석 (게임 UI 구분)
            h, w = gray.shape
            regions = {
                'top': np.mean(gray[:h//4, :]),          # 상단 (UI)
                'center': np.mean(gray[h//4:3*h//4, :]), # 중앙 (메인)
                'bottom': np.mean(gray[3*h//4:, :]),     # 하단 (메뉴)
                'left': np.mean(gray[:, :w//4]),         # 좌측
                'right': np.mean(gray[:, 3*w//4:])       # 우측
            }
            
            state = {
                'hash': state_hash,
                'detail_hash': detail_hash,
                'brightness': brightness,
                'contrast': contrast,
                'dominant_hue': dominant_hue,
                'regions': regions,
                'timestamp': current_time,
                'size': image.shape[:2]
            }
            
            # 캐시 확인 (새로운 상태인지)
            is_new = state_hash not in self.screen_cache
            if is_new:
                self.screen_cache[state_hash] = {
                    'first_seen': current_time,
                    'visit_count': 0,
                    'last_visit': current_time
                }
                
            cache_entry = self.screen_cache[state_hash]
            cache_entry['visit_count'] += 1
            cache_entry['last_visit'] = current_time
            
            state['is_new'] = is_new
            state['visit_count'] = cache_entry['visit_count']
            state['novelty_score'] = 1.0 / max(1, cache_entry['visit_count'])
            
            self.last_capture_time = current_time
            return state
            
        except Exception as e:
            return None

class ZeroBaseBrain:
    """완전 제로베이스 학습 뇌 - 외부 모델 없음"""
    
    def __init__(self):
        # SQLite 메모리 (RAG)
        self.conn = sqlite3.connect(':memory:')  # 메모리 DB로 속도 향상
        self.init_tables()
        
        # 실시간 학습 데이터
        self.experiences = deque(maxlen=10000)  # 더 많은 경험 저장
        self.q_table = {}  # Q-Learning 테이블
        
        # 학습 파라미터
        self.learning_rate = 0.3    # 학습 속도
        self.discount_factor = 0.9  # 미래 보상 할인율
        self.epsilon = 0.7          # 탐험 확률 (높게 시작)
        self.epsilon_decay = 0.995  # 탐험 감소율
        self.epsilon_min = 0.05     # 최소 탐험율
        
        # 통계
        self.step_count = 0
        self.reward_history = deque(maxlen=1000)
        
    def init_tables(self):
        """메모리 DB 테이블 초기화"""
        cursor = self.conn.cursor()
        
        # Q값 테이블 (상태-액션 가치)
        cursor.execute('''
            CREATE TABLE q_values (
                state_hash TEXT,
                action TEXT,
                q_value REAL,
                visit_count INTEGER,
                last_update REAL,
                PRIMARY KEY (state_hash, action)
            )
        ''')
        
        # 경험 테이블
        cursor.execute('''
            CREATE TABLE experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_hash TEXT,
                action TEXT,
                reward REAL,
                next_state TEXT,
                timestamp REAL
            )
        ''')
        
        # 상태 통계
        cursor.execute('''
            CREATE TABLE state_stats (
                state_hash TEXT PRIMARY KEY,
                visit_count INTEGER,
                avg_reward REAL,
                first_seen REAL,
                last_seen REAL
            )
        ''')
        
        self.conn.commit()
    
    def calculate_reward(self, prev_state: Dict, action: str, new_state: Dict) -> float:
        """보상 계산 - 게임 진행에 유리한 행동 학습"""
        if not new_state:
            return -2.0  # 실패 큰 페널티
        
        reward = 0.0
        
        # 1. 탐험 보상 (새로운 상태 발견)
        if new_state.get('is_new', False):
            reward += 10.0
            print(f"🌟 신세계 발견 보너스: +10.0")
        
        # 2. 희귀성 보상 (적게 방문한 곳)
        novelty = new_state.get('novelty_score', 0.5)
        novelty_bonus = novelty * 3.0
        reward += novelty_bonus
        
        # 3. 상태 변화 보상 (뭔가 일어남)
        if prev_state['hash'] != new_state['hash']:
            reward += 5.0
            print(f"🔄 상태 변화: +5.0")
        
        # 4. 시각적 변화 보상
        brightness_change = abs(prev_state['brightness'] - new_state['brightness'])
        if brightness_change > 15:
            reward += min(brightness_change * 0.2, 3.0)
            
        contrast_change = abs(prev_state['contrast'] - new_state['contrast'])
        if contrast_change > 10:
            reward += min(contrast_change * 0.1, 2.0)
        
        # 5. 영역별 변화 보상 (UI 반응)
        region_changes = 0
        for region in ['top', 'center', 'bottom', 'left', 'right']:
            change = abs(prev_state['regions'][region] - new_state['regions'][region])
            if change > 5:
                region_changes += 1
        
        if region_changes > 0:
            reward += region_changes * 1.0
            
        # 6. 색상 변화 보상 (화면 전환)
        if prev_state['dominant_hue'] != new_state['dominant_hue']:
            reward += 2.0
            
        # 7. 정체 페널티 (같은 곳에 너무 오래)
        if new_state.get('visit_count', 1) > 10:
            reward -= 1.0
            
        return reward
    
    def get_q_value(self, state_hash: str, action: str) -> float:
        """Q값 조회"""
        if (state_hash, action) in self.q_table:
            return self.q_table[(state_hash, action)]
        
        # DB에서 조회
        cursor = self.conn.cursor()
        cursor.execute('SELECT q_value FROM q_values WHERE state_hash = ? AND action = ?',
                      (state_hash, action))
        result = cursor.fetchone()
        
        if result:
            q_val = result[0]
            self.q_table[(state_hash, action)] = q_val  # 캐시
            return q_val
        
        return 0.0  # 초기값
    
    def update_q_value(self, state_hash: str, action: str, reward: float, next_state_hash: str):
        """Q값 업데이트 (Q-Learning)"""
        # 현재 Q값
        current_q = self.get_q_value(state_hash, action)
        
        # 다음 상태의 최대 Q값
        cursor = self.conn.cursor()
        cursor.execute('SELECT MAX(q_value) FROM q_values WHERE state_hash = ?', 
                      (next_state_hash,))
        result = cursor.fetchone()
        max_next_q = result[0] if result and result[0] else 0.0
        
        # Q-Learning 업데이트
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # 메모리 캐시 업데이트
        self.q_table[(state_hash, action)] = new_q
        
        # DB 업데이트
        cursor.execute('''
            INSERT OR REPLACE INTO q_values 
            (state_hash, action, q_value, visit_count, last_update)
            VALUES (?, ?, ?, 
                    COALESCE((SELECT visit_count FROM q_values WHERE state_hash = ? AND action = ?), 0) + 1,
                    ?)
        ''', (state_hash, action, new_q, state_hash, action, time.time()))
        
        self.conn.commit()
    
    def choose_action(self, state: Dict, possible_actions: List[str]) -> str:
        """행동 선택 - ε-greedy 전략"""
        state_hash = state['hash']
        
        # 탐험 vs 활용
        if random.random() < self.epsilon:
            # 탐험: 랜덤 행동
            action = random.choice(possible_actions)
            print(f"🔍 탐험: {action} (ε={self.epsilon:.3f})")
        else:
            # 활용: 최고 Q값 행동
            q_values = []
            for action in possible_actions:
                q_val = self.get_q_value(state_hash, action)
                q_values.append((action, q_val))
            
            # Q값 기준 정렬
            q_values.sort(key=lambda x: x[1], reverse=True)
            best_action, best_q = q_values[0]
            
            action = best_action
            print(f"🧠 활용: {action} (Q={best_q:.2f})")
        
        # 탐험 확률 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return action
    
    def learn_experience(self, prev_state: Dict, action: str, new_state: Dict):
        """경험으로부터 학습"""
        # 보상 계산
        reward = self.calculate_reward(prev_state, action, new_state)
        
        # 경험 저장
        experience = Experience(
            state_hash=prev_state['hash'],
            action=action,
            next_state=new_state['hash'],
            reward=reward,
            timestamp=time.time()
        )
        self.experiences.append(experience)
        
        # Q값 업데이트
        self.update_q_value(prev_state['hash'], action, reward, new_state['hash'])
        
        # 통계 업데이트
        self.reward_history.append(reward)
        self.step_count += 1
        
        # DB에 경험 저장
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO experiences (state_hash, action, reward, next_state, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (experience.state_hash, experience.action, experience.reward,
              experience.next_state, experience.timestamp))
        
        self.conn.commit()
        
        return reward
    
    def get_learning_stats(self) -> Dict:
        """학습 통계"""
        cursor = self.conn.cursor()
        
        # 총 경험
        cursor.execute('SELECT COUNT(*) FROM experiences')
        total_exp = cursor.fetchone()[0]
        
        # 평균 보상
        recent_rewards = list(self.reward_history)[-100:]  # 최근 100개
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        # Q값 통계
        cursor.execute('SELECT COUNT(*) FROM q_values')
        q_entries = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(q_value) FROM q_values WHERE q_value > 0')
        avg_q = cursor.fetchone()[0] or 0
        
        # 발견한 상태 수
        cursor.execute('SELECT COUNT(DISTINCT state_hash) FROM experiences')
        states_discovered = cursor.fetchone()[0]
        
        return {
            'total_experiences': total_exp,
            'avg_reward': avg_reward,
            'q_table_size': q_entries,
            'avg_q_value': avg_q,
            'states_discovered': states_discovered,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }

class TurboBabyAI:
    """터보 Baby AI - 초고속 자율학습"""
    
    def __init__(self):
        self.controller = TurboController()
        self.vision = HyperVision()
        self.brain = ZeroBaseBrain()
        
        self.current_state = None
        self.running = False
        
        print("🚀 Turbo Baby AI 시동!")
        print("⚡ 학습 원리:")
        print("   🧠 Q-Learning (상태-행동 가치 학습)")
        print("   📊 SQLite RAG (경험 기반 기억)")
        print("   🎯 ε-greedy (탐험/활용 균형)")
        print("   💰 보상 시스템 (진행 상황 평가)")
        print("   🔄 실시간 패턴 업데이트")
    
    def initialize(self) -> bool:
        """초기화"""
        if not self.controller.find_dosbox_precise():
            print("❌ DOSBox를 찾을 수 없습니다!")
            return False
        
        if not self.vision.setup_vision(self.controller.dosbox_hwnd):
            print("❌ 시각 시스템 초기화 실패!")
            return False
        
        print("✅ Turbo Baby AI 준비 완료!")
        return True
    
    async def turbo_step(self) -> Dict:
        """초고속 학습 스텝"""
        try:
            # 1. 현재 상태 관찰
            new_state = self.vision.capture_state()
            if not new_state:
                return {'success': False, 'reason': 'vision_failed'}
            
            # 2. 행동 선택 (Q-Learning)
            action = self.brain.choose_action(new_state, self.controller.action_space)
            
            # 3. 행동 실행
            success = self.controller.send_key_direct(action)
            if not success:
                return {'success': False, 'reason': 'input_failed'}
            
            # 4. 결과 대기 및 관찰
            await asyncio.sleep(0.1)  # 게임 반응 대기
            
            result_state = self.vision.capture_state()
            if not result_state:
                return {'success': False, 'reason': 'result_vision_failed'}
            
            # 5. 경험 학습
            if self.current_state:
                reward = self.brain.learn_experience(self.current_state, action, result_state)
            else:
                reward = 0
            
            # 6. 상태 업데이트
            self.current_state = new_state
            
            return {
                'success': True,
                'action': action,
                'reward': reward,
                'state_hash': new_state['hash'],
                'is_new': new_state.get('is_new', False),
                'step': self.brain.step_count
            }
            
        except Exception as e:
            return {'success': False, 'reason': f'error: {e}'}

async def turbo_learning_session():
    """터보 학습 세션"""
    print("🚀 Turbo Baby AI 학습 시작!")
    print("=" * 60)
    
    ai = TurboBabyAI()
    
    if not ai.initialize():
        return
    
    print("\n⚡ 초고속 학습 모드 (0.15초 주기)")
    print("🎮 완전 자율 게임플레이 시작!\n")
    
    # 성능 모니터링
    start_time = time.time()
    success_count = 0
    total_steps = 0
    
    try:
        for step in range(1, 1001):  # 1000스텝 터보 학습
            result = await ai.turbo_step()
            total_steps += 1
            
            if result['success']:
                success_count += 1
                
                # 실시간 진행 상황 (10스텝마다)
                if step % 10 == 0:
                    stats = ai.brain.get_learning_stats()
                    elapsed = time.time() - start_time
                    sps = total_steps / elapsed  # Steps Per Second
                    
                    status = "🌟" if result.get('is_new') else "✅"
                    print(f"{status} #{step:4d} | {result['action']:8s} | "
                          f"R:{result['reward']:+5.1f} | "
                          f"Q:{stats['q_table_size']:3d} | "
                          f"ε:{stats['epsilon']:.3f} | "
                          f"{sps:.1f}sps")
                    
                    if result.get('is_new'):
                        print(f"        🎯 새 상태 발견: {result['state_hash']}")
            
            else:
                print(f"❌ #{step:4d} 실패: {result.get('reason', 'unknown')}")
            
            # 50스텝마다 상세 리포트
            if step % 50 == 0:
                stats = ai.brain.get_learning_stats()
                elapsed = time.time() - start_time
                
                print(f"\n📊 터보 학습 리포트 (스텝 {step}):")
                print(f"    ⚡ 속도: {total_steps/elapsed:.1f} 스텝/초")
                print(f"    ✅ 성공률: {success_count/total_steps:.2%}")
                print(f"    🧠 Q테이블: {stats['q_table_size']}개 상태-행동")
                print(f"    🎯 평균 보상: {stats['avg_reward']:+.2f}")
                print(f"    🗺️ 발견 상태: {stats['states_discovered']}개")
                print(f"    🔍 탐험률: {stats['epsilon']:.3f}")
                print()
            
            # 초고속 주기 (0.05초)
            await asyncio.sleep(0.05)
    
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자 중단 (스텝 {total_steps})")
    
    # 최종 성과 리포트
    final_stats = ai.brain.get_learning_stats()
    elapsed = time.time() - start_time
    
    print(f"\n🏁 터보 학습 완료!")
    print(f"⏱️ 소요시간: {elapsed:.1f}초")
    print(f"⚡ 평균속도: {total_steps/elapsed:.1f} 스텝/초")
    print(f"✅ 총 성공: {success_count}/{total_steps} ({success_count/total_steps:.1%})")
    print(f"🧠 학습 성과:")
    print(f"    📚 축적 경험: {final_stats['total_experiences']}개")
    print(f"    🎯 Q테이블: {final_stats['q_table_size']}개 엔트리")
    print(f"    🗺️ 탐험 영역: {final_stats['states_discovered']}개 상태")
    print(f"    💰 평균 보상: {final_stats['avg_reward']:+.2f}")
    
    if final_stats['avg_reward'] > 2.0:
        print("🏆 우수한 학습! AI가 게임 진행 패턴을 습득했어요!")
    elif final_stats['avg_reward'] > 0:
        print("👍 양호한 학습! 더 많은 경험으로 발전 가능해요!")
    else:
        print("🌱 초기 단계! 계속 학습하면 향상될 거예요!")

if __name__ == "__main__":
    try:
        asyncio.run(turbo_learning_session())
    except KeyboardInterrupt:
        print("\n👋 터보 학습 종료")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()