#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 전용 자율학습 AI - 완전 독립형
제로베이스 강화학습으로 게임 마스터하기
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
import os

# 게임 제어
import cv2
import pyautogui
import win32gui
import win32con
import win32api
import win32process

@dataclass
class GameExperience:
    """게임 경험"""
    screen_state: str    # 화면 상태
    action: str         # 행동
    result_state: str   # 결과 상태
    reward: float       # 보상
    game_progress: float # 게임 진행도 추정
    timestamp: float    # 시간

class Hero4Controller:
    """영웅전설4 전용 컨트롤러 - 완전 독립형"""
    
    def __init__(self):
        self.hero4_hwnd = None
        self.window_title = ""
        self.is_connected = False
        
        # 영웅전설4 특화 액션 (게임 분석 기반)
        self.hero4_actions = [
            # === 기본 이동 ===
            'up', 'down', 'left', 'right',
            
            # === 메뉴/대화 ===
            'enter',     # 확인/대화
            'space',     # 확인/진행
            'esc',       # 취소/메뉴닫기
            
            # === 게임 특화 키 ===
            'z',         # 일반적으로 확인
            'x',         # 일반적으로 취소
            'c',         # 캐릭터 정보
            'a',         # 공격/액션
            's',         # 아이템/상태
            'd',         # 방어/대기
            
            # === 숫자 (메뉴 선택) ===
            '1', '2', '3', '4', '5',
            
            # === 기능키 (게임 시스템) ===
            'f1',        # 도움말
            'f2',        # 퀵세이브
            'f3',        # 퀵로드
            'f10',       # 시스템 메뉴
            
            # === 기타 ===
            'tab',       # 지도/정보
            'shift',     # 달리기
        ]
        
        print("🎮 영웅전설4 전용 컨트롤러 초기화")
        print(f"📋 게임 특화 액션: {len(self.hero4_actions)}개")
    
    def find_hero4_exclusive(self) -> bool:
        """영웅전설4 게임만 정확히 찾기"""
        
        def is_hero4_window(hwnd, title, class_name):
            """영웅전설4 윈도우 판별"""
            hero4_signatures = [
                # 타이틀 기반
                'ed4' in title.lower(),
                'legend' in title.lower() and 'hero' in title.lower(),
                '영웅전설' in title,
                'eiyuu' in title.lower(),
                
                # DOSBox + 게임명 조합
                'dosbox' in title.lower() and any(x in title.lower() for x in ['ed4', 'hero', 'legend']),
                
                # 클래스명 (DOSBox)
                class_name == 'SDL_app' and 'dosbox' in title.lower()
            ]
            return any(hero4_signatures)
        
        def enum_callback(hwnd, windows):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            
            try:
                title = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                
                if is_hero4_window(hwnd, title, class_name):
                    # 추가 검증: 프로세스 이름
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        import psutil
                        process = psutil.Process(pid)
                        process_name = process.name().lower()
                        
                        # DOSBox 계열 프로세스 확인
                        if 'dosbox' in process_name or 'sdl' in process_name:
                            windows.append((hwnd, title, class_name, process_name))
                    except:
                        # 프로세스 정보 없어도 윈도우 정보로 판별
                        windows.append((hwnd, title, class_name, 'unknown'))
                        
            except Exception as e:
                pass
            return True
        
        # 윈도우 검색
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if not windows:
            print("❌ 영웅전설4를 찾을 수 없습니다!")
            print("💡 DOSBox로 영웅전설4를 실행한 후 다시 시도하세요.")
            return False
        
        # 최적 윈도우 선택 (가장 큰 윈도우 우선)
        best_window = None
        max_area = 0
        
        for hwnd, title, class_name, process_name in windows:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                if area > max_area:
                    max_area = area
                    best_window = (hwnd, title, class_name, process_name)
            except:
                continue
        
        if best_window:
            self.hero4_hwnd, self.window_title, class_name, process_name = best_window
            self.is_connected = True
            
            print(f"🎯 영웅전설4 연결 성공!")
            print(f"   📝 타이틀: {self.window_title}")
            print(f"   🏷️ 클래스: {class_name}")  
            print(f"   ⚙️ 프로세스: {process_name}")
            print(f"   📐 크기: {max_area}px²")
            return True
        
        return False
    
    def send_game_input(self, action: str) -> bool:
        """영웅전설4에만 키 입력"""
        if not self.is_connected or not self.hero4_hwnd:
            return False
        
        try:
            # 윈도우 존재 확인
            if not win32gui.IsWindow(self.hero4_hwnd):
                self.is_connected = False
                return False
            
            # 게임 윈도우 활성화 (강제)
            try:
                current_fg = win32gui.GetForegroundWindow()
                if current_fg != self.hero4_hwnd:
                    win32gui.ShowWindow(self.hero4_hwnd, win32con.SW_RESTORE)
                    win32gui.BringWindowToTop(self.hero4_hwnd)
                    win32gui.SetForegroundWindow(self.hero4_hwnd)
                    time.sleep(0.02)  # 활성화 대기
            except:
                pass  # 활성화 실패해도 키 입력 시도
            
            # 키 매핑 (영웅전설4 최적화)
            hero4_keys = {
                # 이동 (가장 중요)
                'up': win32con.VK_UP, 'down': win32con.VK_DOWN,
                'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT,
                
                # 확인/진행
                'enter': win32con.VK_RETURN, 'space': win32con.VK_SPACE,
                
                # 게임 특화
                'esc': win32con.VK_ESCAPE, 'tab': win32con.VK_TAB,
                'z': ord('Z'), 'x': ord('X'), 'c': ord('C'),
                'a': ord('A'), 's': ord('S'), 'd': ord('D'),
                
                # 메뉴 선택
                '1': ord('1'), '2': ord('2'), '3': ord('3'),
                '4': ord('4'), '5': ord('5'),
                
                # 시스템
                'f1': win32con.VK_F1, 'f2': win32con.VK_F2,
                'f3': win32con.VK_F3, 'f10': win32con.VK_F10,
                
                # 보조
                'shift': win32con.VK_SHIFT
            }
            
            if action not in hero4_keys:
                return False
            
            vk_code = hero4_keys[action]
            
            # 키 입력 실행
            win32api.keybd_event(vk_code, 0, 0, 0)  # 누르기
            time.sleep(0.05)  # 게임 반응 시간 고려
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  # 떼기
            
            return True
            
        except Exception as e:
            print(f"⚠️ 키 입력 오류: {e}")
            return False
    
    def verify_connection(self) -> bool:
        """게임 연결 상태 확인"""
        if not self.hero4_hwnd:
            return False
        
        try:
            return win32gui.IsWindow(self.hero4_hwnd) and win32gui.IsWindowVisible(self.hero4_hwnd)
        except:
            return False

class Hero4Vision:
    """영웅전설4 전용 시각 시스템"""
    
    def __init__(self):
        self.game_region = None
        self.screen_history = deque(maxlen=100)
        self.state_cache = {}
        
        # 영웅전설4 화면 특징 (게임 분석 기반)
        self.screen_types = {
            'field_map': '필드맵',      # 야외 이동
            'town': '마을',            # 마을 내부
            'dungeon': '던전',         # 던전 탐험
            'battle': '전투',          # 전투 화면
            'menu': '메뉴',           # 각종 메뉴
            'dialogue': '대화',        # 대화/이벤트
            'shop': '상점',           # 상점 화면
            'inn': '여관',            # 여관/회복
            'status': '상태',         # 캐릭터 상태
            'inventory': '아이템',      # 아이템 관리
            'unknown': '미확인'        # 분류 안됨
        }
        
    def setup_hero4_vision(self, hwnd) -> bool:
        """영웅전설4 화면 영역 설정"""
        try:
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            
            # DOSBox 내부 게임 화면만 추출 (정확한 영역)
            # 영웅전설4는 일반적으로 640x480 해상도
            border_x, border_y = 10, 30  # DOSBox 테두리
            bottom_margin = 10
            
            self.game_region = (
                x + border_x,
                y + border_y, 
                x2 - x - border_x * 2,
                y2 - y - border_y - bottom_margin
            )
            
            # 첫 화면 캡처로 검증
            test_shot = pyautogui.screenshot(region=self.game_region)
            if test_shot.size[0] < 100 or test_shot.size[1] < 100:
                print("⚠️ 게임 영역이 너무 작습니다. 영역을 조정합니다.")
                self.game_region = (x + 5, y + 25, x2 - x - 10, y2 - y - 30)
            
            print(f"📸 영웅전설4 시각 영역: {self.game_region}")
            print(f"📏 게임 화면 크기: {self.game_region[2]}×{self.game_region[3]}px")
            
            return True
            
        except Exception as e:
            print(f"❌ 시각 시스템 설정 실패: {e}")
            return False
    
    def analyze_hero4_screen(self) -> Optional[Dict]:
        """영웅전설4 화면 분석 (게임 특화)"""
        try:
            # 게임 화면 캡처
            screenshot = pyautogui.screenshot(region=self.game_region)
            image = np.array(screenshot)
            
            if image.size == 0:
                return None
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 영웅전설4 특화 특징 추출
            h, w = gray.shape
            
            # 1. 화면 해시 (상태 식별)
            small = cv2.resize(gray, (20, 15))
            screen_hash = hashlib.md5(small.tobytes()).hexdigest()[:10]
            
            # 2. 게임 화면 영역별 분석
            regions = {
                'ui_top': np.mean(gray[:h//6, :]),           # 상단 UI
                'game_center': np.mean(gray[h//6:5*h//6, :]), # 게임 중앙
                'ui_bottom': np.mean(gray[5*h//6:, :]),       # 하단 UI/메뉴
                'left_panel': np.mean(gray[:, :w//5]),        # 좌측 패널
                'right_panel': np.mean(gray[:, 4*w//5:])      # 우측 패널
            }
            
            # 3. 색상 분석 (영웅전설4 특징)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 주요 색상 히스토그램
            color_features = {}
            for i, color in enumerate(['red', 'green', 'blue', 'yellow']):
                if i == 0:  # red
                    mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                elif i == 1:  # green  
                    mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
                elif i == 2:  # blue
                    mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
                else:  # yellow
                    mask = cv2.inRange(hsv, (20, 50, 50), (40, 255, 255))
                
                color_features[color] = np.sum(mask) / mask.size
            
            # 4. 게임 상태 추정
            screen_type = self.classify_hero4_screen(regions, color_features, gray)
            
            # 5. 진행도 추정 (화면 복잡도 기반)
            edges = cv2.Canny(gray, 50, 150)
            complexity = np.sum(edges > 0) / edges.size
            
            # 6. 상태 정보 구성
            screen_state = {
                'hash': screen_hash,
                'type': screen_type,
                'regions': regions,
                'colors': color_features,
                'complexity': complexity,
                'brightness': float(np.mean(gray)),
                'contrast': float(np.std(gray)),
                'timestamp': time.time(),
                'size': image.shape[:2]
            }
            
            # 7. 새로운 화면인지 확인
            is_new = screen_hash not in self.state_cache
            if is_new:
                self.state_cache[screen_hash] = {
                    'first_seen': time.time(),
                    'type': screen_type,
                    'visit_count': 0
                }
                print(f"🆕 새로운 {screen_type} 화면: {screen_hash}")
            
            cache_info = self.state_cache[screen_hash]
            cache_info['visit_count'] += 1
            cache_info['last_visit'] = time.time()
            
            screen_state['is_new'] = is_new
            screen_state['visit_count'] = cache_info['visit_count']
            screen_state['familiarity'] = 1.0 / max(1, cache_info['visit_count'])
            
            # 8. 히스토리 저장
            self.screen_history.append(screen_state)
            
            return screen_state
            
        except Exception as e:
            print(f"⚠️ 화면 분석 오류: {e}")
            return None
    
    def classify_hero4_screen(self, regions: Dict, colors: Dict, gray: np.ndarray) -> str:
        """영웅전설4 화면 타입 분류"""
        
        # 간단한 규칙 기반 분류 (게임 분석 기반)
        
        # 전투 화면 (복잡하고 UI가 많음)
        if regions['ui_bottom'] > 100 and regions['ui_top'] > 80:
            if colors['red'] > 0.05:  # 빨간색 많음 (HP/데미지)
                return 'battle'
        
        # 대화 화면 (하단에 텍스트박스)
        if regions['ui_bottom'] > 150 and regions['game_center'] < 100:
            return 'dialogue'
        
        # 메뉴 화면 (전체적으로 밝고 구조적)
        if abs(regions['left_panel'] - regions['right_panel']) > 30:
            if regions['ui_top'] > 120:
                return 'menu'
        
        # 던전 (어둡고 복잡)
        if regions['game_center'] < 80 and np.std(gray) > 40:
            return 'dungeon'
        
        # 마을 (적당히 밝고 안정적)
        if 90 < regions['game_center'] < 140 and colors['green'] > 0.03:
            return 'town'
        
        # 필드맵 (넓고 초록색 많음)
        if colors['green'] > 0.08 and regions['game_center'] > 100:
            return 'field_map'
        
        # 상점 (UI 패턴)
        if colors['yellow'] > 0.04 and regions['ui_bottom'] > 120:
            return 'shop'
        
        return 'unknown'

class Hero4Brain:
    """영웅전설4 전용 학습 뇌"""
    
    def __init__(self):
        # 게임 특화 메모리
        self.db_path = 'hero4_brain.db'
        self.conn = sqlite3.connect(self.db_path)
        self.setup_hero4_memory()
        
        # Q-Learning 파라미터 (게임 특화 튜닝)
        self.learning_rate = 0.2      # 게임은 천천히 학습
        self.discount_factor = 0.95   # 장기 전략 중시
        self.epsilon = 0.8           # 탐험 중시 (게임은 다양성 필요)
        self.epsilon_decay = 0.998   # 천천히 감소
        self.epsilon_min = 0.1       # 최소 탐험 유지
        
        # 게임 진행 추적
        self.game_progress_indicators = {
            'new_screens': 0,      # 새로운 화면 발견
            'battle_count': 0,     # 전투 횟수
            'town_visits': 0,      # 마을 방문
            'dialogue_count': 0,   # 대화 횟수
            'menu_usage': 0        # 메뉴 사용
        }
        
        # 실시간 학습 데이터
        self.q_cache = {}
        self.recent_rewards = deque(maxlen=50)
        
        print("🧠 영웅전설4 전용 학습뇌 초기화")
        print("📚 게임 진행 추적 시스템 활성화")
    
    def setup_hero4_memory(self):
        """영웅전설4 전용 메모리 구조"""
        cursor = self.conn.cursor()
        
        # 게임 상태별 Q값 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hero4_q_values (
                screen_hash TEXT,
                screen_type TEXT,
                action TEXT,
                q_value REAL,
                success_count INTEGER,
                total_count INTEGER,
                last_reward REAL,
                last_update REAL,
                PRIMARY KEY (screen_hash, action)
            )
        ''')
        
        # 게임 진행 기록
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hero4_progress (
                session_id TEXT,
                timestamp REAL,
                screen_type TEXT,
                action TEXT,
                reward REAL,
                game_progress_score REAL
            )
        ''')
        
        # 게임 패턴 학습
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hero4_patterns (
                pattern_name TEXT,
                trigger_condition TEXT,
                recommended_action TEXT,
                success_rate REAL,
                discovery_time REAL,
                usage_count INTEGER
            )
        ''')
        
        self.conn.commit()
        print("💾 영웅전설4 전용 메모리 구조 완료")
    
    def calculate_hero4_reward(self, prev_state: Dict, action: str, new_state: Dict) -> float:
        """영웅전설4 전용 보상 계산"""
        if not new_state:
            return -5.0  # 실패 큰 페널티
        
        reward = 0.0
        
        # === 기본 탐험 보상 ===
        if new_state.get('is_new', False):
            reward += 15.0  # 새로운 발견 보상 증가
            self.game_progress_indicators['new_screens'] += 1
            print(f"🌟 새 세계 발견: +15.0")
        
        # === 게임 타입별 특화 보상 ===
        prev_type = prev_state.get('type', 'unknown')
        new_type = new_state.get('type', 'unknown')
        
        # 게임 진행 보상
        if prev_type != new_type:
            type_rewards = {
                'battle': 8.0,      # 전투 진입 (중요한 게임 요소)
                'dialogue': 6.0,    # 스토리 진행
                'town': 4.0,        # 새로운 지역
                'field_map': 3.0,   # 탐험
                'shop': 5.0,        # 아이템 관리
                'menu': 2.0,        # 시스템 사용
                'dungeon': 7.0      # 던전 탐험
            }
            
            if new_type in type_rewards:
                type_reward = type_rewards[new_type]
                reward += type_reward
                print(f"🎮 {new_type} 진입: +{type_reward}")
                
                # 진행 카운터 업데이트
                if new_type == 'battle':
                    self.game_progress_indicators['battle_count'] += 1
                elif new_type == 'dialogue':
                    self.game_progress_indicators['dialogue_count'] += 1
                elif new_type == 'town':
                    self.game_progress_indicators['town_visits'] += 1
        
        # === 화면 변화 보상 ===
        if prev_state['hash'] != new_state['hash']:
            reward += 3.0
            
        # === 복잡도 변화 보상 (게임 진행 의미) ===
        complexity_change = abs(prev_state.get('complexity', 0) - new_state.get('complexity', 0))
        if complexity_change > 0.1:
            reward += min(complexity_change * 5.0, 4.0)
        
        # === 색상 변화 보상 (화면 전환) ===
        color_changes = 0
        for color in ['red', 'green', 'blue', 'yellow']:
            prev_color = prev_state.get('colors', {}).get(color, 0)
            new_color = new_state.get('colors', {}).get(color, 0)
            if abs(prev_color - new_color) > 0.02:
                color_changes += 1
        
        if color_changes > 0:
            reward += color_changes * 1.5
        
        # === 방문 빈도 보상 (탐험 장려) ===
        familiarity = new_state.get('familiarity', 0.5)
        reward += familiarity * 2.0
        
        # === 정체 페널티 ===
        visit_count = new_state.get('visit_count', 1)
        if visit_count > 15:  # 같은 곳에 너무 많이 방문
            reward -= min(visit_count - 15, 3.0)
        
        # === 행동별 조정 ===
        action_adjustments = {
            'up': 0.1, 'down': 0.1, 'left': 0.1, 'right': 0.1,  # 이동 약간 보상
            'enter': 0.5, 'space': 0.5,  # 진행 행동 보상
            'esc': -0.2,  # 취소는 약간 페널티 (하지만 필요할 때가 있음)
        }
        reward += action_adjustments.get(action, 0)
        
        return reward
    
    def get_hero4_q_value(self, state_hash: str, action: str) -> float:
        """Q값 조회 (캐시 우선)"""
        cache_key = (state_hash, action)
        if cache_key in self.q_cache:
            return self.q_cache[cache_key]
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT q_value FROM hero4_q_values WHERE screen_hash = ? AND action = ?',
                      (state_hash, action))
        result = cursor.fetchone()
        
        q_val = result[0] if result else 0.0
        self.q_cache[cache_key] = q_val
        return q_val
    
    def update_hero4_q_value(self, state_hash: str, screen_type: str, action: str, 
                           reward: float, next_state_hash: str):
        """영웅전설4 특화 Q값 업데이트"""
        
        # 현재 Q값
        current_q = self.get_hero4_q_value(state_hash, action)
        
        # 다음 상태의 최대 Q값
        cursor = self.conn.cursor()
        cursor.execute('SELECT MAX(q_value) FROM hero4_q_values WHERE screen_hash = ?',
                      (next_state_hash,))
        result = cursor.fetchone()
        max_next_q = result[0] if result and result[0] else 0.0
        
        # Q-Learning 업데이트
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # 캐시 업데이트
        self.q_cache[(state_hash, action)] = new_q
        
        # DB 업데이트
        cursor.execute('''
            INSERT OR REPLACE INTO hero4_q_values 
            (screen_hash, screen_type, action, q_value, success_count, total_count, last_reward, last_update)
            VALUES (?, ?, ?, ?, 
                    COALESCE((SELECT success_count FROM hero4_q_values WHERE screen_hash = ? AND action = ?), 0) + ?,
                    COALESCE((SELECT total_count FROM hero4_q_values WHERE screen_hash = ? AND action = ?), 0) + 1,
                    ?, ?)
        ''', (state_hash, screen_type, action, new_q, state_hash, action, 
              1 if reward > 0 else 0, state_hash, action, reward, time.time()))
        
        self.conn.commit()
    
    def choose_hero4_action(self, state: Dict, available_actions: List[str]) -> str:
        """영웅전설4 특화 행동 선택"""
        state_hash = state['hash']
        screen_type = state.get('type', 'unknown')
        
        # 화면 타입별 선호 행동 (게임 지식 기반)
        type_preferences = {
            'battle': ['z', 'x', 'a', 's', 'd', '1', '2', '3'],  # 전투 행동
            'dialogue': ['enter', 'space', 'z'],                  # 대화 진행
            'menu': ['up', 'down', 'enter', 'esc', '1', '2'],    # 메뉴 조작
            'field_map': ['up', 'down', 'left', 'right', 'enter'], # 이동
            'town': ['up', 'down', 'left', 'right', 'enter', 'c'], # 마을 탐험
            'shop': ['up', 'down', 'enter', 'esc', '1', '2'],     # 상점 이용
            'dungeon': ['up', 'down', 'left', 'right', 'f2'],     # 던전 + 저장
            'unknown': available_actions                           # 모든 행동
        }
        
        preferred_actions = type_preferences.get(screen_type, available_actions)
        
        # ε-greedy with 화면 타입 고려
        if random.random() < self.epsilon:
            # 탐험: 화면 타입에 맞는 행동 우선
            if preferred_actions:
                action = random.choice(preferred_actions)
                print(f"🔍 {screen_type} 탐험: {action} (ε={self.epsilon:.3f})")
            else:
                action = random.choice(available_actions)
                print(f"🔍 일반 탐험: {action}")
        else:
            # 활용: Q값 기반 최적 행동
            q_values = []
            for action in available_actions:
                q_val = self.get_hero4_q_value(state_hash, action)
                # 화면 타입 선호도 보너스
                type_bonus = 0.5 if action in preferred_actions else 0
                adjusted_q = q_val + type_bonus
                q_values.append((action, adjusted_q, q_val))
            
            # 최고 Q값 선택
            q_values.sort(key=lambda x: x[1], reverse=True)
            best_action, adjusted_q, original_q = q_values[0]
            action = best_action
            print(f"🧠 {screen_type} 활용: {action} (Q={original_q:.2f}+{adjusted_q-original_q:.1f})")
        
        # 탐험률 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return action
    
    def learn_hero4_experience(self, prev_state: Dict, action: str, new_state: Dict):
        """영웅전설4 경험 학습"""
        # 보상 계산
        reward = self.calculate_hero4_reward(prev_state, action, new_state)
        
        # Q값 업데이트
        self.update_hero4_q_value(
            prev_state['hash'], 
            prev_state.get('type', 'unknown'), 
            action, 
            reward, 
            new_state['hash']
        )
        
        # 통계 업데이트
        self.recent_rewards.append(reward)
        
        # 진행 기록 저장
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO hero4_progress 
            (session_id, timestamp, screen_type, action, reward, game_progress_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (f"session_{int(time.time()//3600)}", time.time(), 
              new_state.get('type', 'unknown'), action, reward,
              sum(self.game_progress_indicators.values())))
        
        self.conn.commit()
        return reward
    
    def get_hero4_stats(self) -> Dict:
        """영웅전설4 학습 통계"""
        cursor = self.conn.cursor()
        
        # 총 학습 데이터
        cursor.execute('SELECT COUNT(*) FROM hero4_q_values')
        q_entries = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM hero4_progress')
        total_actions = cursor.fetchone()[0]
        
        # 평균 보상
        recent_avg = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0
        
        # 화면 타입별 통계
        cursor.execute('''SELECT screen_type, COUNT(*) FROM hero4_progress 
                         GROUP BY screen_type ORDER BY COUNT(*) DESC''')
        screen_stats = dict(cursor.fetchall())
        
        return {
            'q_table_size': q_entries,
            'total_actions': total_actions,
            'avg_reward': recent_avg,
            'epsilon': self.epsilon,
            'game_progress': self.game_progress_indicators.copy(),
            'screen_distribution': screen_stats
        }

class Hero4AI:
    """영웅전설4 전용 자율학습 AI"""
    
    def __init__(self):
        self.controller = Hero4Controller()
        self.vision = Hero4Vision()
        self.brain = Hero4Brain()
        
        self.current_state = None
        self.session_start = time.time()
        
        print("🎮 영웅전설4 전용 AI 시동!")
        print("🤖 완전 자율 게임플레이 시스템")
    
    def initialize(self) -> bool:
        """시스템 초기화"""
        print("\n🔍 영웅전설4 찾는 중...")
        if not self.controller.find_hero4_exclusive():
            return False
        
        print("📸 게임 시각 시스템 설정 중...")
        if not self.vision.setup_hero4_vision(self.controller.hero4_hwnd):
            return False
        
        print("✅ 영웅전설4 AI 준비 완료!\n")
        return True
    
    async def play_hero4_step(self) -> Dict:
        """영웅전설4 게임 스텝"""
        try:
            # 1. 게임 연결 확인
            if not self.controller.verify_connection():
                return {'success': False, 'reason': 'game_disconnected'}
            
            # 2. 현재 화면 분석
            new_state = self.vision.analyze_hero4_screen()
            if not new_state:
                return {'success': False, 'reason': 'vision_failed'}
            
            # 3. 행동 선택
            action = self.brain.choose_hero4_action(new_state, self.controller.hero4_actions)
            
            # 4. 게임에 입력
            success = self.controller.send_game_input(action)
            if not success:
                return {'success': False, 'reason': 'input_failed'}
            
            # 5. 게임 반응 대기
            await asyncio.sleep(0.15)  # 게임 반응 시간
            
            # 6. 결과 관찰
            result_state = self.vision.analyze_hero4_screen()
            if not result_state:
                return {'success': False, 'reason': 'result_vision_failed'}
            
            # 7. 학습
            reward = 0
            if self.current_state:
                reward = self.brain.learn_hero4_experience(self.current_state, action, result_state)
            
            # 8. 상태 업데이트
            self.current_state = new_state
            
            return {
                'success': True,
                'action': action,
                'reward': reward,
                'screen_type': new_state.get('type', 'unknown'),
                'is_new': new_state.get('is_new', False)
            }
            
        except Exception as e:
            return {'success': False, 'reason': f'error: {e}'}

async def play_hero4_autonomous():
    """영웅전설4 자율 플레이 세션"""
    print("🎮 영웅전설4 자율학습 AI")
    print("=" * 50)
    
    ai = Hero4AI()
    
    if not ai.initialize():
        return
    
    print("🚀 영웅전설4 자율 플레이 시작!")
    print("🎯 목표: 게임을 스스로 학습하며 진행하기")
    print("⏱️ 0.2초 주기로 게임 플레이 중...\n")
    
    # 성능 추적
    start_time = time.time()
    step_count = 0
    success_count = 0
    
    try:
        for step in range(1, 501):  # 500스텝 플레이
            result = await ai.play_hero4_step()
            step_count += 1
            
            if result['success']:
                success_count += 1
                
                # 진행상황 출력 (5스텝마다)
                if step % 5 == 0:
                    elapsed = time.time() - start_time
                    sps = step / elapsed
                    
                    status = "🆕" if result.get('is_new') else "✅"
                    print(f"{status} #{step:3d} | {result['screen_type']:8s} | "
                          f"{result['action']:6s} | R:{result['reward']:+5.1f} | "
                          f"{sps:.1f}sps")
                    
                    if result.get('is_new'):
                        print(f"        🌟 새로운 {result['screen_type']} 발견!")
            else:
                print(f"❌ #{step:3d} 실패: {result.get('reason', 'unknown')}")
            
            # 25스텝마다 상세 리포트
            if step % 25 == 0:
                stats = ai.brain.get_hero4_stats()
                elapsed = time.time() - start_time
                
                print(f"\n📊 영웅전설4 플레이 리포트 (스텝 {step}):")
                print(f"    ⚡ 플레이 속도: {step/elapsed:.1f} 액션/초")
                print(f"    ✅ 성공률: {success_count/step:.1%}")
                print(f"    🧠 학습된 패턴: {stats['q_table_size']}개")
                print(f"    🎯 평균 보상: {stats['avg_reward']:+.2f}")
                print(f"    🎮 게임 진행:")
                for key, value in stats['game_progress'].items():
                    if value > 0:
                        print(f"        {key}: {value}")
                print(f"    🔍 탐험률: {stats['epsilon']:.3f}")
                print()
            
            # 게임 속도에 맞춘 대기
            await asyncio.sleep(0.2)
    
    except KeyboardInterrupt:
        print(f"\n⏹️ 플레이 중단 (스텝 {step_count})")
    
    # 최종 성과
    final_stats = ai.brain.get_hero4_stats()
    elapsed = time.time() - start_time
    
    print(f"\n🏁 영웅전설4 플레이 완료!")
    print(f"⏱️ 플레이 시간: {elapsed/60:.1f}분")
    print(f"🎮 총 액션: {step_count}개")
    print(f"✅ 성공률: {success_count/step_count:.1%}")
    print(f"🧠 최종 학습 성과:")
    print(f"    📚 Q테이블: {final_stats['q_table_size']}개 패턴")
    print(f"    💰 평균 보상: {final_stats['avg_reward']:+.2f}")
    print(f"    🎯 게임 진행도:")
    
    for key, value in final_stats['game_progress'].items():
        if value > 0:
            print(f"        📈 {key}: {value}회")
    
    print(f"\n🎓 AI가 배운 화면들:")
    for screen_type, count in final_stats['screen_distribution'].items():
        print(f"    🖥️ {screen_type}: {count}회 경험")
    
    if final_stats['avg_reward'] > 3.0:
        print("\n🏆 뛰어난 성과! AI가 게임을 잘 이해하고 있어요!")
    elif final_stats['avg_reward'] > 1.0:
        print("\n👍 좋은 진전! 계속 학습하면 더 나아질 거예요!")
    else:
        print("\n🌱 학습 초기 단계! 더 많은 경험이 필요해요!")

if __name__ == "__main__":
    try:
        asyncio.run(play_hero4_autonomous())
    except KeyboardInterrupt:
        print("\n👋 영웅전설4 AI 종료")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()