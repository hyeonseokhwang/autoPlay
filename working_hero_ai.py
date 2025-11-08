#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제로 동작하는 영웅전설4 AI - 웹 학습 개선 버전
"""

import asyncio
import json
import sqlite3
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
    
    @property
    def success_rate(self):
        if self.total_count == 0:
            return 0.5
        return self.success_count / self.total_count

class GameVision:
    """게임 화면 분석 - 개선된 버전"""
    
    def __init__(self):
        self.window_title_keywords = ["DOSBox", "dosbox", "ED4"]
        
    def get_game_window(self):
        """DOSBox 윈도우 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if any(keyword in title for keyword in self.window_title_keywords):
                    windows.append(hwnd)
            return True
            
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        return windows[0] if windows else None
    
    def capture_game_screen(self) -> Optional[np.ndarray]:
        """게임 화면 캡처"""
        try:
            hwnd = self.get_game_window()
            if not hwnd:
                return None
                
            # 윈도우 영역 가져오기
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            width = x2 - x
            height = y2 - y
            
            # 화면 캡처
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            return np.array(screenshot)
            
        except Exception as e:
            print(f"❌ 화면 캡처 실패: {e}")
            return None
    
    def analyze_screen(self, image: np.ndarray) -> Dict:
        """화면 상태 분석"""
        if image is None:
            return {'screen_type': 'unknown', 'confidence': 0.0, 'details': {}}
        
        try:
            # 색상 분석
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 기본 통계
            height, width = gray.shape
            brightness_mean = np.mean(gray)
            
            # 메뉴 감지 (파란색 계열)
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            blue_ratio = np.sum(blue_mask > 0) / (width * height)
            
            # 전투 감지 (빨간색 계열)
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = red_mask1 + red_mask2
            red_ratio = np.sum(red_mask > 0) / (width * height)
            
            # 텍스트 박스 감지
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_boxes = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 20000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 2 < aspect_ratio < 10:  # 텍스트 박스 형태
                        text_boxes += 1
            
            # 화면 타입 결정
            screen_type = 'field'  # 기본값
            confidence = 0.5
            
            if blue_ratio > 0.15:
                screen_type = 'menu'
                confidence = min(0.9, blue_ratio * 5)
            elif red_ratio > 0.1:
                screen_type = 'battle'
                confidence = min(0.9, red_ratio * 8)
            elif text_boxes > 2:
                screen_type = 'dialogue'
                confidence = min(0.9, text_boxes * 0.2)
            elif brightness_mean < 30:
                screen_type = 'dark'
                confidence = 0.7
            
            return {
                'screen_type': screen_type,
                'confidence': confidence,
                'details': {
                    'brightness': brightness_mean,
                    'blue_ratio': blue_ratio,
                    'red_ratio': red_ratio, 
                    'text_boxes': text_boxes,
                    'size': (width, height)
                }
            }
            
        except Exception as e:
            print(f"❌ 화면 분석 실패: {e}")
            return {'screen_type': 'error', 'confidence': 0.0, 'details': {}}

class GameController:
    """게임 컨트롤러 - 개선된 키 입력"""
    
    def __init__(self):
        self.window_title_keywords = ["DOSBox", "dosbox", "ED4"]
        self.last_key_time = {}
        
    def get_game_window(self):
        """게임 윈도우 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if any(keyword in title for keyword in self.window_title_keywords):
                    windows.append(hwnd)
            return True
            
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        return windows[0] if windows else None
    
    def send_key_to_game(self, key: str) -> bool:
        """게임에 키 입력"""
        try:
            # 중복 입력 방지 (0.3초 간격)
            current_time = time.time()
            if key in self.last_key_time:
                if current_time - self.last_key_time[key] < 0.3:
                    return False
            
            self.last_key_time[key] = current_time
            
            # 게임 윈도우 활성화
            hwnd = self.get_game_window()
            if not hwnd:
                print("❌ 게임 윈도우를 찾을 수 없음")
                return False
            
            # 윈도우 포그라운드로
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.05)  # 활성화 대기
            
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
                'c': ord('C')
            }
            
            if key.lower() not in key_map:
                print(f"❌ 지원하지 않는 키: {key}")
                return False
            
            vk_code = key_map[key.lower()]
            
            # 키 입력 실행
            win32api.keybd_event(vk_code, 0, 0, 0)  # 키 누름
            time.sleep(0.08)  # 키 홀드
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  # 키 뗌
            
            print(f"✅ 키 입력 성공: {key.upper()}")
            return True
            
        except Exception as e:
            print(f"❌ 키 입력 실패 ({key}): {e}")
            return False

class StaticGameKnowledge:
    """정적 게임 지식 - 웹에서 얻기 어려운 기본 정보"""
    
    GAME_CONTROLS = {
        'movement': {
            'up': '캐릭터를 위로 이동',
            'down': '캐릭터를 아래로 이동',
            'left': '캐릭터를 왼쪽으로 이동',
            'right': '캐릭터를 오른쪽으로 이동'
        },
        'action': {
            'enter': '메뉴 확인, 대화 진행, 선택',
            'esc': '메뉴 취소, 뒤로 가기',
            'space': '조사, 액션',
            'z': '확인 (일부 버전)',
            'x': '취소 (일부 버전)'
        }
    }
    
    SCREEN_STRATEGIES = {
        'field': ['explore', 'move_random', 'search_items'],
        'menu': ['navigate_menu', 'select_item', 'confirm'],
        'dialogue': ['advance_dialogue', 'make_choice'],
        'battle': ['select_attack', 'use_skill', 'defend'],
        'dark': ['move_carefully', 'search_light']
    }
    
    @classmethod
    def get_actions_for_screen(cls, screen_type: str) -> List[str]:
        """화면 타입에 맞는 액션 목록"""
        return cls.SCREEN_STRATEGIES.get(screen_type, ['move_random'])
    
    @classmethod
    def get_control_info(cls, key: str) -> str:
        """키 설명"""
        for category, controls in cls.GAME_CONTROLS.items():
            if key in controls:
                return controls[key]
        return f"{key} 키 입력"

class SmartHeroAI:
    """영리한 영웅전설4 AI"""
    
    def __init__(self):
        self.vision = GameVision()
        self.controller = GameController()
        self.knowledge = StaticGameKnowledge()
        
        # 액션 정의
        self.actions = [
            GameAction("move_up", ["up"], "위로 이동"),
            GameAction("move_down", ["down"], "아래로 이동"), 
            GameAction("move_left", ["left"], "왼쪽으로 이동"),
            GameAction("move_right", ["right"], "오른쪽으로 이동"),
            GameAction("confirm", ["enter"], "확인/진행"),
            GameAction("cancel", ["esc"], "취소/뒤로"),
            GameAction("action", ["space"], "조사/액션"),
            GameAction("alt_confirm", ["z"], "확인 (Z키)"),
            GameAction("alt_cancel", ["x"], "취소 (X키)")
        ]
        
        # 상태 추적
        self.screen_history = []
        self.action_history = []
        self.stuck_counter = 0
        
    def choose_smart_action(self, screen_analysis: Dict) -> GameAction:
        """지능적 액션 선택"""
        try:
            screen_type = screen_analysis.get('screen_type', 'field')
            confidence = screen_analysis.get('confidence', 0.5)
            
            print(f"🔍 화면 분석: {screen_type} (신뢰도: {confidence:.2f})")
            
            # 화면 히스토리 업데이트
            self.screen_history.append(screen_type)
            if len(self.screen_history) > 10:
                self.screen_history.pop(0)
            
            # 반복 상황 감지
            if len(set(self.screen_history[-5:])) <= 1 and len(self.screen_history) >= 5:
                self.stuck_counter += 1
                print(f"⚠️ 반복 상황 감지 ({self.stuck_counter})")
            else:
                self.stuck_counter = 0
            
            # 화면 타입별 전략
            if screen_type == 'dialogue' or screen_analysis.get('details', {}).get('text_boxes', 0) > 2:
                # 대화 중 - 진행
                candidates = [a for a in self.actions if a.name in ['confirm', 'alt_confirm']]
                
            elif screen_type == 'menu':
                # 메뉴 화면 - 네비게이션
                if self.stuck_counter > 3:
                    candidates = [a for a in self.actions if a.name in ['cancel', 'esc']]
                else:
                    candidates = [a for a in self.actions if a.name in ['move_up', 'move_down', 'confirm']]
                    
            elif screen_type == 'battle':
                # 전투 화면 - 공격 액션
                candidates = [a for a in self.actions if a.name in ['confirm', 'move_up', 'move_down']]
                
            else:
                # 필드/탐험 - 이동 및 조사
                if self.stuck_counter > 5:
                    # 너무 오래 같은 화면 - 강제 이동
                    candidates = [a for a in self.actions if 'move' in a.name]
                else:
                    candidates = self.actions
            
            # 성공률 기반 가중치 적용
            if candidates:
                # 성공률이 높은 액션 우선 선택
                weights = []
                for action in candidates:
                    base_weight = action.success_rate if action.total_count > 0 else 0.5
                    # 최근 성공한 액션에 보너스
                    if len(self.action_history) > 0 and self.action_history[-1] == action.name:
                        base_weight *= 1.2
                    weights.append(max(0.1, base_weight))
                
                # 가중치 기반 랜덤 선택
                selected = random.choices(candidates, weights=weights)[0]
            else:
                # 후보가 없으면 랜덤
                selected = random.choice(self.actions)
            
            return selected
            
        except Exception as e:
            print(f"❌ 액션 선택 오류: {e}")
            return random.choice(self.actions)
    
    def execute_action(self, action: GameAction) -> bool:
        """액션 실행"""
        try:
            print(f"🎯 실행: {action.description}")
            
            success = True
            for key in action.keys:
                if not self.controller.send_key_to_game(key):
                    success = False
                    break
                time.sleep(0.1)  # 키 간격
            
            # 통계 업데이트
            action.total_count += 1
            if success:
                action.success_count += 1
                
            # 히스토리 업데이트
            self.action_history.append(action.name)
            if len(self.action_history) > 20:
                self.action_history.pop(0)
            
            return success
            
        except Exception as e:
            print(f"❌ 액션 실행 오류: {e}")
            return False
    
    async def play_step(self) -> Dict:
        """게임 1스텝 실행"""
        try:
            # 화면 캡처 및 분석
            screenshot = self.vision.capture_game_screen()
            if screenshot is None:
                return {'success': False, 'error': '화면 캡처 실패'}
            
            screen_analysis = self.vision.analyze_screen(screenshot)
            
            # 액션 선택
            action = self.choose_smart_action(screen_analysis)
            
            # 액션 실행
            success = self.execute_action(action)
            
            # 결과 대기
            await asyncio.sleep(1.2)  # 게임 반응 대기
            
            return {
                'success': success,
                'action': action.name,
                'screen_type': screen_analysis.get('screen_type'),
                'confidence': screen_analysis.get('confidence', 0.0),
                'success_rate': action.success_rate,
                'stuck_counter': self.stuck_counter
            }
            
        except Exception as e:
            print(f"❌ 플레이 스텝 오류: {e}")
            return {'success': False, 'error': str(e)}

async def main():
    """메인 실행 함수"""
    print("🎮 실전 영웅전설4 AI")
    print("=" * 40)
    
    ai = SmartHeroAI()
    
    print("🔍 게임 상태 확인...")
    
    # 초기 화면 확인
    screenshot = ai.vision.capture_game_screen()
    if screenshot is None:
        print("❌ 게임 화면을 찾을 수 없습니다. DOSBox가 실행 중인지 확인하세요.")
        return
    
    print("✅ 게임 화면 감지됨!")
    
    # 게임 플레이 시작
    total_steps = 50
    success_count = 0
    
    for step in range(1, total_steps + 1):
        print(f"\n--- 스텝 {step}/{total_steps} ---")
        
        result = await ai.play_step()
        
        if result['success']:
            success_count += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} 결과: {result.get('action', '?')} "
              f"(화면: {result.get('screen_type', '?')}, "
              f"성공률: {result.get('success_rate', 0):.2f})")
        
        # 전체 통계
        overall_success_rate = success_count / step
        print(f"📊 전체 성공률: {overall_success_rate:.2f} ({success_count}/{step})")
        
        # 막힘 감지
        if result.get('stuck_counter', 0) > 10:
            print("⚠️ 너무 오래 막혀있음. 전략 변경 필요.")
        
        # 잠깐 대기
        await asyncio.sleep(0.5)
    
    # 최종 결과
    print(f"\n🏁 게임 플레이 완료!")
    print(f"총 스텝: {total_steps}")
    print(f"성공: {success_count}")
    print(f"최종 성공률: {success_count/total_steps:.2f}")
    
    # 액션별 통계
    print(f"\n📈 액션별 성공률:")
    for action in ai.actions:
        if action.total_count > 0:
            print(f"  {action.description}: {action.success_rate:.2f} ({action.success_count}/{action.total_count})")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        traceback.print_exc()