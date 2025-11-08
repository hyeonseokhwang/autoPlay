#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
강력한 메뉴 탈출 + 안정적인 키 입력 AI
"""

import asyncio
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import hashlib

# 게임 제어
import cv2
import pyautogui
import win32gui
import win32con
import win32api
import win32process

@dataclass
class GameScreen:
    hash_id: str
    screen_type: str
    brightness: float
    is_menu: bool
    is_dialogue: bool
    is_battle: bool
    timestamp: float

class RobustController:
    """영웅전설4 전용 강력한 컨트롤러"""
    
    def __init__(self):
        self.hero4_pid = None
        self.hero4_hwnd = None
        self.last_input = 0
        self.window_title = ""
        
    def find_and_lock_hero4_exclusive(self):
        """영웅전설4만 정확히 찾고 고정"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                try:
                    title = win32gui.GetWindowText(hwnd)
                    class_name = win32gui.GetClassName(hwnd)
                    
                    # 영웅전설4 전용 식별자
                    hero4_identifiers = [
                        'ed4' in title.lower(),
                        'legend' in title.lower() and 'hero' in title.lower(),
                        '영웅전설' in title,
                        'eiyuu' in title.lower(),
                        ('dosbox' in title.lower() and 
                         any(x in title.lower() for x in ['ed4', 'hero', 'legend'])),
                        (class_name == 'SDL_app' and 
                         any(x in title.lower() for x in ['dosbox', 'ed4']))
                    ]
                    
                    if any(hero4_identifiers):
                        try:
                            _, pid = win32process.GetWindowThreadProcessId(hwnd)
                            # 프로세스 검증
                            import psutil
                            process = psutil.Process(pid)
                            process_name = process.name().lower()
                            
                            # DOSBox 계열만 허용
                            if 'dosbox' in process_name or 'sdl' in process_name:
                                windows.append((hwnd, title, pid, class_name, process_name))
                        except:
                            # 프로세스 정보 없어도 타이틀로 허용
                            windows.append((hwnd, title, 0, class_name, 'unknown'))
                except:
                    pass
            return True
        
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if not windows:
            print("❌ 영웅전설4를 찾을 수 없습니다!")
            print("💡 DOSBox로 영웅전설4(ED4)를 실행한 후 다시 시도하세요.")
            return False
        
        # 가장 큰 창 선택 (메인 게임 창)
        best_window = None
        max_area = 0
        
        for hwnd, title, pid, class_name, process_name in windows:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                if area > max_area:
                    max_area = area
                    best_window = (hwnd, title, pid, class_name, process_name)
            except:
                continue
        
        if best_window:
            self.hero4_hwnd, self.window_title, self.hero4_pid, class_name, process_name = best_window
            
            print(f"🎯 영웅전설4 전용 연결!")
            print(f"   📝 게임: {self.window_title}")
            print(f"   🏷️ 클래스: {class_name}")
            print(f"   ⚙️ 프로세스: {process_name}")
            print(f"   🆔 PID: {self.hero4_pid}")
            print(f"   📐 창 크기: {max_area}px²")
            return True
        
        return False
    
    def force_key_input_to_hero4(self, key: str) -> bool:
        """영웅전설4에만 강제 키 입력"""
        if not self.hero4_hwnd:
            return False
        
        # 윈도우 유효성 검증
        try:
            if not win32gui.IsWindow(self.hero4_hwnd):
                print("⚠️ 영웅전설4 창이 닫혔습니다!")
                return False
        except:
            return False
        
        current_time = time.time()
        if current_time - self.last_input < 0.15:
            return False
        
        # 영웅전설4 전용 키 매핑
        hero4_keys = {
            'up': win32con.VK_UP, 'down': win32con.VK_DOWN,
            'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT,
            'enter': win32con.VK_RETURN, 'space': win32con.VK_SPACE,
            'esc': win32con.VK_ESCAPE, 'tab': win32con.VK_TAB,
            'z': ord('Z'), 'x': ord('X'), 'c': ord('C'),
            'a': ord('A'), 's': ord('S'), 'd': ord('D'),
            '1': ord('1'), '2': ord('2'), '3': ord('3'),
            'f1': win32con.VK_F1, 'f2': win32con.VK_F2, 'f10': win32con.VK_F10
        }
        
        if key.lower() not in hero4_keys:
            print(f"⚠️ 영웅전설4에서 지원하지 않는 키: {key}")
            return False
        
        vk_code = hero4_keys[key.lower()]
        success = False
        
        try:
            # 방법 1: 영웅전설4 창 강제 활성화
            try:
                # 창 상태 확인 및 복원
                if win32gui.IsIconic(self.hero4_hwnd):  # 최소화 상태면
                    win32gui.ShowWindow(self.hero4_hwnd, win32con.SW_RESTORE)
                
                # 최상위로 가져오기
                win32gui.BringWindowToTop(self.hero4_hwnd)
                win32gui.SetForegroundWindow(self.hero4_hwnd)
                time.sleep(0.05)
                success = True
                
                # 현재 활성 창 확인
                current_fg = win32gui.GetForegroundWindow()
                if current_fg == self.hero4_hwnd:
                    print(f"✅ 영웅전설4 활성화 성공")
                else:
                    print(f"⚠️ 다른 창이 활성화됨: {win32gui.GetWindowText(current_fg)}")
                
            except Exception as e:
                print(f"⚠️ 창 활성화 실패: {e}")
                success = False
            
            # 방법 2: 직접 키 이벤트 전송 (활성화 실패해도 시도)
            try:
                win32api.keybd_event(vk_code, 0, 0, 0)
                time.sleep(0.08)
                win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
                success = True
            except Exception as e:
                print(f"⚠️ 키 입력 오류 ({key}): {e}")
                success = False
            
            self.last_input = current_time
            
            if success:
                print(f"🎮 영웅전설4 키 입력: {key.upper()}")
            else:
                print(f"❌ 영웅전설4 키 입력 실패: {key.upper()}")
            
            return success
            
        except Exception as e:
            print(f"❌ 영웅전설4 키 입력 전체 실패: {e}")
            return False
    
    def verify_hero4_connection(self) -> bool:
        """영웅전설4 연결 상태 확인"""
        if not self.hero4_hwnd:
            return False
        
        try:
            is_valid = win32gui.IsWindow(self.hero4_hwnd) and win32gui.IsWindowVisible(self.hero4_hwnd)
            if not is_valid:
                print("⚠️ 영웅전설4 연결이 끊어졌습니다!")
            return is_valid
        except:
            return False

class AdvancedScreenAnalyzer:
    """영웅전설4 전용 고급 화면 분석기"""
    
    def __init__(self):
        self.hero4_capture_region = None
        self.screen_history = deque(maxlen=10)
        self.hero4_hwnd = None
        
    def setup_hero4_capture(self, hwnd):
        """영웅전설4 화면 캡처 설정"""
        try:
            self.hero4_hwnd = hwnd
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            
            # 영웅전설4 게임 영역만 정확히 캡처 (DOSBox 테두리 제외)
            border_x, border_y = 10, 35
            bottom_margin = 45
            
            self.hero4_capture_region = (
                x + border_x, 
                y + border_y, 
                x2 - x - border_x * 2, 
                y2 - y - border_y - bottom_margin
            )
            
            # 캡처 영역 검증
            if (self.hero4_capture_region[2] < 200 or 
                self.hero4_capture_region[3] < 150):
                print("⚠️ 캡처 영역이 너무 작습니다. 조정합니다.")
                self.hero4_capture_region = (x + 5, y + 30, x2 - x - 10, y2 - y - 40)
            
            print(f"📸 영웅전설4 캡처 영역: {self.hero4_capture_region}")
            print(f"📏 게임 화면: {self.hero4_capture_region[2]}×{self.hero4_capture_region[3]}px")
            return True
            
        except Exception as e:
            print(f"❌ 영웅전설4 캡처 설정 실패: {e}")
            return False
    
    def analyze_hero4_screen(self) -> Optional[GameScreen]:
        """영웅전설4 화면 전용 분석"""
        try:
            if not self.hero4_capture_region:
                return None
            
            # 영웅전설4 창 상태 확인
            if not win32gui.IsWindow(self.hero4_hwnd):
                print("⚠️ 영웅전설4 창이 사라졌습니다!")
                return None
            
            # 영웅전설4 화면 캡처
            screenshot = pyautogui.screenshot(region=self.hero4_capture_region)
            image = np.array(screenshot)
            
            # 이미지 전처리
            if image.shape[1] > 400:
                scale = 400 / image.shape[1]
                new_w = int(image.shape[1] * scale)
                new_h = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_w, new_h))
            
            # 기본 분석
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 화면 해시
            tiny = cv2.resize(gray, (16, 12))
            hash_id = hashlib.md5(tiny.tobytes()).hexdigest()[:8]
            
            # 밝기
            brightness = np.mean(gray)
            
            # 영웅전설4 화면 분류
            screen_type, is_menu, is_dialogue, is_battle = self._classify_hero4_screen(
                image, gray, hsv, brightness
            )
            
            screen = GameScreen(
                hash_id=hash_id,
                screen_type=screen_type,
                brightness=brightness,
                is_menu=is_menu,
                is_dialogue=is_dialogue,
                is_battle=is_battle,
                timestamp=time.time()
            )
            
            self.screen_history.append(screen)
            return screen
            
        except Exception as e:
            print(f"❌ 영웅전설4 화면 분석 오류: {e}")
            return None
    
    def _classify_hero4_screen(self, image, gray, hsv, brightness):
        """영웅전설4 화면 분류 (게임 특화)"""
        total_pixels = image.shape[0] * image.shape[1]
        
        # 색상 분석
        blue_mask = cv2.inRange(hsv, (100, 40, 40), (130, 255, 255))
        red_mask1 = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 40, 40), (180, 255, 255))
        
        blue_ratio = np.sum(blue_mask > 0) / total_pixels
        red_ratio = (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / total_pixels
        
        # 에지 분석
        edges = cv2.Canny(gray, 30, 100)
        edge_ratio = np.sum(edges > 0) / total_pixels
        
        # 밝은 영역 (텍스트)
        bright_mask = gray > 180
        bright_ratio = np.sum(bright_mask) / total_pixels
        
        # 영웅전설4 전용 분류 로직
        is_menu = False
        is_dialogue = False
        is_battle = False
        screen_type = 'field'
        
        # 영웅전설4 메뉴 감지 (최우선)
        if (60 < brightness < 95 and 
            (blue_ratio > 0.02 or edge_ratio > 0.2 or bright_ratio > 0.25)):
            is_menu = True
            screen_type = 'hero4_menu'
            print(f"🔵 영웅전설4 메뉴 감지! (밝기:{brightness:.1f}, 파랑:{blue_ratio:.3f})")
        
        # 영웅전설4 대화 감지
        elif bright_ratio > 0.15 and edge_ratio > 0.05:
            is_dialogue = True
            screen_type = 'hero4_dialogue'
        
        # 영웅전설4 전투 감지
        elif red_ratio > 0.05:
            is_battle = True
            screen_type = 'hero4_battle'
        
        # 영웅전설4 필드 판단
        elif brightness < 50:
            screen_type = 'hero4_dark_field'
        elif brightness > 100:
            screen_type = 'hero4_bright_field'
        else:
            screen_type = 'hero4_field'
        
        return screen_type, is_menu, is_dialogue, is_battle
    
    def is_screen_stuck(self) -> bool:
        """화면 막힘 감지"""
        if len(self.screen_history) < 5:
            return False
        
        recent_hashes = [s.hash_id for s in list(self.screen_history)[-5:]]
        return len(set(recent_hashes)) <= 2

class Hero4MenuEscapeStrategy:
    """영웅전설4 전용 메뉴 탈출 전략"""
    
    def __init__(self):
        self.escape_attempts = 0
        self.successful_escapes = []
        self.failed_attempts = []
        
        # 영웅전설4 전용 메뉴 탈출 전략들 (게임 분석 기반)
        self.hero4_escape_strategies = [
            ['esc'],                    # 1단계: ESC (가장 일반적)
            ['x'],                      # 2단계: X키 (취소)
            ['esc', 'esc'],             # 3단계: ESC 연타
            ['x', 'x'],                 # 4단계: X키 연타
            ['c'],                      # 5단계: C키 (캐릭터 정보 닫기)
            ['tab'],                    # 6단계: TAB (메뉴 전환)
            ['space'],                  # 7단계: 스페이스
            ['enter'],                  # 8단계: 엔터
            ['f10'],                    # 9단계: F10 (시스템 메뉴)
            ['esc', 'x'],               # 10단계: ESC + X 조합
            ['x', 'esc'],               # 11단계: X + ESC 조합
            ['f1'],                     # 12단계: F1 (도움말 토글)
            ['esc', 'esc', 'x'],        # 13단계: 강력한 조합
            ['1'],                      # 14단계: 숫자 1 (첫 번째 옵션)
            ['2'],                      # 15단계: 숫자 2 (두 번째 옵션)
        ]
        
        self.current_strategy = 0
        self.strategy_attempts = 0
    
    def get_next_hero4_escape_action(self) -> str:
        """영웅전설4 전용 다음 탈출 액션"""
        self.escape_attempts += 1
        
        # 성공한 전략이 있으면 우선 사용
        if self.successful_escapes:
            best_escape = max(set(self.successful_escapes), 
                            key=self.successful_escapes.count)
            if random.random() < 0.7:  # 70% 확률로 검증된 방법 사용
                print(f"🎯 영웅전설4 검증된 탈출법: {best_escape}")
                return best_escape
        
        # 전략 순차 시도
        if self.current_strategy < len(self.hero4_escape_strategies):
            strategy = self.hero4_escape_strategies[self.current_strategy]
            action = strategy[self.strategy_attempts % len(strategy)]
            
            self.strategy_attempts += 1
            
            # 한 전략을 2번 시도했으면 다음 전략으로 (더 빠른 전환)
            if self.strategy_attempts >= 2:
                self.current_strategy += 1
                self.strategy_attempts = 0
                print(f"🔄 영웅전설4 탈출 전략 변경: {self.current_strategy}/{len(self.hero4_escape_strategies)}")
            
            print(f"🚪 영웅전설4 탈출 시도 #{self.escape_attempts}: {action}")
            return action
        
        # 모든 전략 시도했으면 처음부터 다시
        self.current_strategy = 0
        self.strategy_attempts = 0
        print("🔄 영웅전설4 탈출 전략 초기화")
        return 'esc'
    
    def record_hero4_escape_result(self, action: str, success: bool):
        """영웅전설4 탈출 결과 기록"""
        if success:
            self.successful_escapes.append(action)
            print(f"✅ 영웅전설4 메뉴 탈출 성공! {action}")
            # 성공하면 전략 초기화
            self.current_strategy = 0
            self.strategy_attempts = 0
        else:
            self.failed_attempts.append(action)
    
    def get_hero4_escape_stats(self) -> Dict:
        """영웅전설4 탈출 통계"""
        success_count = len(self.successful_escapes)
        total_attempts = len(self.successful_escapes) + len(self.failed_attempts)
        
        return {
            'total_attempts': total_attempts,
            'successful_escapes': success_count,
            'success_rate': success_count / max(1, total_attempts),
            'best_escape_method': max(set(self.successful_escapes), 
                                    key=self.successful_escapes.count) if self.successful_escapes else None
        }

class Hero4SuperSmartAI:
    """영웅전설4 전용 슈퍼 스마트 AI"""
    
    def __init__(self):
        self.controller = RobustController()
        self.analyzer = AdvancedScreenAnalyzer()
        self.escape_strategy = Hero4MenuEscapeStrategy()
        
        self.last_screen = None
        self.cycle_count = 0
        self.successful_escapes = 0
        self.screen_changes = 0
        
    def initialize_hero4(self):
        """영웅전설4 전용 초기화"""
        print("🔍 영웅전설4 전용 연결 중...")
        if not self.controller.find_and_lock_hero4_exclusive():
            return False
        
        print("📸 영웅전설4 화면 분석 설정 중...")
        if not self.analyzer.setup_hero4_capture(self.controller.hero4_hwnd):
            return False
        
        print("🧠 영웅전설4 전용 슈퍼 스마트 AI 초기화 완료")
        print("💪 영웅전설4 특화 기능:")
        print("  - 15단계 영웅전설4 메뉴 탈출 전략")
        print("  - 영웅전설4 전용 키 입력 시스템")
        print("  - 영웅전설4 화면 상태 분류")
        print("  - 영웅전설4 창 독립 제어")
        return True
    
    def choose_hero4_action(self, screen: GameScreen) -> str:
        """영웅전설4 전용 액션 선택"""
        # 연결 상태 확인
        if not self.controller.verify_hero4_connection():
            print("⚠️ 영웅전설4 연결 끊어짐!")
            return 'esc'  # 안전한 기본 액션
        
        # 영웅전설4 메뉴 탈출이 최우선!
        if screen.is_menu:
            return self.escape_strategy.get_next_hero4_escape_action()
        
        # 영웅전설4 대화 진행
        elif screen.is_dialogue:
            return random.choice(['enter', 'space', 'z'])  # Z키 추가
        
        # 영웅전설4 전투 대응
        elif screen.is_battle:
            return random.choice(['enter', 'space', 'z', 'a', '1', '2'])  # 숫자키 추가
        
        # 영웅전설4 필드 탐험
        else:
            if self.analyzer.is_screen_stuck():
                # 막혔을 때는 영웅전설4 전용 시도
                return random.choice(['up', 'down', 'left', 'right', 'space', 'enter', 'esc', 'tab', 'c'])
            else:
                # 정상 영웅전설4 탐험
                return random.choice(['up', 'down', 'left', 'right', 'space', 'enter'])
    
    async def hero4_super_cycle(self) -> Dict:
        """영웅전설4 전용 슈퍼 사이클"""
        try:
            # 1. 영웅전설4 화면 분석
            current_screen = self.analyzer.analyze_hero4_screen()
            if not current_screen:
                return {'success': False, 'error': '영웅전설4 화면 분석 실패'}
            
            # 2. 영웅전설4 전용 액션 선택
            action = self.choose_hero4_action(current_screen)
            
            # 3. 영웅전설4에 액션 실행
            success = self.controller.force_key_input_to_hero4(action)
            
            # 4. 영웅전설4 반응 대기
            await asyncio.sleep(0.3)  # 영웅전설4에 최적화된 대기시간
            
            # 5. 영웅전설4 결과 확인
            result_screen = self.analyzer.analyze_hero4_screen()
            
            # 6. 영웅전설4 메뉴 탈출 성공 확인
            if (self.last_screen and self.last_screen.is_menu and 
                result_screen and not result_screen.is_menu):
                self.successful_escapes += 1
                self.escape_strategy.record_hero4_escape_result(action, True)
                print(f"🎉 영웅전설4 메뉴 탈출 성공! #{self.successful_escapes}")
            
            # 7. 화면 변화 추적
            if (self.last_screen and result_screen and 
                self.last_screen.hash_id != result_screen.hash_id):
                self.screen_changes += 1
            
            # 8. 상태 업데이트
            self.last_screen = current_screen
            self.cycle_count += 1
            
            return {
                'success': success,
                'action': action,
                'before_type': current_screen.screen_type,
                'after_type': result_screen.screen_type if result_screen else '?',
                'is_menu': current_screen.is_menu,
                'is_dialogue': current_screen.is_dialogue,
                'is_battle': current_screen.is_battle,
                'screen_changes': self.screen_changes,
                'escape_count': self.successful_escapes
            }
            
        except Exception as e:
            return {'success': False, 'error': f'영웅전설4 사이클 오류: {e}'}

async def main():
    """영웅전설4 전용 메인"""
    print("🚀 영웅전설4 전용 슈퍼 스마트 AI")
    print("=" * 60)
    print("🎮 영웅전설4에만 독립적으로 작동")
    
    ai = Hero4SuperSmartAI()
    
    if not ai.initialize():
        return
    
    print("\n💪 슈퍼 플레이 시작!")
    
    total_cycles = 150
    success_count = 0
    
    for cycle in range(1, total_cycles + 1):
        result = await ai.super_cycle()
        
        if result['success']:
            success_count += 1
            status = "✅"
        else:
            status = "❌"
        
        # 3사이클마다 리포트
        if cycle % 3 == 0:
            success_rate = success_count / cycle
            print(f"{status} #{cycle:3d} | {result.get('action', '?'):5s} | "
                  f"{result.get('before_type', '?'):12s} → {result.get('after_type', '?'):12s} | "
                  f"성공률:{success_rate:.2f}")
            
            # 상태 알림
            if result.get('is_menu'):
                print(f"      🔴 메뉴 상태")
            if result.get('is_dialogue'):
                print(f"      💬 대화 상태")
            if result.get('is_battle'):
                print(f"      ⚔️ 전투 상태")
        
        # 15사이클마다 진행 상황
        if cycle % 15 == 0:
            print(f"📊 진행 상황:")
            print(f"    성공률: {success_count/cycle:.2f}")
            print(f"    화면 변화: {result.get('screen_changes', 0)}회")
            print(f"    메뉴 탈출: {result.get('escape_count', 0)}회")
            print()
    
    # 최종 결과
    print(f"\n🏁 슈퍼 플레이 완료!")
    print(f"총 사이클: {total_cycles}")
    print(f"성공률: {success_count/total_cycles:.2f}")
    print(f"화면 변화: {ai.screen_changes}회")
    print(f"메뉴 탈출: {ai.successful_escapes}회")
    
    if ai.successful_escapes > 0:
        print(f"🏆 탈출 효율: {ai.screen_changes/total_cycles:.2f}")
    else:
        print("⚠️ 메뉴 탈출 실패 - 전략 재검토 필요")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()