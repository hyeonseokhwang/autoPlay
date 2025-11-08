#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 DOSBox 창 화면 안으로 이동 + 게임 실행
"""

import time
import win32gui
import win32con
import win32api
from PIL import ImageGrab
import numpy as np

class Hero4GameStarter:
    """영웅전설4 게임 시작 도우미"""
    
    def __init__(self):
        """초기화"""
        self.dosbox_window = None
        self.launcher_window = None
        
    def find_windows(self) -> bool:
        """게임 관련 창들 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                
                if 'dosbox' in window_text.lower() or 'ED4' in window_text:
                    self.dosbox_window = hwnd
                    print(f"📦 DOSBox 창 발견: {window_text}")
                    
                elif '게임런처' in window_text or 'launcher' in window_text.lower():
                    self.launcher_window = hwnd
                    print(f"🚀 런처 창 발견: {window_text}")
            return True

        win32gui.EnumWindows(enum_callback, None)
        
        return self.dosbox_window is not None
    
    def move_dosbox_to_center(self) -> bool:
        """DOSBox 창을 화면 중앙으로 이동"""
        if not self.dosbox_window:
            print("❌ DOSBox 창을 찾을 수 없습니다!")
            return False
        
        try:
            # 현재 위치 확인
            current_rect = win32gui.GetWindowRect(self.dosbox_window)
            print(f"📍 현재 DOSBox 위치: {current_rect}")
            
            # 화면 중앙 계산 (1920x1080 기준)
            new_x, new_y = 300, 200  # 화면 안쪽으로
            width = current_rect[2] - current_rect[0]
            height = current_rect[3] - current_rect[1]
            
            print(f"🎯 이동 목표: ({new_x}, {new_y}) 크기: {width}x{height}")
            
            # 창 이동
            win32gui.SetWindowPos(
                self.dosbox_window,
                win32con.HWND_TOP,
                new_x, new_y, width, height,
                win32con.SWP_SHOWWINDOW
            )
            
            # 활성화
            win32gui.SetForegroundWindow(self.dosbox_window)
            win32gui.ShowWindow(self.dosbox_window, win32con.SW_RESTORE)
            
            time.sleep(2)
            
            # 새 위치 확인
            new_rect = win32gui.GetWindowRect(self.dosbox_window)
            print(f"✅ 새 DOSBox 위치: {new_rect}")
            
            return True
            
        except Exception as e:
            print(f"❌ 창 이동 실패: {e}")
            return False
    
    def test_dosbox_capture(self) -> bool:
        """DOSBox 화면 캡처 테스트"""
        if not self.dosbox_window:
            return False
            
        try:
            rect = win32gui.GetWindowRect(self.dosbox_window)
            screenshot = ImageGrab.grab(rect)
            image_array = np.array(screenshot)
            
            brightness = np.mean(image_array)
            print(f"🔆 DOSBox 화면 밝기: {brightness}")
            
            # 이미지 저장
            screenshot.save(f'dosbox_test_{int(time.time())}.png')
            print(f"💾 DOSBox 화면 저장됨")
            
            if brightness > 10:
                print("✅ DOSBox 화면이 정상적으로 보입니다!")
                return True
            else:
                print("⚠️ DOSBox 화면이 여전히 검습니다.")
                return False
                
        except Exception as e:
            print(f"❌ 캡처 테스트 실패: {e}")
            return False
    
    def send_keys_to_dosbox(self, keys: list) -> None:
        """DOSBox에 키 시퀀스 전송"""
        if not self.dosbox_window:
            print("❌ DOSBox 창이 없습니다!")
            return
            
        # DOSBox 활성화
        win32gui.SetForegroundWindow(self.dosbox_window)
        time.sleep(0.5)
        
        print(f"⌨️ DOSBox에 키 입력 중...")
        
        key_map = {
            'enter': 0x0D, 'space': 0x20, 'esc': 0x1B,
            'left': 0x25, 'right': 0x27, 'up': 0x26, 'down': 0x28,
            'z': 0x5A, 'x': 0x58, 'a': 0x41, 's': 0x53,
            '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34
        }
        
        for key in keys:
            if key in key_map:
                vk_code = key_map[key]
                
                # 키 누르기
                win32api.keybd_event(vk_code, 0, 0, 0)
                time.sleep(0.1)
                win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
                
                print(f"   🔑 키 입력: {key.upper()}")
                time.sleep(0.3)
            else:
                print(f"   ❌ 알 수 없는 키: {key}")
    
    def try_start_game(self) -> None:
        """게임 시작 시도"""
        print("🎮 게임 시작 시퀀스 실행...")
        
        # 일반적인 게임 시작 키들
        start_sequences = [
            ['enter', 'enter'],           # 엔터 연속
            ['space', 'space'],           # 스페이스 연속
            ['z', 'enter'],               # Z + 엔터
            ['1', 'enter'],               # 1 + 엔터  
            ['enter', '1', 'enter'],      # 엔터, 1, 엔터
            ['esc', 'enter', 'enter'],    # ESC로 메뉴, 엔터들
        ]
        
        for i, sequence in enumerate(start_sequences):
            print(f"🔄 시퀀스 #{i+1}: {sequence}")
            self.send_keys_to_dosbox(sequence)
            
            time.sleep(2)
            
            # 화면 변화 확인
            if self.test_dosbox_capture():
                print("✅ 게임이 시작된 것 같습니다!")
                return
            else:
                print("⏳ 아직 검은 화면...")
        
        print("😅 모든 시퀀스 시도했지만 게임이 시작되지 않았습니다.")
    
    def run_full_setup(self) -> bool:
        """전체 설정 실행"""
        print("🚀 영웅전설4 게임 시작 도우미")
        print("=" * 50)
        
        # 1. 창 찾기
        print("1️⃣ 게임 창 찾는 중...")
        if not self.find_windows():
            print("❌ DOSBox 창을 찾을 수 없습니다!")
            print("💡 게임런처에서 게임을 실행해주세요!")
            return False
        
        # 2. DOSBox 창 이동
        print("\n2️⃣ DOSBox 창을 화면 안으로 이동...")
        if not self.move_dosbox_to_center():
            return False
        
        # 3. 초기 화면 테스트
        print("\n3️⃣ 초기 화면 테스트...")
        initial_capture = self.test_dosbox_capture()
        
        # 4. 게임 시작 시도
        if not initial_capture:
            print("\n4️⃣ 게임 시작 시도...")
            self.try_start_game()
        
        # 5. 최종 확인
        print("\n5️⃣ 최종 화면 확인...")
        final_result = self.test_dosbox_capture()
        
        if final_result:
            print("🎉 성공! 게임이 실행 중입니다!")
            return True
        else:
            print("😞 게임 화면을 활성화하지 못했습니다.")
            print("💡 수동으로 게임을 시작해보세요.")
            return False

# 실행
if __name__ == "__main__":
    starter = Hero4GameStarter()
    
    if starter.run_full_setup():
        print("\n✅ 설정 완료! 이제 AI를 실행할 수 있습니다!")
        
        # 간단한 테스트 이동
        print("🤖 간단한 테스트 이동 (5초)...")
        test_keys = ['right', 'right', 'left', 'left', 'space']
        starter.send_keys_to_dosbox(test_keys)
        
    else:
        print("\n❌ 설정 실패. 수동으로 게임을 실행한 후 다시 시도하세요.")