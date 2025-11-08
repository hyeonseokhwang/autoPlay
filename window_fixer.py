#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 화면 캡처 문제 해결사
"""

import win32gui
import win32con
import win32api
import time
from PIL import ImageGrab
import numpy as np

class Hero4WindowFixer:
    """영웅전설4 창 문제 해결"""
    
    def __init__(self):
        """초기화"""
        self.hero4_windows = []
        
    def find_all_windows(self) -> None:
        """모든 창 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                
                # DOSBox 또는 영웅전설4 관련 창
                if (window_text and 
                    ('dosbox' in window_text.lower() or 
                     'ED4' in window_text or
                     'DOS' in window_text or
                     '영웅전설' in window_text)):
                    
                    rect = win32gui.GetWindowRect(hwnd)
                    windows.append({
                        'hwnd': hwnd,
                        'title': window_text,
                        'class': class_name,
                        'rect': rect,
                        'visible': win32gui.IsWindowVisible(hwnd),
                        'enabled': win32gui.IsWindowEnabled(hwnd)
                    })
            return True

        print("🔍 모든 관련 창 찾는 중...")
        win32gui.EnumWindows(enum_callback, self.hero4_windows)
        
        print(f"📋 발견된 창: {len(self.hero4_windows)}개")
        for i, window in enumerate(self.hero4_windows):
            print(f"   {i+1}. {window['title']}")
            print(f"      클래스: {window['class']}")
            print(f"      위치: {window['rect']}")
            print(f"      보임: {window['visible']}, 활성: {window['enabled']}")
    
    def fix_window_state(self, window_index: int = 0) -> bool:
        """창 상태 수정"""
        if not self.hero4_windows or window_index >= len(self.hero4_windows):
            print("❌ 수정할 창이 없습니다!")
            return False
            
        window = self.hero4_windows[window_index]
        hwnd = window['hwnd']
        
        print(f"🔧 창 상태 수정 중: {window['title']}")
        
        try:
            # 1. 창을 맨 앞으로
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0, 0, 0, 0, 
                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
            print("   ✅ 창을 맨 앞으로 이동")
            
            # 2. 창 활성화
            win32gui.SetForegroundWindow(hwnd)
            print("   ✅ 창 활성화")
            
            # 3. 창 복원 (최소화 해제)
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            print("   ✅ 창 복원")
            
            time.sleep(1)
            
            # 4. 새로운 위치 확인
            new_rect = win32gui.GetWindowRect(hwnd)
            print(f"   📐 새 위치: {new_rect}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            return False
    
    def test_capture(self, window_index: int = 0) -> None:
        """캡처 테스트"""
        if not self.hero4_windows or window_index >= len(self.hero4_windows):
            print("❌ 테스트할 창이 없습니다!")
            return
            
        window = self.hero4_windows[window_index]
        rect = win32gui.GetWindowRect(window['hwnd'])
        
        print(f"📸 캡처 테스트: {rect}")
        
        try:
            # 전체 화면 캡처
            full_screenshot = ImageGrab.grab()
            print(f"   🖥️ 전체 화면: {full_screenshot.size}")
            
            # 창 영역 캡처
            window_screenshot = ImageGrab.grab(rect)
            print(f"   🎮 게임 창: {window_screenshot.size}")
            
            # 분석
            window_array = np.array(window_screenshot)
            brightness = np.mean(window_array)
            
            print(f"   🌟 평균 밝기: {brightness}")
            
            if brightness < 5:
                print("   ⚠️ 너무 어두움! (검은 화면)")
            elif brightness > 250:
                print("   ⚠️ 너무 밝음! (흰 화면)")
            else:
                print("   ✅ 정상 범위")
                
            # 이미지 저장
            timestamp = int(time.time())
            window_screenshot.save(f'test_capture_{timestamp}.png')
            print(f"   💾 저장: test_capture_{timestamp}.png")
            
        except Exception as e:
            print(f"   ❌ 캡처 실패: {e}")
    
    def multiple_capture_regions(self) -> None:
        """다양한 영역에서 캡처 시도"""
        print("🎯 다양한 영역 캡처 시도...")
        
        # 화면 크기 가져오기
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)
        
        regions = [
            (0, 0, screen_width//2, screen_height//2),      # 좌상
            (screen_width//2, 0, screen_width, screen_height//2),  # 우상
            (0, screen_height//2, screen_width//2, screen_height), # 좌하
            (screen_width//2, screen_height//2, screen_width, screen_height), # 우하
            (screen_width//4, screen_height//4, screen_width*3//4, screen_height*3//4), # 중앙
        ]
        
        region_names = ['좌상단', '우상단', '좌하단', '우하단', '중앙']
        
        for i, (region, name) in enumerate(zip(regions, region_names)):
            try:
                screenshot = ImageGrab.grab(region)
                brightness = np.mean(np.array(screenshot))
                
                print(f"   {name}: {region} → 밝기 {brightness:.1f}")
                
                if brightness > 10:  # 의미있는 내용이 있을 경우
                    screenshot.save(f'region_{i}_{name}_{int(time.time())}.png')
                    print(f"      💾 저장됨 (의미있는 내용 감지)")
                    
            except Exception as e:
                print(f"   {name}: 실패 ({e})")

# 실행
if __name__ == "__main__":
    print("🎮 영웅전설4 화면 캡처 문제 해결사")
    print("=" * 50)
    
    fixer = Hero4WindowFixer()
    
    # 1. 모든 관련 창 찾기
    fixer.find_all_windows()
    
    if fixer.hero4_windows:
        # 2. 첫 번째 창 상태 수정
        print(f"\n🔧 첫 번째 창 수정 시도...")
        if fixer.fix_window_state(0):
            print("✅ 창 상태 수정 완료!")
            
            # 3. 캡처 테스트
            print(f"\n📸 캡처 테스트...")
            time.sleep(2)
            fixer.test_capture(0)
        else:
            print("❌ 창 상태 수정 실패")
    else:
        print("❌ 영웅전설4 창을 찾을 수 없습니다!")
    
    # 4. 다양한 영역 캡처 시도
    print(f"\n🎯 화면 전체 영역 스캔...")
    fixer.multiple_capture_regions()
    
    print(f"\n🏁 진단 완료! 저장된 이미지를 확인하세요.")