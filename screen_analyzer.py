#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 화면 실시간 분석 및 디버깅 도구
"""

import cv2
import numpy as np
import time
from PIL import ImageGrab
import win32gui

class Hero4ScreenAnalyzer:
    """영웅전설4 화면 분석기"""
    
    def __init__(self):
        """초기화"""
        self.hero4_window = None
        self.capture_region = None
        
    def find_hero4_window(self) -> bool:
        """영웅전설4 창 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if ('ED4' in window_text or 'dosbox' in window_text.lower()):
                    windows.append((hwnd, window_text))
            return True

        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            self.hero4_window = windows[0][0]
            rect = win32gui.GetWindowRect(self.hero4_window)
            self.capture_region = rect
            print(f"🎮 창 발견: {windows[0][1]} | 영역: {rect}")
            return True
        
        print("❌ 영웅전설4 창을 찾을 수 없습니다!")
        return False
    
    def capture_and_analyze(self) -> None:
        """화면 캡처 및 상세 분석"""
        if not self.capture_region:
            print("❌ 캡처 영역이 설정되지 않았습니다!")
            return
        
        try:
            # 화면 캡처
            screenshot = ImageGrab.grab(self.capture_region)
            image = np.array(screenshot)
            
            # 기본 정보
            height, width = image.shape[:2]
            print(f"\n📸 화면 캡처 완료: {width}x{height}")
            
            # 색상 분석
            print(f"🎨 평균 밝기: {np.mean(image):.1f}")
            print(f"🔴 빨간색 평균: {np.mean(image[:,:,0]):.1f}")
            print(f"🟢 녹색 평균: {np.mean(image[:,:,1]):.1f}")
            print(f"🔵 파란색 평균: {np.mean(image[:,:,2]):.1f}")
            
            # HSV 색상 공간으로 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 다양한 색상 범위 분석
            colors = {
                '빨강1': ((0, 50, 50), (10, 255, 255)),
                '빨강2': ((170, 50, 50), (180, 255, 255)), 
                '파랑': ((100, 50, 50), (130, 255, 255)),
                '녹색': ((40, 50, 50), (80, 255, 255)),
                '노랑': ((20, 50, 50), (40, 255, 255)),
                '자주': ((140, 50, 50), (170, 255, 255)),
                '하늘': ((80, 50, 50), (100, 255, 255))
            }
            
            print("\n🌈 색상 분포 분석:")
            total_pixels = width * height
            
            for color_name, (lower, upper) in colors.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixel_count = np.sum(mask > 0)
                percentage = (pixel_count / total_pixels) * 100
                print(f"   {color_name}: {pixel_count:6d}픽셀 ({percentage:5.2f}%)")
            
            # 화면 영역별 분석
            print(f"\n📍 화면 영역별 밝기:")
            h_third, w_third = height // 3, width // 3
            
            regions = {
                '좌상': image[0:h_third, 0:w_third],
                '중상': image[0:h_third, w_third:2*w_third], 
                '우상': image[0:h_third, 2*w_third:width],
                '좌중': image[h_third:2*h_third, 0:w_third],
                '중앙': image[h_third:2*h_third, w_third:2*w_third],
                '우중': image[h_third:2*h_third, 2*w_third:width],
                '좌하': image[2*h_third:height, 0:w_third],
                '중하': image[2*h_third:height, w_third:2*w_third],
                '우하': image[2*h_third:height, 2*w_third:width]
            }
            
            for region_name, region_img in regions.items():
                if region_img.size > 0:
                    brightness = np.mean(region_img)
                    print(f"   {region_name}: {brightness:6.1f}")
            
            # 화면 저장 (디버깅용)
            timestamp = int(time.time())
            screenshot.save(f'hero4_screen_{timestamp}.png')
            print(f"\n💾 화면 저장: hero4_screen_{timestamp}.png")
            
            # 전투 가능성 판단
            print(f"\n⚔️ 전투 징후 분석:")
            
            # 빨간색 (HP, 데미지)
            red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
            red_ratio = (np.sum(red_mask1) + np.sum(red_mask2)) / (total_pixels * 3)
            
            # 파란색 (마나, UI)
            blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
            blue_ratio = np.sum(blue_mask) / (total_pixels * 3)
            
            print(f"   🔴 진한 빨강 비율: {red_ratio:.4f}")
            print(f"   🔵 진한 파랑 비율: {blue_ratio:.4f}")
            
            # 전투 판정
            battle_score = 0
            if red_ratio > 0.02:   # 2% 이상
                battle_score += 3
                print(f"   ✅ 빨간색 충분 (+3점)")
                
            if blue_ratio > 0.05:  # 5% 이상
                battle_score += 2
                print(f"   ✅ 파란색 충분 (+2점)")
                
            if np.mean(image) > 80:  # 밝은 화면
                battle_score += 1
                print(f"   ✅ 화면 밝음 (+1점)")
            
            print(f"\n🎯 전투 가능성 점수: {battle_score}/6")
            
            if battle_score >= 3:
                print("⚔️ 전투 상황일 가능성 높음!")
            elif battle_score >= 1:
                print("🤔 전투 상황일 가능성 있음")
            else:
                print("🚶 일반 이동/탐험 상황")
                
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
    
    def continuous_monitor(self, duration: int = 30) -> None:
        """지속적 모니터링"""
        print(f"🔄 {duration}초 동안 연속 모니터링 시작...")
        
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < duration:
            count += 1
            print(f"\n{'='*50}")
            print(f"📊 모니터링 #{count} ({time.time() - start_time:.1f}초)")
            
            self.capture_and_analyze()
            
            time.sleep(3)  # 3초마다 분석
            
        print(f"\n🏁 모니터링 완료! (총 {count}회 분석)")

# 실행
if __name__ == "__main__":
    print("🎮 영웅전설4 화면 분석 도구")
    print("=" * 50)
    
    analyzer = Hero4ScreenAnalyzer()
    
    if analyzer.find_hero4_window():
        print("\n🔍 현재 화면 상세 분석:")
        analyzer.capture_and_analyze()
        
        print(f"\n⏱️ 10초 후 연속 모니터링 시작...")
        time.sleep(10)
        
        # 30초 동안 연속 모니터링 
        analyzer.continuous_monitor(30)
    else:
        print("❌ 영웅전설4를 실행한 후 다시 시도하세요!")