#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 화면 분석 및 학습 디버깅 도구
"""

import time
import cv2
import numpy as np
import pyautogui
import win32gui
import win32con
import win32api
import hashlib
from datetime import datetime

class ScreenAnalyzer:
    """실시간 화면 분석기"""
    
    def __init__(self):
        self.dosbox_hwnd = None
        self.capture_region = None
        self.screen_history = []
        
    def find_dosbox(self):
        """DOSBox 찾기"""
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
            print(f"✅ DOSBox 발견: {title}")
            
            # 캡처 영역 설정
            rect = win32gui.GetWindowRect(self.dosbox_hwnd)
            x, y, x2, y2 = rect
            self.capture_region = (x + 8, y + 30, x2 - x - 16, y2 - y - 38)
            print(f"📸 캡처 영역: {self.capture_region}")
            return True
        return False
    
    def capture_and_analyze(self):
        """캡처 및 분석"""
        if not self.capture_region:
            return None
        
        try:
            # 화면 캡처
            screenshot = pyautogui.screenshot(region=self.capture_region)
            image = np.array(screenshot)
            
            # 기본 분석
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 화면 해시
            small_gray = cv2.resize(gray, (32, 24))
            screen_hash = hashlib.md5(small_gray.tobytes()).hexdigest()[:8]
            
            # 밝기 분석
            brightness = np.mean(gray)
            
            # 색상 분석
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
            green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            
            total_pixels = image.shape[0] * image.shape[1]
            blue_ratio = np.sum(blue_mask > 0) / total_pixels
            red_ratio = (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / total_pixels
            green_ratio = np.sum(green_mask > 0) / total_pixels
            
            # 텍스트/에지 분석
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / total_pixels
            
            bright_mask = gray > 180
            bright_ratio = np.sum(bright_mask) / total_pixels
            
            # 상황 분류
            screen_type = self.classify_screen(brightness, blue_ratio, red_ratio, green_ratio, bright_ratio, edge_ratio)
            
            analysis = {
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'hash': screen_hash,
                'type': screen_type,
                'brightness': brightness,
                'colors': {
                    'blue': blue_ratio,
                    'red': red_ratio,
                    'green': green_ratio
                },
                'bright_ratio': bright_ratio,
                'edge_ratio': edge_ratio,
                'image_shape': image.shape
            }
            
            # 히스토리에 추가
            self.screen_history.append(analysis)
            if len(self.screen_history) > 20:
                self.screen_history.pop(0)
            
            return analysis
            
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            return None
    
    def classify_screen(self, brightness, blue_ratio, red_ratio, green_ratio, bright_ratio, edge_ratio):
        """화면 분류"""
        if bright_ratio > 0.2 and edge_ratio > 0.05:
            return 'dialogue'
        elif blue_ratio > 0.12:
            return 'menu'
        elif red_ratio > 0.08:
            return 'battle'
        elif green_ratio > 0.05:
            return 'status'
        elif brightness < 60:
            return 'dark'
        elif brightness > 120:
            return 'bright'
        else:
            return 'field'
    
    def get_change_analysis(self):
        """변화 분석"""
        if len(self.screen_history) < 2:
            return "변화 분석 불가 (데이터 부족)"
        
        recent = self.screen_history[-5:]
        unique_hashes = len(set(s['hash'] for s in recent))
        unique_types = len(set(s['type'] for s in recent))
        
        brightness_changes = []
        for i in range(1, len(recent)):
            change = abs(recent[i]['brightness'] - recent[i-1]['brightness'])
            brightness_changes.append(change)
        
        avg_brightness_change = sum(brightness_changes) / len(brightness_changes) if brightness_changes else 0
        
        return {
            'unique_screens': unique_hashes,
            'unique_types': unique_types,
            'avg_brightness_change': avg_brightness_change,
            'stuck': unique_hashes <= 1 and len(recent) >= 3
        }
    
    def send_test_key(self, key):
        """테스트 키 전송"""
        if not self.dosbox_hwnd:
            return False
        
        try:
            win32gui.SetForegroundWindow(self.dosbox_hwnd)
            time.sleep(0.05)
            
            key_map = {
                'up': win32con.VK_UP,
                'down': win32con.VK_DOWN,
                'left': win32con.VK_LEFT,
                'right': win32con.VK_RIGHT,
                'enter': win32con.VK_RETURN,
                'space': win32con.VK_SPACE,
                'esc': win32con.VK_ESCAPE
            }
            
            if key.lower() in key_map:
                vk = key_map[key.lower()]
                win32api.keybd_event(vk, 0, 0, 0)
                time.sleep(0.05)
                win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
                return True
        except Exception as e:
            print(f"❌ 키 전송 실패: {e}")
        return False

def main():
    """실시간 분석 메인"""
    print("🔍 실시간 화면 분석 및 학습 디버깅")
    print("=" * 50)
    
    analyzer = ScreenAnalyzer()
    
    if not analyzer.find_dosbox():
        print("❌ DOSBox를 찾을 수 없습니다!")
        return
    
    print("🚀 실시간 분석 시작 (Ctrl+C로 중단)")
    print("📋 분석 항목:")
    print("  - 화면 해시 (변화 감지)")
    print("  - 화면 타입 분류")
    print("  - 색상 비율 분석")
    print("  - 학습 상태 추적")
    print()
    
    test_keys = ['right', 'down', 'left', 'up', 'space', 'enter']
    key_index = 0
    
    try:
        for cycle in range(1, 101):  # 100사이클
            print(f"--- 사이클 {cycle} ---")
            
            # 분석 실행
            analysis = analyzer.capture_and_analyze()
            
            if analysis:
                # 기본 정보 출력
                print(f"🕐 {analysis['timestamp']} | 해시: {analysis['hash']}")
                print(f"📱 화면타입: {analysis['type']} | 밝기: {analysis['brightness']:.1f}")
                print(f"🎨 색상비율 - 파랑:{analysis['colors']['blue']:.2f} "
                      f"빨강:{analysis['colors']['red']:.2f} 초록:{analysis['colors']['green']:.2f}")
                print(f"📄 밝은영역:{analysis['bright_ratio']:.2f} 에지:{analysis['edge_ratio']:.2f}")
                
                # 변화 분석
                change_info = analyzer.get_change_analysis()
                if isinstance(change_info, dict):
                    print(f"🔄 변화분석 - 고유화면:{change_info['unique_screens']} "
                          f"고유타입:{change_info['unique_types']} "
                          f"평균밝기변화:{change_info['avg_brightness_change']:.1f}")
                    
                    if change_info['stuck']:
                        print("⚠️ 화면 변화 없음 - 막힌 상태!")
                else:
                    print(f"🔄 {change_info}")
            
            # 테스트 키 전송
            test_key = test_keys[key_index % len(test_keys)]
            print(f"🎮 테스트 키 전송: {test_key.upper()}")
            
            if analyzer.send_test_key(test_key):
                print("✅ 키 전송 성공")
            else:
                print("❌ 키 전송 실패")
            
            key_index += 1
            
            # 1초 대기
            time.sleep(1.0)
            print()
            
            # 10사이클마다 요약
            if cycle % 10 == 0:
                print("📊 최근 10사이클 요약:")
                recent_hashes = [s['hash'] for s in analyzer.screen_history[-10:]]
                recent_types = [s['type'] for s in analyzer.screen_history[-10:]]
                
                unique_hash_count = len(set(recent_hashes))
                unique_type_count = len(set(recent_types))
                
                print(f"  고유 화면: {unique_hash_count}/10")
                print(f"  고유 타입: {unique_type_count}/10")
                
                type_counts = {}
                for t in recent_types:
                    type_counts[t] = type_counts.get(t, 0) + 1
                
                print(f"  타입 분포: {type_counts}")
                
                if unique_hash_count <= 2:
                    print("  ⚠️ 학습 필요: 화면이 거의 변하지 않음!")
                else:
                    print("  ✅ 정상 탐험: 다양한 화면 변화 감지")
                print("=" * 30)
    
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    
    # 최종 분석
    print("\n📊 최종 분석 결과:")
    
    if analyzer.screen_history:
        all_hashes = [s['hash'] for s in analyzer.screen_history]
        all_types = [s['type'] for s in analyzer.screen_history]
        
        total_unique_screens = len(set(all_hashes))
        total_unique_types = len(set(all_types))
        
        print(f"총 분석 횟수: {len(analyzer.screen_history)}")
        print(f"발견한 고유 화면: {total_unique_screens}")
        print(f"발견한 화면 타입: {total_unique_types}")
        
        type_distribution = {}
        for t in all_types:
            type_distribution[t] = type_distribution.get(t, 0) + 1
        
        print(f"화면 타입 분포: {type_distribution}")
        
        # 학습 평가
        exploration_ratio = total_unique_screens / len(analyzer.screen_history)
        print(f"탐험 효율: {exploration_ratio:.2f}")
        
        if exploration_ratio > 0.3:
            print("✅ 좋은 탐험: 다양한 화면을 발견했습니다!")
        elif exploration_ratio > 0.1:
            print("⚠️ 보통 탐험: 일부 새로운 화면을 발견했습니다.")
        else:
            print("❌ 탐험 부족: 같은 화면만 반복하고 있습니다!")

if __name__ == "__main__":
    main()