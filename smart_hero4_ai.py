#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 실제 게임 화면 찾기 및 자동 전투 AI
"""

import asyncio
import time
import random
import numpy as np
import cv2
from PIL import ImageGrab
import win32gui
import win32con
import win32api

class SmartHero4AI:
    """스마트 영웅전설4 AI - 실제 화면 찾기 + 전투"""
    
    def __init__(self):
        """초기화"""
        self.battle_count = 0
        self.total_actions = 0
        self.start_time = time.time()
        
        # 다양한 가능한 캡처 영역들
        self.capture_regions = []
        self.active_region = None
        
        # 게임 관련 창들
        self.game_windows = []
        
    def find_all_game_windows(self) -> bool:
        """모든 게임 관련 창 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                
                # 영웅전설4, DOSBox, 게임런처 등
                keywords = ['dosbox', 'ED4', 'DOS', '영웅전설', '게임런처', 'launcher']
                
                if window_text and any(keyword.lower() in window_text.lower() for keyword in keywords):
                    rect = win32gui.GetWindowRect(hwnd)
                    windows.append({
                        'hwnd': hwnd,
                        'title': window_text,
                        'class': class_name,
                        'rect': rect
                    })
            return True

        win32gui.EnumWindows(enum_callback, self.game_windows)
        
        print(f"🎮 발견된 게임 관련 창: {len(self.game_windows)}개")
        for window in self.game_windows:
            print(f"   📝 {window['title']} | {window['rect']}")
            
        return len(self.game_windows) > 0
    
    def setup_capture_regions(self) -> None:
        """캡처 영역들 설정"""
        # 1. 발견된 게임 창들의 영역
        for window in self.game_windows:
            self.capture_regions.append({
                'name': f"창_{window['title'][:20]}...",
                'region': window['rect'],
                'type': 'window'
            })
        
        # 2. 화면의 주요 영역들  
        screen_regions = [
            {'name': '좌상단', 'region': (0, 0, 1280, 720), 'type': 'screen'},
            {'name': '우상단', 'region': (1280, 0, 2560, 720), 'type': 'screen'},
            {'name': '좌하단', 'region': (0, 720, 1280, 1440), 'type': 'screen'},
            {'name': '우하단', 'region': (1280, 720, 2560, 1440), 'type': 'screen'},
            {'name': '중앙', 'region': (640, 360, 1920, 1080), 'type': 'screen'},
            {'name': '전체', 'region': (0, 0, 3840, 2160), 'type': 'screen'},
        ]
        
        self.capture_regions.extend(screen_regions)
        
        print(f"📍 설정된 캡처 영역: {len(self.capture_regions)}개")
    
    def test_capture_region(self, region_info: dict) -> dict:
        """캡처 영역 테스트"""
        try:
            screenshot = ImageGrab.grab(region_info['region'])
            image_array = np.array(screenshot)
            
            # 기본 통계
            brightness = np.mean(image_array)
            std_dev = np.std(image_array)
            
            # 색상 분석
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            
            # 게임스러운 색상 패턴 체크
            red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)) + cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            
            total_pixels = image_array.shape[0] * image_array.shape[1]
            color_ratio = (np.sum(red_mask) + np.sum(blue_mask) + np.sum(green_mask)) / (total_pixels * 3)
            
            # 게임 가능성 점수 계산
            game_score = 0
            
            if 20 < brightness < 200:     # 적절한 밝기
                game_score += 2
                
            if std_dev > 15:             # 충분한 변화량
                game_score += 2
                
            if color_ratio > 0.1:       # 다양한 색상
                game_score += 3
                
            if 500 < total_pixels < 2000000:  # 적절한 크기
                game_score += 1
            
            return {
                'region_info': region_info,
                'brightness': brightness,
                'std_dev': std_dev,
                'color_ratio': color_ratio,
                'game_score': game_score,
                'screenshot': screenshot
            }
            
        except Exception as e:
            return {
                'region_info': region_info,
                'error': str(e),
                'game_score': 0
            }
    
    def find_best_game_region(self) -> bool:
        """최적의 게임 화면 영역 찾기"""
        print("🔍 최적의 게임 화면 영역 탐색 중...")
        
        best_region = None
        best_score = 0
        
        for region_info in self.capture_regions:
            result = self.test_capture_region(region_info)
            
            if 'error' not in result:
                score = result['game_score']
                print(f"   {region_info['name']}: 점수 {score}/8 (밝기:{result['brightness']:.1f}, 변화:{result['std_dev']:.1f}, 색상:{result['color_ratio']:.3f})")
                
                if score > best_score:
                    best_score = score
                    best_region = result
            else:
                print(f"   {region_info['name']}: 오류 - {result['error']}")
        
        if best_region and best_score >= 3:
            self.active_region = best_region['region_info']['region'] 
            print(f"✅ 최적 영역 선택: {best_region['region_info']['name']} (점수: {best_score}/8)")
            
            # 샘플 이미지 저장
            best_region['screenshot'].save(f'best_game_region_{int(time.time())}.png')
            
            return True
        else:
            print("❌ 적절한 게임 화면을 찾을 수 없습니다!")
            return False
    
    def send_key_to_game(self, key: str) -> bool:
        """게임에 키 입력"""
        # 게임 창 활성화 시도
        for window in self.game_windows:
            if 'dosbox' in window['title'].lower() or 'ED4' in window['title']:
                try:
                    win32gui.SetForegroundWindow(window['hwnd'])
                    break
                except:
                    pass
        
        time.sleep(0.05)
        
        # 키 매핑
        key_map = {
            'left': 0x25, 'right': 0x27, 'up': 0x26, 'down': 0x28,
            'space': 0x20, 'enter': 0x0D, 'esc': 0x1B,
            'z': 0x5A, 'x': 0x58, 'a': 0x41, 's': 0x53,
            '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34
        }
        
        if key in key_map:
            vk_code = key_map[key]
            win32api.keybd_event(vk_code, 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            return True
        
        return False
    
    def analyze_current_screen(self) -> dict:
        """현재 화면 분석"""
        if not self.active_region:
            return {'battle_detected': False, 'confidence': 0}
        
        try:
            screenshot = ImageGrab.grab(self.active_region)
            image = np.array(screenshot)
            
            # HSV 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 전투 관련 색상 감지
            red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
            red_pixels = np.sum(red_mask1) + np.sum(red_mask2)
            
            blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
            blue_pixels = np.sum(blue_mask)
            
            yellow_mask = cv2.inRange(hsv, (20, 80, 80), (40, 255, 255))
            yellow_pixels = np.sum(yellow_mask)
            
            total_pixels = image.shape[0] * image.shape[1] * 3
            
            # 전투 신호 계산
            battle_signals = 0
            confidence = 0
            
            if red_pixels / total_pixels > 0.03:    # 빨간색 3% 이상
                battle_signals += 3
                confidence += 40
                
            if blue_pixels / total_pixels > 0.05:   # 파란색 5% 이상
                battle_signals += 2
                confidence += 30
                
            if yellow_pixels / total_pixels > 0.02: # 노란색 2% 이상
                battle_signals += 1
                confidence += 20
                
            brightness = np.mean(image)
            if brightness > 100:                     # 밝은 화면
                battle_signals += 1
                confidence += 10
            
            return {
                'battle_detected': battle_signals >= 2,
                'confidence': min(confidence, 100),
                'red_ratio': red_pixels / total_pixels,
                'blue_ratio': blue_pixels / total_pixels,
                'yellow_ratio': yellow_pixels / total_pixels,
                'brightness': brightness
            }
            
        except Exception as e:
            return {'battle_detected': False, 'confidence': 0, 'error': str(e)}
    
    async def battle_sequence(self, analysis: dict) -> None:
        """전투 시퀀스 실행"""
        print(f"⚔️ 전투 감지! 신뢰도: {analysis['confidence']}%")
        
        # 전투 액션들
        actions = ['z', 'enter', 'a', '1', 'space', 'x', '2', 's']
        
        for action in actions[:4]:  # 처음 4개 액션만
            self.send_key_to_game(action)
            await asyncio.sleep(0.15)
            
        self.battle_count += 1
        print(f"🏆 전투 #{self.battle_count} 완료!")
    
    async def run_smart_ai(self, max_actions: int = 150, target_battles: int = 10) -> None:
        """스마트 AI 실행"""
        print("🚀 스마트 영웅전설4 AI 시작!")
        
        # 1. 게임 창 찾기
        if not self.find_all_game_windows():
            print("❌ 게임 창을 찾을 수 없습니다!")
            return
        
        # 2. 캡처 영역 설정
        self.setup_capture_regions()
        
        # 3. 최적 게임 영역 찾기
        if not self.find_best_game_region():
            print("❌ 게임 화면을 찾을 수 없습니다!")
            return
            
        print(f"🎯 목표: {target_battles}회 전투!")
        
        # 4. 게임플레이 루프
        movement_pattern = ['right', 'right', 'down', 'left', 'left', 'up'] * 10
        interaction_keys = ['space', 'enter', 'z', 'a', '1']
        
        while self.total_actions < max_actions and self.battle_count < target_battles:
            self.total_actions += 1
            
            # 화면 분석
            analysis = self.analyze_current_screen()
            
            # 전투 감지 시 대응
            if analysis.get('battle_detected', False):
                await self.battle_sequence(analysis)
                await asyncio.sleep(0.5)
                continue
            
            # 일반 행동
            if self.total_actions % 8 == 0:  # 가끔 상호작용
                action = random.choice(interaction_keys)
                print(f"🤝 상호작용: {action}")
            else:  # 이동
                action = movement_pattern[self.total_actions % len(movement_pattern)]
            
            self.send_key_to_game(action)
            await asyncio.sleep(0.12)
            
            # 진행 상황
            if self.total_actions % 15 == 0:
                elapsed = time.time() - self.start_time
                aps = self.total_actions / elapsed if elapsed > 0 else 0
                print(f"📊 진행: {self.total_actions}/{max_actions} | 전투: {self.battle_count}/{target_battles} | {aps:.1f}aps")
        
        # 결과
        elapsed = time.time() - self.start_time
        print(f"\n🏁 완료! 시간: {elapsed:.1f}초, 전투: {self.battle_count}/{target_battles}회")
        
        if self.battle_count >= target_battles:
            print("🎉 목표 달성!")
        else:
            print("😅 목표 미달성")

# 실행
if __name__ == "__main__":
    async def main():
        ai = SmartHero4AI()
        await ai.run_smart_ai(max_actions=150, target_battles=10)
    
    print("🧠 스마트 영웅전설4 AI")
    print("=" * 50)
    asyncio.run(main())