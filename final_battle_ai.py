#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 최종 전투 AI - 실제 게임 화면에서 전투
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

class FinalHero4BattleAI:
    """최종 영웅전설4 전투 AI"""
    
    def __init__(self):
        """초기화"""
        self.battle_count = 0
        self.total_actions = 0
        self.start_time = time.time()
        self.dosbox_window = None
        self.game_region = None
        
        # 전투 패턴들
        self.exploration_patterns = [
            ['right', 'right', 'right', 'down', 'left', 'left', 'left', 'up'],
            ['down', 'down', 'right', 'right', 'up', 'up', 'left', 'left'],
            ['right', 'down', 'left', 'up'] * 2,
            ['up', 'right', 'down', 'left'] * 2
        ]
        self.current_pattern = 0
        self.pattern_step = 0
        
    def find_dosbox_window(self) -> bool:
        """DOSBox 창 찾기"""
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if 'dosbox' in window_text.lower() or 'ED4' in window_text:
                    windows.append(hwnd)
            return True

        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            self.dosbox_window = windows[0]
            self.game_region = win32gui.GetWindowRect(self.dosbox_window)
            print(f"🎮 DOSBox 연결: {self.game_region}")
            return True
        
        print("❌ DOSBox를 찾을 수 없습니다!")
        return False
    
    def send_key(self, key: str) -> bool:
        """키 입력"""
        if not self.dosbox_window:
            return False
        
        # DOSBox 활성화
        win32gui.SetForegroundWindow(self.dosbox_window)
        time.sleep(0.03)
        
        key_map = {
            'left': 0x25, 'right': 0x27, 'up': 0x26, 'down': 0x28,
            'space': 0x20, 'enter': 0x0D, 'esc': 0x1B,
            'z': 0x5A, 'x': 0x58, 'a': 0x41, 's': 0x53,
            '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34, '5': 0x35
        }
        
        if key in key_map:
            vk_code = key_map[key]
            win32api.keybd_event(vk_code, 0, 0, 0)
            time.sleep(0.08)
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            return True
        
        return False
    
    def analyze_game_screen(self) -> dict:
        """게임 화면 분석"""
        try:
            screenshot = ImageGrab.grab(self.game_region)
            image = np.array(screenshot)
            
            # HSV 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            height, width = image.shape[:2]
            total_pixels = height * width
            
            # 색상별 분석
            # 빨간색 (HP, 데미지, 적)
            red_mask1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
            red_count = np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)
            
            # 파란색 (MP, UI, 마법)
            blue_mask = cv2.inRange(hsv, (100, 60, 60), (130, 255, 255))
            blue_count = np.sum(blue_mask > 0)
            
            # 녹색 (HP 풀, 필드)
            green_mask = cv2.inRange(hsv, (40, 60, 60), (80, 255, 255))
            green_count = np.sum(green_mask > 0)
            
            # 노란색 (경험치, 골드, 선택)
            yellow_mask = cv2.inRange(hsv, (20, 60, 60), (40, 255, 255))
            yellow_count = np.sum(yellow_mask > 0)
            
            # 흰색 (텍스트, 테두리)
            white_mask = cv2.inRange(image, (200, 200, 200), (255, 255, 255))
            white_count = np.sum(white_mask > 0)
            
            # 비율 계산
            red_ratio = red_count / total_pixels
            blue_ratio = blue_count / total_pixels
            green_ratio = green_count / total_pixels
            yellow_ratio = yellow_count / total_pixels
            white_ratio = white_count / total_pixels
            
            # 밝기 및 대비
            brightness = np.mean(image)
            contrast = np.std(image)
            
            # 전투 가능성 계산
            battle_score = 0
            battle_reasons = []
            
            # 빨간색 많음 (HP바, 적, 데미지)
            if red_ratio > 0.05:
                battle_score += 3
                battle_reasons.append(f"빨간색많음({red_ratio:.3f})")
            
            # 파란색 많음 (UI, MP)
            if blue_ratio > 0.08:
                battle_score += 2
                battle_reasons.append(f"파란색많음({blue_ratio:.3f})")
            
            # 노란색 (선택메뉴, 경험치)
            if yellow_ratio > 0.03:
                battle_score += 2
                battle_reasons.append(f"노란색감지({yellow_ratio:.3f})")
            
            # 흰색 텍스트 많음
            if white_ratio > 0.15:
                battle_score += 1
                battle_reasons.append(f"텍스트많음({white_ratio:.3f})")
            
            # 높은 대비 (UI 활성화)
            if contrast > 40:
                battle_score += 1
                battle_reasons.append(f"고대비({contrast:.1f})")
            
            # 적절한 밝기
            if 50 < brightness < 150:
                battle_score += 1
                battle_reasons.append(f"적정밝기({brightness:.1f})")
            
            return {
                'battle_detected': battle_score >= 4,
                'battle_score': battle_score,
                'confidence': min(battle_score * 15, 100),
                'reasons': battle_reasons,
                'red_ratio': red_ratio,
                'blue_ratio': blue_ratio,
                'yellow_ratio': yellow_ratio,
                'brightness': brightness,
                'contrast': contrast
            }
            
        except Exception as e:
            return {'battle_detected': False, 'error': str(e)}
    
    def get_next_move(self) -> str:
        """다음 이동 결정"""
        # 현재 패턴 진행
        pattern = self.exploration_patterns[self.current_pattern]
        move = pattern[self.pattern_step % len(pattern)]
        
        self.pattern_step += 1
        
        # 패턴 변경 (20스텝마다)
        if self.pattern_step % 20 == 0:
            self.current_pattern = (self.current_pattern + 1) % len(self.exploration_patterns)
            print(f"🔄 패턴 변경: #{self.current_pattern}")
        
        return move
    
    async def battle_action(self, analysis: dict) -> None:
        """전투 액션 수행"""
        print(f"⚔️ 전투 감지! 점수:{analysis['battle_score']}/8, 신뢰도:{analysis['confidence']}%")
        print(f"   이유: {', '.join(analysis['reasons'])}")
        
        # 전투 액션 시퀀스
        battle_actions = [
            'z',      # 공격
            'enter',  # 확인
            'a',      # 액션/아이템
            '1',      # 선택 1
            'space',  # 스페이스
            'enter',  # 엔터
            '2',      # 선택 2
            'x'       # 취소/뒤로
        ]
        
        # 빠른 전투 처리
        for i in range(4):  # 처음 4개 액션만
            action = battle_actions[i]
            self.send_key(action)
            await asyncio.sleep(0.12)
        
        self.battle_count += 1
        print(f"🏆 전투 #{self.battle_count} 완료!")
    
    async def exploration_action(self) -> None:
        """탐험 액션"""
        # 기본 이동
        move = self.get_next_move()
        self.send_key(move)
        
        # 가끔 상호작용
        if self.total_actions % 12 == 0:
            interaction = random.choice(['space', 'enter', 'z'])
            await asyncio.sleep(0.1)
            self.send_key(interaction)
            print(f"🤝 상호작용: {interaction}")
    
    async def run_battle_ai(self, max_actions: int = 200, target_battles: int = 10) -> None:
        """전투 AI 실행"""
        print("⚔️ 영웅전설4 최종 전투 AI 시작!")
        print(f"🎯 목표: {target_battles}회 전투, 최대 {max_actions}회 행동")
        
        if not self.find_dosbox_window():
            return
        
        print("🚀 게임플레이 시작!")
        
        while self.total_actions < max_actions and self.battle_count < target_battles:
            self.total_actions += 1
            
            # 화면 분석
            analysis = self.analyze_game_screen()
            
            if 'error' not in analysis:
                # 전투 감지 시 대응
                if analysis['battle_detected']:
                    await self.battle_action(analysis)
                    await asyncio.sleep(0.4)  # 전투 후 대기
                else:
                    # 일반 탐험
                    await self.exploration_action()
                    await asyncio.sleep(0.1)  # 빠른 이동
            else:
                print(f"❌ 분석 오류: {analysis['error']}")
                await asyncio.sleep(0.2)
            
            # 진행 상황 출력
            if self.total_actions % 20 == 0:
                elapsed = time.time() - self.start_time
                aps = self.total_actions / elapsed if elapsed > 0 else 0
                print(f"📊 {self.total_actions}/{max_actions} | 전투:{self.battle_count}/{target_battles} | {aps:.1f}aps")
        
        # 결과 출력
        elapsed = time.time() - self.start_time
        efficiency = self.battle_count / self.total_actions if self.total_actions > 0 else 0
        
        print(f"\n🏁 최종 결과:")
        print(f"⏱️ 시간: {elapsed:.1f}초")
        print(f"⚔️ 전투: {self.battle_count}/{target_battles}회")
        print(f"🎮 행동: {self.total_actions}회")
        print(f"📈 효율: {efficiency:.3f}")
        print(f"⚡ 속도: {self.total_actions/elapsed:.1f} 행동/초")
        
        if self.battle_count >= target_battles:
            print("🎉 목표 달성 성공!")
        else:
            print("😅 목표 미달성")

# 실행
if __name__ == "__main__":
    async def main():
        ai = FinalHero4BattleAI()
        await ai.run_battle_ai(max_actions=200, target_battles=10)
    
    print("🏆 영웅전설4 최종 전투 AI")
    print("=" * 60)
    asyncio.run(main())