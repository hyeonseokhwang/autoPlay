#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 실제 AI 모델 연결 시스템
- 실제 Ollama LLM 연결
- 화면을 텍스트로 설명해서 모델에게 전달
- 모델이 직접 추론하고 행동 결정
- 진짜 AI의 사고 과정
"""

import asyncio
import time
import json
import base64
import numpy as np
import cv2
import aiohttp
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
from PIL import ImageGrab, Image
import win32gui
import win32con
import win32api
import io

class RealAIVision:
    """실제 AI를 위한 시각 처리"""
    
    def __init__(self):
        """초기화"""
        self.last_screenshot = None
        self.vision_history = deque(maxlen=5)
        
    def describe_screen_for_ai(self, screenshot: np.ndarray) -> str:
        """AI가 이해할 수 있도록 화면을 텍스트로 설명"""
        if screenshot is None:
            return "화면을 볼 수 없습니다."
        
        try:
            # 기본 분석
            height, width = screenshot.shape[:2]
            brightness = np.mean(screenshot)
            
            # HSV 변환으로 색상 분석
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV)
            
            # 색상 영역 분석
            red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            yellow_mask = cv2.inRange(hsv, (20, 50, 50), (40, 255, 255))
            
            total_pixels = width * height
            red_ratio = (np.sum(red_mask1) + np.sum(red_mask2)) / total_pixels
            blue_ratio = np.sum(blue_mask) / total_pixels
            green_ratio = np.sum(green_mask) / total_pixels
            yellow_ratio = np.sum(yellow_mask) / total_pixels
            
            # 엣지 및 텍스처 분석
            gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_pixels
            
            # 화면을 9개 영역으로 나누어 분석
            h_step, w_step = height // 3, width // 3
            region_descriptions = []
            
            region_names = [
                "좌상단", "상단중앙", "우상단",
                "좌측중앙", "정중앙", "우측중앙", 
                "좌하단", "하단중앙", "우하단"
            ]
            
            for i in range(3):
                for j in range(3):
                    y1, y2 = i * h_step, (i + 1) * h_step
                    x1, x2 = j * w_step, (j + 1) * w_step
                    region = screenshot[y1:y2, x1:x2]
                    
                    if region.size > 0:
                        region_brightness = np.mean(region)
                        region_name = region_names[i * 3 + j]
                        
                        if region_brightness > 100:
                            brightness_desc = "밝음"
                        elif region_brightness > 50:
                            brightness_desc = "보통"
                        else:
                            brightness_desc = "어둠"
                        
                        region_descriptions.append(f"{region_name}: {brightness_desc}")
            
            # AI를 위한 자연어 설명 생성
            description = f"""
게임 화면 분석 결과:

기본 정보:
- 화면 크기: {width}x{height}
- 전체 밝기: {brightness:.1f} ({'밝음' if brightness > 80 else '보통' if brightness > 40 else '어둠'})

색상 분포:
- 빨간색 영역: {red_ratio*100:.1f}% {'(많음)' if red_ratio > 0.05 else '(적음)'}
- 파란색 영역: {blue_ratio*100:.1f}% {'(많음)' if blue_ratio > 0.08 else '(적음)'}  
- 녹색 영역: {green_ratio*100:.1f}% {'(많음)' if green_ratio > 0.1 else '(적음)'}
- 노란색 영역: {yellow_ratio*100:.1f}% {'(많음)' if yellow_ratio > 0.03 else '(적음)'}

화면 특성:
- 엣지 밀도: {edge_density*100:.1f}% {'(복잡함)' if edge_density > 0.1 else '(단순함)'}
- 전반적 특성: {'UI/메뉴 화면' if blue_ratio > 0.1 or yellow_ratio > 0.05 else '게임 필드' if green_ratio > 0.05 else '불명확한 화면'}

화면 영역별 상태:
{chr(10).join(region_descriptions)}

이전 화면과의 차이:
{self._describe_screen_changes()}
"""
            
            self.vision_history.append({
                'timestamp': datetime.now(),
                'description': description,
                'stats': {
                    'brightness': brightness,
                    'red_ratio': red_ratio,
                    'blue_ratio': blue_ratio,
                    'green_ratio': green_ratio,
                    'yellow_ratio': yellow_ratio,
                    'edge_density': edge_density
                }
            })
            
            return description
            
        except Exception as e:
            return f"화면 분석 중 오류 발생: {e}"
    
    def _describe_screen_changes(self) -> str:
        """이전 화면과의 변화 설명"""
        if len(self.vision_history) < 2:
            return "첫 번째 관찰입니다."
        
        prev_stats = self.vision_history[-2]['stats']
        
        changes = []
        if len(self.vision_history) >= 2:
            current_stats = {
                'brightness': np.mean(self.last_screenshot) if self.last_screenshot is not None else 0,
                'red_ratio': 0, 'blue_ratio': 0, 'green_ratio': 0, 'yellow_ratio': 0
            }
            
            brightness_change = abs(current_stats['brightness'] - prev_stats['brightness'])
            if brightness_change > 20:
                changes.append(f"밝기 {'증가' if current_stats['brightness'] > prev_stats['brightness'] else '감소'}")
            
            for color in ['red_ratio', 'blue_ratio', 'green_ratio']:
                if abs(current_stats.get(color, 0) - prev_stats.get(color, 0)) > 0.03:
                    changes.append(f"{color.split('_')[0]} 색상 변화 감지")
        
        return "변화: " + ", ".join(changes) if changes else "큰 변화 없음"

class RealAIBrain:
    """실제 AI 모델 연결 및 추론"""
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b"):
        """초기화"""
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434"
        self.conversation_history = deque(maxlen=20)
        self.total_thoughts = 0
        
    async def think_and_decide(self, screen_description: str, 
                              action_history: List[str], 
                              battle_count: int,
                              step_count: int) -> Dict[str, Any]:
        """AI가 직접 생각하고 행동 결정"""
        
        self.total_thoughts += 1
        
        # AI에게 보낼 프롬프트 구성
        prompt = f"""
영웅전설4 RPG 플레이 AI. 빠른 결정 필요.

화면: {screen_description[:300]}

스텝: {step_count} | 전투: {battle_count} | 최근: {action_history[-3:] if action_history else '없음'}

행동: left/right/up/down/space/enter/z/x/a/s/1/2

목표: 탐험, 전투 찾기, 상호작용

빠르게 JSON으로 답변:
{{
    "thoughts": "짧은 분석",
    "action": "행동선택",
    "reason": "이유",
    "confidence": 0.8
}}
"""
        
        try:
            # Ollama API 호출
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # 더 빠른 결정
                        "top_p": 0.7,       # 더 집중된 선택
                        "max_tokens": 200,   # 더 짧은 응답
                        "num_ctx": 1024,     # 더 작은 컨텍스트
                        "num_predict": 150   # 더 적은 예측
                    }
                }
                
                print(f"🧠 AI 사고 중... ({self.total_thoughts}번째 생각)")
                
                async with session.post(f"{self.ollama_url}/api/generate", 
                                      json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get('response', '')
                        
                        # JSON 파싱 시도
                        try:
                            # JSON 부분 추출
                            json_start = ai_response.find('{')
                            json_end = ai_response.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = ai_response[json_start:json_end]
                                ai_decision = json.loads(json_str)
                                
                                # 대화 기록 저장
                                self.conversation_history.append({
                                    'step': step_count,
                                    'screen': screen_description[:200] + "...",
                                    'ai_response': ai_decision,
                                    'timestamp': datetime.now()
                                })
                                
                                return ai_decision
                            
                        except json.JSONDecodeError as e:
                            print(f"⚠️ AI 응답 JSON 파싱 실패: {e}")
                            print(f"원본 응답: {ai_response[:500]}...")
                    
                    else:
                        print(f"❌ Ollama API 오류: {response.status}")
                        
        except Exception as e:
            print(f"❌ AI 연결 실패: {e}")
        
        # 실패 시 기본 응답
        return {
            "thoughts": "AI 연결에 문제가 있어 기본 탐험을 시작합니다.",
            "reasoning": "안전한 탐험 행동을 선택합니다.",
            "action": "right",
            "reason": "우측 탐험으로 새로운 영역을 찾아보겠습니다.",
            "expectation": "새로운 화면이나 상호작용을 발견하기를 기대합니다.",
            "curiosity_level": 0.7,
            "confidence": 0.5
        }
    
    def get_learning_summary(self) -> str:
        """AI의 학습 요약"""
        if not self.conversation_history:
            return "아직 학습 데이터가 없습니다."
        
        actions_taken = [conv['ai_response'].get('action', '') for conv in self.conversation_history]
        action_counts = {}
        for action in actions_taken:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        most_used_action = max(action_counts, key=action_counts.get) if action_counts else "없음"
        
        return f"""
AI 학습 요약:
- 총 사고 횟수: {self.total_thoughts}
- 기록된 대화: {len(self.conversation_history)}
- 가장 선호하는 행동: {most_used_action}
- 행동 분포: {action_counts}
"""

class RealAIGameController:
    """실제 AI용 게임 컨트롤러"""
    
    def __init__(self):
        """초기화"""
        self.dosbox_window = None
        self.game_region = None
        
    def find_game_window(self) -> bool:
        """게임 창 찾기"""
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
            print(f"🎮 실제 게임 연결: {self.game_region}")
            return True
        
        return False
    
    def capture_screen(self) -> np.ndarray:
        """화면 캡처"""
        try:
            screenshot = ImageGrab.grab(self.game_region)
            return np.array(screenshot)
        except Exception as e:
            print(f"❌ 화면 캡처 실패: {e}")
            return None
    
    def execute_ai_action(self, action: str) -> bool:
        """AI의 결정을 게임에 전송"""
        if not self.dosbox_window:
            return False
        
        try:
            win32gui.SetForegroundWindow(self.dosbox_window)
            time.sleep(0.02)  # 최소 지연
            
            key_map = {
                'left': 0x25, 'right': 0x27, 'up': 0x26, 'down': 0x28,
                'space': 0x20, 'enter': 0x0D, 'z': 0x5A, 'x': 0x58,
                'a': 0x41, 's': 0x53, '1': 0x31, '2': 0x32
            }
            
            if action in key_map:
                vk_code = key_map[action]
                win32api.keybd_event(vk_code, 0, 0, 0)
                time.sleep(0.03)  # 최고속 키입력
                win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
                return True
                
        except Exception as e:
            print(f"❌ 행동 실행 실패: {e}")
        
        return False

class RealAIPlayer:
    """실제 AI 플레이어 시스템"""
    
    def __init__(self):
        """초기화"""
        self.vision = RealAIVision()
        self.brain = RealAIBrain()
        self.controller = RealAIGameController()
        
        # 게임 상태 추적
        self.step_count = 0
        self.battle_count = 0
        self.action_history = deque(maxlen=50)
        self.session_start = time.time()
        
        print("🤖 실제 AI 플레이어 시스템 초기화 완료")
        print("🧠 AI 모델: Ollama LLM 연결")
        print("👁️ 비전: 실시간 화면 분석 및 자연어 변환")
        print("🎮 컨트롤러: 직접 게임 조작")
        
    async def ai_gaming_step(self) -> bool:
        """AI의 한 번 게임 스텝"""
        self.step_count += 1
        
        # 1. 화면 관찰
        screenshot = self.controller.capture_screen()
        if screenshot is None:
            print("❌ 화면을 볼 수 없습니다.")
            return False
        
        # 2. AI가 이해할 수 있도록 화면 설명
        screen_description = self.vision.describe_screen_for_ai(screenshot)
        
        # 3. AI가 직접 생각하고 결정
        ai_decision = await self.brain.think_and_decide(
            screen_description, 
            list(self.action_history),
            self.battle_count,
            self.step_count
        )
        
        # 4. 간단한 출력 (속도 최적화)
        if self.step_count % 5 == 0:  # 5번에 한번만 출력
            print(f"� #{self.step_count}: {ai_decision.get('action')} | {ai_decision.get('thoughts', '...')[:50]}...")
        
        # 5. 행동 실행
        action = ai_decision.get('action', 'right')
        success = self.controller.execute_ai_action(action)
        
        if success:
            self.action_history.append(action)
            print(f"   ✅ 실행됨: {action.upper()}")
            
            # 최소 대기
            await asyncio.sleep(0.02)
            
            # 간단한 전투 감지 (화면 변화 기반)
            if self._detect_battle_from_ai_perspective(ai_decision):
                self.battle_count += 1
                print(f"   ⚔️ AI가 전투 상황 감지! 총 {self.battle_count}회")
            
            return True
        else:
            print(f"   ❌ 행동 실행 실패")
            return False
    
    def _detect_battle_from_ai_perspective(self, ai_decision: Dict) -> bool:
        """AI 관점에서 전투 감지"""
        # AI의 생각이나 추론에 전투 관련 키워드가 있는지 확인
        thoughts = ai_decision.get('thoughts', '').lower()
        reasoning = ai_decision.get('reasoning', '').lower()
        
        battle_keywords = ['전투', 'battle', '적', 'enemy', '싸움', 'fight', '공격', 'attack', '몬스터', 'monster']
        
        for keyword in battle_keywords:
            if keyword in thoughts or keyword in reasoning:
                return True
        
        # 호기심이나 확신도가 높을 때도 특별한 상황으로 간주
        curiosity = ai_decision.get('curiosity_level', 0)
        confidence = ai_decision.get('confidence', 0)
        
        if curiosity > 0.8 and confidence > 0.7:
            return True
        
        return False
    
    async def run_real_ai_session(self, max_steps: int = 999999, target_battles: int = 999999) -> None:
        """실제 AI 세션 실행"""
        print(f"\n🚀 최고속도 AI 플레이 세션 시작!")
        print(f"⚡ 무제한 모드: 횟수 제한 없음")
        print(f"🤖 AI가 최고 속도로 생각하고 판단합니다!\n")
        
        if not self.controller.find_game_window():
            print("❌ 게임을 찾을 수 없습니다!")
            return
        
        successful_steps = 0
        
        while (self.step_count < max_steps and 
               self.battle_count < target_battles):
            
            step_success = await self.ai_gaming_step()
            if step_success:
                successful_steps += 1
            
            await asyncio.sleep(0.05)  # 최고 속도
            
            # 진행 상황 출력 (더 자주)
            if self.step_count % 25 == 0:
                elapsed = time.time() - self.session_start
                print(f"\n📊 진행 상황 (스텝 {self.step_count}):")
                print(f"   ⏱️ 경과 시간: {elapsed:.1f}초")
                print(f"   ⚔️ 전투 발견: {self.battle_count}/{target_battles}")
                print(f"   ✅ 성공한 행동: {successful_steps}/{self.step_count}")
                print(f"   🎮 최근 행동: {list(self.action_history)[-5:]}")
        
        # 최종 결과
        elapsed = time.time() - self.session_start
        success_rate = successful_steps / max(self.step_count, 1)
        
        print(f"\n🏁 실제 AI 세션 완료!")
        print(f"⏱️ 총 시간: {elapsed:.1f}초")
        print(f"🎮 총 스텝: {self.step_count}")
        print(f"⚔️ 전투 발견: {self.battle_count}/{target_battles}")
        print(f"📈 행동 성공률: {success_rate:.1%}")
        
        # AI 학습 요약
        print(f"\n📚 AI 학습 요약:")
        print(self.brain.get_learning_summary())
        
        if self.battle_count >= target_battles:
            print("\n🎉 목표 달성! 실제 AI가 성공적으로 게임을 플레이했습니다!")
        else:
            print("\n📈 AI가 실제 게임 경험을 쌓았습니다!")

# 실행
if __name__ == "__main__":
    async def main():
        player = RealAIPlayer()
        await player.run_real_ai_session()  # 무제한 모드
    
    print("🤖 실제 AI 모델 연결 게임 플레이어")
    print("=" * 70)
    print("🧠 특징: 진짜 LLM 추론 + 실시간 화면 분석 + 자연어 사고")
    asyncio.run(main())