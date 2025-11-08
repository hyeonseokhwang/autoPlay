#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 기반 영웅전설4 AI - 실제 추론 능력으로 게임 플레이
목표: 좌우 이동하며 전투 10회 이상 달성
"""

import asyncio
import time
import json
import requests
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import base64
import io

# 게임 제어 & 비전
import cv2
import numpy as np
import pyautogui
import win32gui
import win32con
import win32api
import win32process
from PIL import Image

# 웹 검색
import aiohttp
from bs4 import BeautifulSoup

@dataclass
class GameSituation:
    """게임 상황 정보"""
    screen_type: str        # 화면 타입
    description: str        # 상황 설명
    possible_actions: List[str]  # 가능한 행동들
    screenshot_b64: str     # 스크린샷 (base64)
    battle_count: int       # 전투 횟수
    movement_pattern: str   # 이동 패턴

class LLMConnector:
    """로컬 Ollama LLM과 연결"""
    
    def __init__(self, model_name="qwen2.5-coder:7b"):
        self.base_url = "http://localhost:11434/api"
        self.model = model_name
        self.conversation_history = []
        
        print(f"🧠 LLM 연결: {model_name}")
        
    async def query_llm(self, prompt: str, image_b64: Optional[str] = None) -> str:
        """LLM에 질의 - 수정된 API"""
        try:
            # Ollama Generate API 사용 (더 단순하고 안정적)
            payload = {
                "model": self.model,
                "prompt": f"""You are an expert Legend of Heroes 4 (ED4) game AI.
Your goal: Move left/right and experience 10+ battles through intelligent gameplay.

Action priorities:
1. Seek battle opportunities
2. Explore new areas  
3. Handle menus/dialogs efficiently
4. Progress safely

Current situation: {prompt}

Respond ONLY in this exact JSON format:
{{"action": "up", "reason": "exploring upward", "strategy": "seeking battles", "battle_expectation": true}}

Available actions: up, down, left, right, enter, space, esc, z, x, a, s, 1, 2, 3""",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/generate",  # chat 대신 generate 사용
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        answer = result.get("response", "")
                        
                        # 대화 기록
                        self.conversation_history.append({
                            "prompt": prompt,
                            "response": answer,
                            "timestamp": time.time()
                        })
                        
                        print(f"🧠 LLM 응답: {answer[:100]}...")
                        return answer
                    else:
                        error_text = await response.text()
                        print(f"❌ LLM 오류 {response.status}: {error_text[:100]}")
                        return ""
                        
        except asyncio.TimeoutError:
            print(f"⏱️ LLM 응답 시간 초과")
            return ""
        except Exception as e:
            print(f"❌ LLM 연결 실패: {e}")
            return ""

class WebLearner:
    """웹에서 영웅전설4 정보 학습"""
    
    def __init__(self):
        self.knowledge_db = sqlite3.connect("hero4_knowledge.db")
        self.init_knowledge_db()
        
    def init_knowledge_db(self):
        """지식 DB 초기화"""
        cursor = self.knowledge_db.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hero4_knowledge (
                id INTEGER PRIMARY KEY,
                topic TEXT,
                content TEXT,
                source_url TEXT,
                learned_at REAL,
                relevance_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS battle_strategies (
                id INTEGER PRIMARY KEY,
                situation TEXT,
                strategy TEXT,
                success_rate REAL,
                source TEXT
            )
        ''')
        
        self.knowledge_db.commit()
        print("📚 영웅전설4 지식 DB 준비")
    
    async def search_hero4_info(self, query: str) -> List[str]:
        """영웅전설4 관련 정보 웹 검색"""
        search_queries = [
            f"영웅전설4 {query}",
            f"Legend of Heroes 4 {query}",
            f"ED4 {query} 공략",
            f"영웅전설4 전투 {query}"
        ]
        
        knowledge = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for search_query in search_queries[:2]:  # 2개만 검색 (속도)
                    # 간단한 검색 (실제로는 더 정교한 검색 API 사용)
                    search_url = f"https://www.google.com/search?q={search_query}"
                    
                    try:
                        async with session.get(
                            search_url,
                            headers={"User-Agent": "Mozilla/5.0"},
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')
                                
                                # 간단한 텍스트 추출
                                texts = soup.find_all('p')
                                for text in texts[:3]:  # 처음 3개만
                                    content = text.get_text().strip()
                                    if len(content) > 50 and '영웅전설' in content:
                                        knowledge.append(content)
                                        
                    except Exception as e:
                        print(f"⚠️ 검색 오류: {e}")
                        continue
                    
                    await asyncio.sleep(1)  # 요청 간격
                    
        except Exception as e:
            print(f"⚠️ 웹 학습 오류: {e}")
        
        # DB 저장
        cursor = self.knowledge_db.cursor()
        for info in knowledge:
            cursor.execute('''
                INSERT INTO hero4_knowledge (topic, content, learned_at, relevance_score)
                VALUES (?, ?, ?, ?)
            ''', (query, info, time.time(), 0.8))
        
        self.knowledge_db.commit()
        print(f"📖 '{query}' 관련 지식 {len(knowledge)}개 학습")
        
        return knowledge

class Hero4GameController:
    """영웅전설4 전용 게임 제어 - 완전 독립형"""
    
    def __init__(self):
        self.hero4_hwnd = None
        self.last_action_time = 0
        self.window_title = ""
        self.is_connected = False
        
        # 영웅전설4 키 맵핑
        self.hero4_keys = {
            'up': win32con.VK_UP, 'down': win32con.VK_DOWN,
            'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT,
            'enter': win32con.VK_RETURN, 'space': win32con.VK_SPACE,
            'esc': win32con.VK_ESCAPE, 'z': ord('Z'), 'x': ord('X'),
            'c': ord('C'), 'a': ord('A'), 's': ord('S'),
            '1': ord('1'), '2': ord('2'), '3': ord('3'),
            'tab': win32con.VK_TAB, 'f1': win32con.VK_F1, 'f2': win32con.VK_F2
        }
        
    def find_hero4_window_exclusive(self) -> bool:
        """영웅전설4만 정확히 찾기 - 다른 프로그램 배제"""
        def enum_callback(hwnd, windows):
            if not win32gui.IsWindowVisible(hwnd):
                return True
                
            try:
                title = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                
                # 영웅전설4 전용 식별자 (매우 엄격)
                hero4_exact_match = [
                    'ed4' in title.lower(),
                    '영웅전설' in title,
                    ('legend' in title.lower() and 'hero' in title.lower()),
                    ('dosbox' in title.lower() and 
                     any(x in title.lower() for x in ['ed4', 'hero', 'legend', '영웅전설'])),
                    (class_name == 'SDL_app' and 'dosbox' in title.lower())
                ]
                
                if any(hero4_exact_match):
                    # 추가 검증: 프로세스 확인
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        import psutil
                        process = psutil.Process(pid)
                        process_name = process.name().lower()
                        
                        # DOSBox 계열만 허용 (다른 프로그램 차단)
                        if ('dosbox' in process_name or 
                            'sdl' in process_name):
                            
                            # 창 크기로 한번 더 검증 (너무 작으면 제외)
                            rect = win32gui.GetWindowRect(hwnd)
                            width = rect[2] - rect[0]
                            height = rect[3] - rect[1]
                            
                            if width > 300 and height > 200:
                                windows.append((hwnd, title, pid, class_name, process_name, width * height))
                                
                    except Exception as e:
                        # 프로세스 정보 없어도 타이틀이 명확하면 허용
                        if 'ed4' in title.lower() or '영웅전설' in title:
                            windows.append((hwnd, title, 0, class_name, 'unknown', 0))
                            
            except Exception:
                pass
            return True
        
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if not windows:
            print("❌ 영웅전설4를 찾을 수 없습니다!")
            print("💡 DOSBox로 영웅전설4(ED4)를 실행해주세요.")
            return False
        
        # 가장 큰 창을 메인 게임으로 선택
        best_window = max(windows, key=lambda x: x[5])  # 면적 기준
        
        self.hero4_hwnd, self.window_title, pid, class_name, process_name, area = best_window
        self.is_connected = True
        
        print(f"� 영웅전설4 전용 연결!")
        print(f"   📝 게임: {self.window_title}")
        print(f"   🏷️ 클래스: {class_name}")
        print(f"   ⚙️ 프로세스: {process_name}")
        print(f"   📐 크기: {area}px²")
        print(f"   🔒 독립 모드: ON")
        return True
    
    def send_key_to_hero4_only(self, key: str) -> bool:
        """영웅전설4에만 키 전송 - 다른 창에는 절대 전송 안함"""
        if not self.is_connected or not self.hero4_hwnd or key not in self.hero4_keys:
            return False
        
        # 연결 상태 재확인
        try:
            if not win32gui.IsWindow(self.hero4_hwnd):
                print("⚠️ 영웅전설4 창이 사라졌습니다!")
                self.is_connected = False
                return False
        except:
            self.is_connected = False
            return False
        
        # 키 입력 간격 제한
        current_time = time.time()
        if current_time - self.last_action_time < 0.15:
            return False
        
        try:
            # 영웅전설4 창만 정확히 타겟팅
            current_fg = win32gui.GetForegroundWindow()
            current_title = win32gui.GetWindowText(current_fg) if current_fg else ""
            
            # 영웅전설4가 활성창이 아니면 강제 활성화
            if current_fg != self.hero4_hwnd:
                try:
                    # 최소화 상태면 복원
                    if win32gui.IsIconic(self.hero4_hwnd):
                        win32gui.ShowWindow(self.hero4_hwnd, win32con.SW_RESTORE)
                    
                    # 최상위로 가져오기
                    win32gui.BringWindowToTop(self.hero4_hwnd)
                    win32gui.SetForegroundWindow(self.hero4_hwnd)
                    time.sleep(0.08)  # 활성화 대기
                    
                    # 활성화 확인
                    new_fg = win32gui.GetForegroundWindow()
                    if new_fg != self.hero4_hwnd:
                        print(f"⚠️ 영웅전설4 활성화 실패. 현재: {win32gui.GetWindowText(new_fg)}")
                        return False
                        
                except Exception as e:
                    print(f"⚠️ 창 활성화 실패: {e}")
                    return False
            
            # 영웅전설4에만 키 전송
            vk_code = self.hero4_keys[key]
            win32api.keybd_event(vk_code, 0, 0, 0)
            time.sleep(0.08)
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            
            self.last_action_time = current_time
            print(f"🎮 영웅전설4 전용 입력: {key.upper()}")
            return True
            
        except Exception as e:
            print(f"❌ 영웅전설4 키 입력 실패: {e}")
            return False
    
    def verify_hero4_exclusive_connection(self) -> bool:
        """영웅전설4 전용 연결 상태 확인"""
        if not self.hero4_hwnd:
            return False
        
        try:
            # 창 존재 확인
            if not win32gui.IsWindow(self.hero4_hwnd):
                return False
            
            # 창 제목 재확인 (다른 프로그램으로 바뀌지 않았는지)
            current_title = win32gui.GetWindowText(self.hero4_hwnd)
            hero4_indicators = ['ed4', '영웅전설', 'legend', 'hero', 'dosbox']
            
            if not any(indicator in current_title.lower() for indicator in hero4_indicators):
                print("⚠️ 창 제목이 변경됨. 영웅전설4가 아닐 수 있습니다.")
                return False
                
            return win32gui.IsWindowVisible(self.hero4_hwnd)
            
        except Exception:
            return False

class GameScreenAnalyzer:
    """게임 화면 분석"""
    
    def __init__(self):
        self.capture_region = None
        self.last_screenshot = None
        
    def setup_capture_region(self, hwnd) -> bool:
        """캡처 영역 설정"""
        try:
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            
            # 게임 영역만 캡처 (DOSBox 테두리 제외)
            self.capture_region = (x + 10, y + 30, x2 - x - 20, y2 - y - 40)
            print(f"📸 캡처 영역: {self.capture_region}")
            return True
            
        except Exception as e:
            print(f"❌ 캡처 설정 실패: {e}")
            return False
    
    def capture_and_analyze(self) -> GameSituation:
        """화면 캡처 및 분석"""
        try:
            # 화면 캡처
            screenshot = pyautogui.screenshot(region=self.capture_region)
            self.last_screenshot = screenshot
            
            # 이미지를 base64로 변환 (LLM 전송용)
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # 기본 분석
            image = np.array(screenshot)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 화면 타입 추정
            brightness = np.mean(gray)
            
            if brightness > 120:
                screen_type = "menu_or_dialogue"
                description = "밝은 화면 - 메뉴나 대화창으로 추정"
                actions = ["enter", "space", "esc", "z"]
            elif brightness < 60:
                screen_type = "dark_area"
                description = "어두운 화면 - 던전이나 야외 필드"
                actions = ["up", "down", "left", "right", "space"]
            else:
                screen_type = "normal_field"
                description = "일반 필드 화면"
                actions = ["up", "down", "left", "right", "enter", "space"]
            
            # 전투 징후 감지 (빨간색 많으면 전투 가능성)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            red_ratio = np.sum(red_mask > 0) / red_mask.size
            
            battle_indication = red_ratio > 0.05
            
            return GameSituation(
                screen_type=screen_type,
                description=description,
                possible_actions=actions,
                screenshot_b64=img_b64,
                battle_count=0,  # 외부에서 관리
                movement_pattern="exploring"
            )
            
        except Exception as e:
            print(f"❌ 화면 분석 실패: {e}")
            return None

class IntelligentHero4AI:
    """LLM 기반 지능형 영웅전설4 AI"""
    
    def __init__(self):
        self.llm = LLMConnector()
        self.web_learner = WebLearner()
        self.controller = Hero4GameController()
        self.analyzer = GameScreenAnalyzer()
        
        # 게임 상태 추적
        self.battle_count = 0
        self.movement_history = []
        self.session_start = time.time()
        self.total_actions = 0
        self.last_screen_hash = None
        
        print("🤖 지능형 영웅전설4 AI 초기화")
    
    async def initialize(self) -> bool:
        """시스템 초기화"""
        print("🚀 지능형 AI 초기화 중...")
        
        # 1. 영웅전설4 전용 연결
        if not self.controller.find_hero4_window_exclusive():
            return False
        
        # 2. 영웅전설4 화면 분석 설정
        if not self.analyzer.setup_capture_region(self.controller.hero4_hwnd):
            return False
        
        # 3. 초기 지식 학습
        print("📚 영웅전설4 정보 학습 중...")
        await self.web_learner.search_hero4_info("전투 방법")
        await self.web_learner.search_hero4_info("이동 조작법")
        
        print("✅ 지능형 AI 준비 완료!")
        return True
    
    async def intelligent_game_step(self) -> Dict[str, Any]:
        """지능형 게임 스텝"""
        try:
            # 1. 현재 상황 분석
            situation = self.analyzer.capture_and_analyze()
            if not situation:
                return {"success": False, "error": "화면 분석 실패"}
            
            situation.battle_count = self.battle_count
            
            # 2. LLM에 상황 설명 및 행동 요청
            prompt = f"""
현재 영웅전설4 게임 상황:
- 화면 타입: {situation.screen_type}
- 상황 설명: {situation.description}
- 현재까지 전투 횟수: {self.battle_count}
- 총 행동 횟수: {self.total_actions}
- 가능한 행동: {situation.possible_actions}

목표: 좌우로 이동하며 전투를 10회 이상 경험하기
현재 전투 {self.battle_count}회 완료, {10 - self.battle_count}회 더 필요

최적의 다음 행동을 JSON으로 알려주세요.
"""
            
            # 3. LLM 추론
            response = await self.llm.query_llm(prompt, situation.screenshot_b64)
            
            # 4. JSON 파싱
            try:
                # LLM 응답에서 JSON 추출
                if "```json" in response:
                    json_part = response.split("```json")[1].split("```")[0]
                else:
                    json_part = response
                
                decision = json.loads(json_part)
                action = decision.get("action", "space")
                reason = decision.get("reason", "기본 행동")
                strategy = decision.get("strategy", "탐험")
                battle_expected = decision.get("battle_expectation", False)
                
            except Exception as e:
                print(f"⚠️ JSON 파싱 실패: {e}")
                # 폴백: 간단한 규칙 기반
                if "menu" in situation.screen_type:
                    action = "esc"
                    reason = "메뉴 탈출"
                else:
                    action = "right" if self.total_actions % 2 == 0 else "left"
                    reason = "좌우 이동 탐험"
                
                strategy = "기본 전략"
                battle_expected = False
            
            # 5. 영웅전설4 연결 확인 후 행동 실행
            if not self.controller.verify_hero4_exclusive_connection():
                return {"success": False, "error": "영웅전설4 연결 끊어짐"}
            
            success = self.controller.send_key_to_hero4_only(action)
            
            # 6. 빠른 결과 대기 (속도 향상)
            await asyncio.sleep(0.15)
            
            # 7. 실제 화면 인식 및 전투 감지
            result_situation = self.analyzer.capture_and_analyze()
            real_battle_detected = False
            
            if result_situation:
                # 실제 전투 화면 감지 (더 정확하게)
                current_image = np.array(self.analyzer.last_screenshot) if self.analyzer.last_screenshot else None
                
                if current_image is not None:
                    # HSV로 색상 분석
                    hsv = cv2.cvtColor(current_image, cv2.COLOR_RGB2HSV)
                    
                    # 빨간색 (HP 바, 데미지 등) 많으면 전투
                    red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                    red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
                    red_ratio = (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / current_image.size
                    
                    # 파란색 (마나, UI) 많으면 전투/메뉴
                    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
                    blue_ratio = np.sum(blue_mask > 0) / current_image.size
                    
                    # 실제 전투 조건 (더 엄격)
                    if (red_ratio > 0.08 or  # 빨간색 8% 이상
                        blue_ratio > 0.15 or  # 파란색 15% 이상  
                        "battle" in result_situation.screen_type.lower() or
                        (battle_expected and np.mean(current_image) > 80)):
                        
                        real_battle_detected = True
                        self.battle_count += 1
                        print(f"⚔️ 실제 전투 감지! (빨강:{red_ratio:.3f}, 파랑:{blue_ratio:.3f}) 총 {self.battle_count}회")
                        
                        # 진짜 전투 행동
                        battle_sequence = ['z', 'enter', 'space', 'a', '1', '2']
                        battle_action = battle_sequence[(self.battle_count - 1) % len(battle_sequence)]
                        await asyncio.sleep(0.1)
                        self.controller.send_key_to_hero4_only(battle_action)
                        print(f"🔥 전투 액션: {battle_action}")
                        await asyncio.sleep(0.2)
            
            # 8. 막힘 감지 및 탈출 (중요!)
            if not real_battle_detected:
                # 같은 화면에 오래 머물면 탈출 시도
                if (hasattr(self, 'last_screen_hash') and 
                    self.last_screen_hash == result_situation.screen_type and 
                    self.total_actions % 5 == 0):
                    
                    # 막힘 탈출 시퀀스
                    escape_actions = ['up', 'down', 'esc', 'space', 'enter']
                    escape_action = escape_actions[self.total_actions % len(escape_actions)]
                    print(f"🚫 막힘 감지! 탈출 시도: {escape_action}")
                    self.controller.send_key_to_hero4_only(escape_action)
                    await asyncio.sleep(0.1)
                
                # 이동 패턴 다양화 (오른쪽만 가지 않도록)
                elif action in ['left', 'right']:
                    movement_patterns = [
                        ['left', 'up', 'right', 'down'],     # 사각형 이동
                        ['right', 'up', 'left', 'down'],    # 역사각형  
                        ['up', 'right', 'down', 'left'],    # 십자 이동
                        ['down', 'left', 'up', 'right']     # 역십자
                    ]
                    
                    pattern = movement_patterns[self.total_actions // 10 % len(movement_patterns)]
                    next_move = pattern[self.total_actions % len(pattern)]
                    
                    if next_move != action:  # 다른 방향으로 변경
                        print(f"🔄 패턴 이동: {action} → {next_move}")
                        self.controller.send_key_to_hero4_only(next_move)
                        await asyncio.sleep(0.1)
            
            # 9. 화면 해시 저장 (막힘 감지용)
            self.last_screen_hash = result_situation.screen_type if result_situation else None
            
            # 9. 이동 기록
            if action in ["left", "right", "up", "down"]:
                self.movement_history.append(action)
            
            self.total_actions += 1
            
            return {
                "success": success,
                "action": action,
                "reason": reason,
                "strategy": strategy,
                "battle_count": self.battle_count,
                "total_actions": self.total_actions,
                "screen_type": situation.screen_type,
                "llm_response": response[:100] + "..." if len(response) > 100 else response
            }
            
        except Exception as e:
            return {"success": False, "error": f"게임 스텝 오류: {e}"}

async def main():
    """메인 실행"""
    print("🎮 영웅전설4 전용 LLM 지능형 AI")
    print("=" * 60)
    print("🎯 목표: 좌우 이동하며 전투 10회 이상 달성")
    print("🧠 LLM: Ollama 로컬 모델 사용")
    print("🌐 학습: 실시간 웹 정보 수집")
    print("🔒 독립 모드: 영웅전설4에만 작동")
    
    ai = IntelligentHero4AI()
    
    # 초기화
    if not await ai.initialize():
        return
    
    print("\n🚀 지능형 게임플레이 시작!")
    
    # 게임 플레이 루프
    max_actions = 200  # 최대 200액션
    start_time = time.time()
    
    try:
        for step in range(1, max_actions + 1):
            result = await ai.intelligent_game_step()
            
            if result["success"]:
                print(f"✅ #{step:3d} | {result['action']:6s} | {result['reason']:20s} | "
                      f"전투:{result['battle_count']:2d}/10 | {result['screen_type']}")
                
                # 전략 출력
                if step % 10 == 0:
                    print(f"    🧠 전략: {result['strategy']}")
                    print(f"    💭 LLM: {result['llm_response']}")
                
                # 목표 달성 확인
                if result['battle_count'] >= 10:
                    elapsed = time.time() - start_time
                    print(f"\n🎉 목표 달성! 전투 {result['battle_count']}회 완료!")
                    print(f"⏱️ 소요 시간: {elapsed:.1f}초 ({step}액션)")
                    print(f"🎯 전투 효율: {result['battle_count']/step:.3f} (액션당 전투율)")
                    break
            
            else:
                print(f"❌ #{step:3d} 실패: {result.get('error', 'unknown')}")
            
            # 5액션마다 상태 리포트
            if step % 5 == 0:
                elapsed = time.time() - start_time
                aps = step / elapsed  # Actions Per Second
                print(f"📊 진행: {step}/{max_actions} | "
                      f"전투:{ai.battle_count}/10 | "
                      f"속도:{aps:.1f}aps")
            
            # 적절한 대기
            await asyncio.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    
    # 최종 결과
    elapsed = time.time() - start_time
    print(f"\n📊 최종 결과:")
    print(f"    ⏱️ 플레이 시간: {elapsed:.1f}초")
    print(f"    🎮 총 액션: {ai.total_actions}개")
    print(f"    ⚔️ 전투 횟수: {ai.battle_count}회")
    print(f"    🎯 목표 달성: {'✅' if ai.battle_count >= 10 else '❌'}")
    
    if ai.battle_count >= 10:
        print(f"🏆 성공! LLM AI가 영웅전설4를 지능적으로 플레이했습니다!")
    else:
        print(f"🌱 진행 중... 더 많은 시간이 필요합니다.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 AI 종료")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()