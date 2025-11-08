#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import sqlite3
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

# 웹 크롤링 관련
import requests
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# AI 및 임베딩
try:
    from sentence_transformers import SentenceTransformer
    import torch
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ sentence-transformers 없음. 임베딩 기능 비활성화")

# 게임 제어
import cv2
import numpy as np
import pyautogui
import win32gui
import win32con
import win32api

@dataclass
class GameKnowledge:
    """게임 지식 구조"""
    topic: str
    content: str
    source: str
    confidence: float
    timestamp: datetime
    usage_count: int = 0
    effectiveness: float = 0.0

@dataclass
class GameAction:
    """게임 액션 구조"""
    action_type: str
    keys: List[str]
    description: str
    success_rate: float = 0.0
    usage_count: int = 0

class WebKnowledgeGatherer:
    """웹에서 게임 지식을 수집하는 클래스"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 영웅전설4 관련 검색 키워드
        self.search_keywords = [
            "영웅전설4 백의 마녀 공략",
            "영웅전설4 조작법",
            "영웅전설4 아이템",
            "영웅전설4 스킬",
            "영웅전설4 전투",
            "영웅전설4 퀘스트",
            "영웅전설4 캐릭터",
            "Legend of Heroes 4 guide",
            "白き魔女 攻略"
        ]
        
    def search_game_info(self, query: str, max_results: int = 3) -> List[Dict]:
        """게임 정보 검색"""
        try:
            # 네이버 블로그 검색
            search_url = f"https://search.naver.com/search.naver?where=blog&query={query}"
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for link in soup.find_all('a', href=True)[:max_results]:
                if 'blog.naver.com' in link['href'] or 'tistory.com' in link['href']:
                    results.append({
                        'url': link['href'],
                        'title': link.get_text(strip=True)[:100]
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return []
    
    def extract_game_knowledge(self, url: str) -> List[GameKnowledge]:
        """URL에서 게임 지식 추출"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 텍스트 추출
            text = soup.get_text()
            
            # 게임 관련 핵심 정보 추출
            knowledge_list = []
            
            # 조작법 관련
            if any(word in text for word in ['방향키', '엔터', '스페이스', '조작', '키보드']):
                knowledge_list.append(GameKnowledge(
                    topic="controls",
                    content=f"조작법 정보: {text[:200]}...",
                    source=url,
                    confidence=0.8,
                    timestamp=datetime.now()
                ))
            
            # 전투 관련
            if any(word in text for word in ['전투', '스킬', '마법', '공격', '방어']):
                knowledge_list.append(GameKnowledge(
                    topic="combat",
                    content=f"전투 정보: {text[:200]}...",
                    source=url,
                    confidence=0.7,
                    timestamp=datetime.now()
                ))
            
            # 아이템 관련
            if any(word in text for word in ['아이템', '장비', '무기', '방어구', '소모품']):
                knowledge_list.append(GameKnowledge(
                    topic="items",
                    content=f"아이템 정보: {text[:200]}...",
                    source=url,
                    confidence=0.7,
                    timestamp=datetime.now()
                ))
            
            return knowledge_list
            
        except Exception as e:
            print(f"❌ 지식 추출 오류 ({url}): {e}")
            return []

class GameVision:
    """게임 화면 분석"""
    
    def __init__(self):
        self.window_title = "DOSBox"
        
    def get_game_window(self):
        """게임 윈도우 핸들 찾기"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.window_title in title:
                    windows.append(hwnd)
            return True
            
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        return windows[0] if windows else None
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """게임 화면 캡처"""
        try:
            hwnd = self.get_game_window()
            if not hwnd:
                return None
                
            rect = win32gui.GetWindowRect(hwnd)
            screenshot = pyautogui.screenshot(region=(rect[0], rect[1], 
                                                   rect[2]-rect[0], rect[3]-rect[1]))
            return np.array(screenshot)
            
        except Exception as e:
            print(f"❌ 화면 캡처 오류: {e}")
            return None
    
    def analyze_game_state(self, image: np.ndarray) -> Dict:
        """게임 상태 분석"""
        try:
            # 간단한 색상 기반 분석
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 메뉴 화면 감지 (특정 색상 패턴)
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            menu_ratio = np.sum(blue_mask > 0) / blue_mask.size
            
            # 전투 화면 감지
            red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            combat_ratio = np.sum(red_mask > 0) / red_mask.size
            
            # 텍스트 박스 감지
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # 적당한 크기의 사각형
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > h and w > 100:  # 텍스트 박스 형태
                        text_boxes.append((x, y, w, h))
            
            return {
                'menu_detected': menu_ratio > 0.1,
                'combat_detected': combat_ratio > 0.05,
                'text_boxes': len(text_boxes),
                'menu_ratio': menu_ratio,
                'combat_ratio': combat_ratio
            }
            
        except Exception as e:
            print(f"❌ 게임 상태 분석 오류: {e}")
            return {'menu_detected': False, 'combat_detected': False, 'text_boxes': 0}

class GameController:
    """게임 제어"""
    
    def __init__(self):
        self.window_title = "DOSBox"
        
    def send_key(self, key: str, duration: float = 0.1):
        """키 입력"""
        try:
            hwnd = win32gui.FindWindow(None, self.window_title)
            if hwnd:
                win32gui.SetForegroundWindow(hwnd)
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
                
                if key in key_map:
                    win32api.keybd_event(key_map[key], 0, 0, 0)
                    time.sleep(duration)
                    win32api.keybd_event(key_map[key], 0, win32con.KEYEVENTF_KEYUP, 0)
                    return True
                    
        except Exception as e:
            print(f"❌ 키 입력 오류 ({key}): {e}")
            return False

class SmartGameAI:
    """웹 지식 기반 게임 AI"""
    
    def __init__(self):
        self.vision = GameVision()
        self.controller = GameController()
        self.web_gatherer = WebKnowledgeGatherer()
        
        # 지식 데이터베이스
        self.init_database()
        
        # 임베딩 모델 (선택적)
        self.embedder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                print("✅ 임베딩 모델 로드 성공")
            except Exception as e:
                print(f"⚠️ 임베딩 모델 로드 실패: {e}")
        
        # 기본 액션 정의
        self.base_actions = [
            GameAction("move_up", ["up"], "위로 이동", 0.5, 0),
            GameAction("move_down", ["down"], "아래로 이동", 0.5, 0),
            GameAction("move_left", ["left"], "왼쪽으로 이동", 0.5, 0),
            GameAction("move_right", ["right"], "오른쪽으로 이동", 0.5, 0),
            GameAction("confirm", ["enter"], "확인/선택", 0.6, 0),
            GameAction("cancel", ["esc"], "취소/뒤로", 0.4, 0),
            GameAction("action", ["space"], "액션/조사", 0.5, 0)
        ]
        
    def init_database(self):
        """데이터베이스 초기화"""
        self.conn = sqlite3.connect('game_knowledge.db')
        cursor = self.conn.cursor()
        
        # 지식 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                topic TEXT,
                content TEXT,
                source TEXT,
                confidence REAL,
                timestamp TEXT,
                usage_count INTEGER DEFAULT 0,
                effectiveness REAL DEFAULT 0.0
            )
        ''')
        
        # 액션 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY,
                action_type TEXT,
                keys TEXT,
                description TEXT,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0
            )
        ''')
        
        self.conn.commit()
    
    def store_knowledge(self, knowledge: GameKnowledge):
        """지식 저장"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO knowledge 
            (topic, content, source, confidence, timestamp, usage_count, effectiveness)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (knowledge.topic, knowledge.content, knowledge.source, 
              knowledge.confidence, knowledge.timestamp.isoformat(),
              knowledge.usage_count, knowledge.effectiveness))
        self.conn.commit()
    
    def get_relevant_knowledge(self, context: str, limit: int = 5) -> List[GameKnowledge]:
        """관련 지식 검색"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM knowledge 
            WHERE content LIKE ? 
            ORDER BY confidence DESC, usage_count DESC 
            LIMIT ?
        ''', (f'%{context}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append(GameKnowledge(
                topic=row[1], content=row[2], source=row[3],
                confidence=row[4], timestamp=datetime.fromisoformat(row[5]),
                usage_count=row[6], effectiveness=row[7]
            ))
        return results
    
    async def autonomous_web_learning(self):
        """자율적 웹 학습"""
        print("🌐 웹에서 게임 정보 수집 중...")
        
        learned_count = 0
        for keyword in self.web_gatherer.search_keywords:
            try:
                print(f"🔍 검색 중: {keyword}")
                search_results = self.web_gatherer.search_game_info(keyword)
                
                for result in search_results:
                    knowledge_list = self.web_gatherer.extract_game_knowledge(result['url'])
                    for knowledge in knowledge_list:
                        self.store_knowledge(knowledge)
                        learned_count += 1
                        print(f"📚 학습: {knowledge.topic} ({knowledge.confidence:.2f})")
                
                # 요청 간격
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ 웹 학습 오류 ({keyword}): {e}")
        
        print(f"✅ 웹 학습 완료: {learned_count}개 지식 수집")
        return learned_count
    
    def choose_action(self, game_state: Dict) -> GameAction:
        """상황에 맞는 액션 선택"""
        try:
            # 관련 지식 검색
            if game_state.get('menu_detected'):
                knowledge = self.get_relevant_knowledge("메뉴")
                context = "menu"
            elif game_state.get('combat_detected'):
                knowledge = self.get_relevant_knowledge("전투")
                context = "combat"
            else:
                knowledge = self.get_relevant_knowledge("이동")
                context = "exploration"
            
            # 지식 기반 액션 선택
            if knowledge:
                # 지식을 바탕으로 최적 액션 결정
                for k in knowledge:
                    if "확인" in k.content or "선택" in k.content:
                        return self.base_actions[4]  # confirm
                    elif "이동" in k.content:
                        return random.choice(self.base_actions[:4])  # movement
            
            # 기본 전략
            if game_state.get('text_boxes', 0) > 0:
                return self.base_actions[4]  # confirm - 텍스트 진행
            elif game_state.get('menu_detected'):
                return random.choice([self.base_actions[0], self.base_actions[1], self.base_actions[4]])
            else:
                return random.choice(self.base_actions)  # 랜덤 탐색
                
        except Exception as e:
            print(f"❌ 액션 선택 오류: {e}")
            return random.choice(self.base_actions)
    
    async def play_game_step(self) -> Dict:
        """게임 1스텝 실행"""
        try:
            # 화면 캡처 및 분석
            screenshot = self.vision.capture_screen()
            if screenshot is None:
                return {'success': False, 'error': '화면 캡처 실패'}
            
            game_state = self.vision.analyze_game_state(screenshot)
            
            # 액션 선택 및 실행
            action = self.choose_action(game_state)
            
            print(f"🎮 상태: {game_state}")
            print(f"🎯 액션: {action.description}")
            
            # 키 입력
            success = True
            for key in action.keys:
                if not self.controller.send_key(key):
                    success = False
                    break
                await asyncio.sleep(0.2)
            
            # 결과 대기
            await asyncio.sleep(1.0)
            
            # 액션 통계 업데이트
            action.usage_count += 1
            if success:
                action.success_rate = (action.success_rate * (action.usage_count - 1) + 1.0) / action.usage_count
            else:
                action.success_rate = (action.success_rate * (action.usage_count - 1)) / action.usage_count
            
            return {
                'success': success,
                'action': action.action_type,
                'game_state': game_state,
                'action_success_rate': action.success_rate
            }
            
        except Exception as e:
            print(f"❌ 게임 스텝 오류: {e}")
            return {'success': False, 'error': str(e)}

async def main():
    """메인 실행 함수"""
    print("🌐 웹 기반 자율학습 영웅전설4 AI")
    print("=" * 50)
    
    # AI 초기화
    ai = SmartGameAI()
    
    # 웹 학습 실행
    print("📚 초기 웹 학습 시작...")
    await ai.autonomous_web_learning()
    
    print("\n🎮 게임 플레이 시작!")
    print("게임이 실행되어 있는지 확인하세요 (DOSBox 윈도우)")
    
    # 게임 플레이
    total_steps = 30  # 30스텝 실행
    success_count = 0
    
    for step in range(1, total_steps + 1):
        print(f"\n--- 스텝 {step}/{total_steps} ---")
        
        result = await ai.play_game_step()
        
        if result['success']:
            success_count += 1
            print(f"✅ 성공!")
        else:
            print(f"❌ 실패: {result.get('error', '알 수 없음')}")
        
        # 통계 출력
        success_rate = success_count / step
        print(f"📊 성공률: {success_rate:.2f} ({success_count}/{step})")
        
        # 주기적 추가 학습
        if step % 10 == 0:
            print("🔄 추가 웹 학습 중...")
            await ai.autonomous_web_learning()
        
        # 잠깐 대기
        await asyncio.sleep(0.5)
    
    print(f"\n🎯 최종 결과:")
    print(f"총 스텝: {total_steps}")
    print(f"성공: {success_count}")
    print(f"성공률: {success_count/total_steps:.2f}")
    
    ai.conn.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        traceback.print_exc()