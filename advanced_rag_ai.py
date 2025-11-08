"""
고도화된 RAG AI 시스템 - 영웅전설4 전용 AI
- 완전 독립 실행
- 고속 학습 및 경험 축적
- 격리된 윈도우 제어
- 무제한 전투 모드
"""

import asyncio
import aiohttp
import json
import time
import threading
import queue
from collections import deque
from typing import Dict, List, Optional
import hashlib
import sqlite3
import numpy as np
import cv2
import win32gui
import win32con
from PIL import ImageGrab
import os

# 간단 설정 (필요시 환경변수로 제어 가능)
CONFIG = {
    'enable_llm': bool(int(os.environ.get('HERO4_ENABLE_LLM', '0'))),  # 1로 설정시 활성화
    'llm_provider': 'ollama',
    'ollama_url': os.environ.get('OLLAMA_URL', 'http://localhost:11434'),
    'model_name': os.environ.get('HERO4_MODEL', 'qwen2.5-coder:7b'),
    'log_every_steps': int(os.environ.get('HERO4_LOG_STEPS', '5'))
}


class AdvancedRAGDatabase:
    """고도화된 RAG 데이터베이스"""
    
    def __init__(self):
        """초기화"""
        self.db_path = os.path.join(os.path.dirname(__file__), 'advanced_rag_data.db')
        self.batch_queue = queue.Queue()
        self.batch_size = 15  # 배치 크기 증가
        self.batch_thread = None
        self.running = False
        
        # 데이터베이스 초기화
        self._init_database()
        self._start_batch_processor()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # WAL 모드 활성화
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                
                # 고급 테이블 구조
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS advanced_experiences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        screen_hash TEXT,
                        screen_description TEXT,
                        brightness REAL,
                        color_ratios TEXT,
                        situation_type TEXT,
                        action TEXT,
                        reasoning TEXT,
                        confidence REAL,
                        success_score REAL,
                        battle_detected INTEGER,
                        battle_count INTEGER,
                        reward REAL,
                        episode INTEGER,
                        timestamp REAL,
                        session_id TEXT,
                        learning_context TEXT
                    )
                """)
                
                # 성공 패턴 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS success_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_type TEXT,
                        trigger_conditions TEXT,
                        action_sequence TEXT,
                        success_rate REAL,
                        avg_reward REAL,
                        usage_count INTEGER,
                        last_used REAL
                    )
                """)
                
                # 전투 기록 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS battle_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        battle_id TEXT UNIQUE,
                        start_time REAL,
                        duration REAL,
                        actions_taken TEXT,
                        result TEXT,
                        total_reward REAL,
                        ai_performance REAL
                    )
                """)
                
                # 인덱스 생성
                conn.execute("CREATE INDEX IF NOT EXISTS idx_screen_hash ON advanced_experiences(screen_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_situation_type ON advanced_experiences(situation_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_battle_detected ON advanced_experiences(battle_detected)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_success_score ON advanced_experiences(success_score)")
                
                conn.commit()
                print("💾 고급 RAG 데이터베이스 초기화 완료")
                
        except Exception as e:
            print(f"❌ 데이터베이스 초기화 실패: {e}")
    
    def _start_batch_processor(self):
        """배치 처리기 시작"""
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
    
    def _batch_processor(self):
        """배치 처리 스레드"""
        batch = []
        
        while self.running:
            try:
                # 타임아웃으로 배치 수집
                try:
                    item = self.batch_queue.get(timeout=2.0)
                    batch.append(item)
                except queue.Empty:
                    if batch:
                        self._process_batch(batch)
                        batch = []
                    continue
                
                # 배치 크기 도달시 처리
                if len(batch) >= self.batch_size:
                    self._process_batch(batch)
                    batch = []
                    
            except Exception as e:
                print(f"❌ 배치 처리 오류: {e}")
                batch = []
    
    def _process_batch(self, batch: List[Dict]):
        """배치 데이터 처리"""
        if not batch:
            return
            
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                for item in batch:
                    if item['type'] == 'experience':
                        conn.execute("""
                            INSERT INTO advanced_experiences 
                            (screen_hash, screen_description, brightness, color_ratios, 
                             situation_type, action, reasoning, confidence, success_score, 
                             battle_detected, battle_count, reward, episode, timestamp, 
                             session_id, learning_context)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, item['data'])
                    
                    elif item['type'] == 'pattern':
                        conn.execute("""
                            INSERT OR REPLACE INTO success_patterns
                            (pattern_type, trigger_conditions, action_sequence, 
                             success_rate, avg_reward, usage_count, last_used)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, item['data'])
                
                conn.commit()
                print(f"💾 배치 저장: {len(batch)}개 항목")
                
        except Exception as e:
            print(f"❌ 배치 저장 실패: {e}")
    
    def add_advanced_experience(self, screen_data: Dict, ai_decision: Dict, 
                              result: Dict, episode: int, session_id: str):
        """고급 경험 추가"""
        
        # 화면 해시 생성
        screen_desc = screen_data.get('description', '')
        screen_hash = hashlib.md5(screen_desc.encode()).hexdigest()[:16]
        
        # 색상 비율 JSON 저장
        color_ratios = json.dumps({
            'red': screen_data.get('red_ratio', 0),
            'blue': screen_data.get('blue_ratio', 0),
            'green': screen_data.get('green_ratio', 0),
            'brightness': screen_data.get('brightness', 0)
        })
        
        # 학습 컨텍스트
        learning_context = json.dumps({
            'confidence_level': ai_decision.get('confidence', 0),
            'reasoning_type': ai_decision.get('reasoning_type', 'standard'),
            'rag_influence': ai_decision.get('rag_influence', 0.5),
            'action_history': ai_decision.get('recent_actions', [])
        })
        
        # 배치 큐에 추가
        experience_data = (
            screen_hash,
            screen_desc,
            screen_data.get('brightness', 0),
            color_ratios,
            ai_decision.get('situation_type', 'unknown'),
            ai_decision.get('action', 'right'),
            ai_decision.get('reasoning', ''),
            ai_decision.get('confidence', 0.5),
            result.get('success_score', 0.5),
            result.get('battle_detected', 0),
            result.get('battle_count', 0),
            result.get('reward', 0.1),
            episode,
            time.time(),
            session_id,
            learning_context
        )
        
        self.batch_queue.put({'type': 'experience', 'data': experience_data})
    
    def get_advanced_context(self, screen_data: Dict, situation_type: str, limit: int = 8) -> str:
        """고급 RAG 컨텍스트 생성"""
        try:
            screen_desc = screen_data.get('description', '')
            screen_hash = hashlib.md5(screen_desc.encode()).hexdigest()[:16]
            
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                # 유사 상황 경험 검색
                cursor = conn.execute("""
                    SELECT situation_type, action, reasoning, confidence, success_score, reward, battle_detected
                    FROM advanced_experiences 
                    WHERE (screen_hash = ? OR situation_type = ? OR screen_hash LIKE ?)
                    AND success_score > 0.3
                    ORDER BY success_score DESC, timestamp DESC
                    LIMIT ?
                """, (screen_hash, situation_type, f"{screen_hash[:3]}%", limit))
                
                experiences = cursor.fetchall()
                
                # 성공 패턴 검색
                cursor = conn.execute("""
                    SELECT pattern_type, action_sequence, success_rate, avg_reward
                    FROM success_patterns
                    WHERE pattern_type = ? OR pattern_type LIKE '%general%'
                    ORDER BY success_rate DESC
                    LIMIT 3
                """, (situation_type,))
                
                patterns = cursor.fetchall()
                
                # 통계 정보
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_exp,
                        AVG(success_score) as avg_success,
                        SUM(battle_detected) as total_battles,
                        AVG(reward) as avg_reward
                    FROM advanced_experiences
                    WHERE timestamp > ?
                """, (time.time() - 3600,))  # 최근 1시간
                
                stats = cursor.fetchone()
                # None 안전 처리
                if stats:
                    total_exp = stats[0] or 0
                    avg_success = stats[1] if (stats[1] is not None) else 0.0
                    total_battles = stats[2] or 0
                    avg_reward = stats[3] if (stats[3] is not None) else 0.0
                else:
                    total_exp = 0
                    avg_success = 0.0
                    total_battles = 0
                    avg_reward = 0.0
                
                # RAG 컨텍스트 구성
                context_parts = [
                    f"상황: {situation_type}",
                    f"경험: {len(experiences)}개 유사 상황"
                ]
                
                if experiences:
                    best_exp = experiences[0]
                    context_parts.append(f"최고 성공: {best_exp[1]} (신뢰도 {best_exp[3]:.2f}, 점수 {best_exp[4]:.2f})")
                    
                    # 행동 분포
                    actions = [exp[1] for exp in experiences]
                    action_counts = {}
                    for action in actions:
                        action_counts[action] = action_counts.get(action, 0) + 1
                    
                    top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    context_parts.append(f"추천 행동: {', '.join([f'{a}({c})' for a, c in top_actions])}")
                
                if patterns:
                    best_pattern = patterns[0]
                    context_parts.append(f"성공 패턴: {best_pattern[1]} (성공률 {best_pattern[2]:.2f})")
                
                # 세션 통계 (None 안전)
                context_parts.append(
                    f"세션 통계: 경험 {total_exp}, 성공률 {avg_success:.2f}, 전투 {total_battles}, 평균보상 {avg_reward:.2f}"
                )
                
                return "\n".join(context_parts)
                
        except Exception as e:
            print(f"❌ RAG 컨텍스트 생성 실패: {e}")
            return f"상황: {situation_type}\n기본 컨텍스트 모드"
    
    def update_success_pattern(self, pattern_type: str, action_sequence: List[str], 
                             success_rate: float, reward: float):
        """성공 패턴 업데이트"""
        
        pattern_data = (
            pattern_type,
            json.dumps({'conditions': 'dynamic'}),
            json.dumps(action_sequence),
            success_rate,
            reward,
            1,  # usage_count
            time.time()
        )
        
        self.batch_queue.put({'type': 'pattern', 'data': pattern_data})


class SuperIsolatedController:
    """완전 격리된 게임 컨트롤러"""
    
    def __init__(self):
        """초기화"""
        self.running = False
        self.action_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.control_thread = None
        self.performance_stats = {
            'actions_sent': 0,
            'actions_successful': 0,
            'avg_response_time': 0.0,
            'focus_preserved_count': 0
        }
    
    def start_super_isolated_control(self):
        """완전 격리 제어 시작"""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_worker, daemon=True)
        self.control_thread.start()
        print("🔒 완전 격리 컨트롤러 시작")
    
    def stop_super_isolated_control(self):
        """격리 제어 중지"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        print("⏹️ 격리 컨트롤러 중지")
    
    def _control_worker(self):
        """제어 워커 스레드"""
        while self.running:
            try:
                action_data = self.action_queue.get(timeout=1.0)
                
                start_time = time.time()
                result = self._execute_super_isolated_action(action_data['action'])
                response_time = time.time() - start_time
                
                # 성능 통계 업데이트
                self.performance_stats['actions_sent'] += 1
                if result.get('success'):
                    self.performance_stats['actions_successful'] += 1
                
                # 평균 응답시간 업데이트
                current_avg = self.performance_stats['avg_response_time']
                total_actions = self.performance_stats['actions_sent']
                self.performance_stats['avg_response_time'] = (current_avg * (total_actions - 1) + response_time) / total_actions
                
                if result.get('focus_preserved'):
                    self.performance_stats['focus_preserved_count'] += 1
                
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.result_queue.put({'success': False, 'error': str(e)})
    
    def _find_dosbox(self) -> Optional[int]:
        """DOSBox 창 찾기"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd).lower()
                if 'dosbox' in window_title or 'dos' in window_title:
                    windows.append(hwnd)
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        return windows[0] if windows else None
    
    def _execute_super_isolated_action(self, action: str) -> Dict:
        """완전 격리된 액션 실행"""
        try:
            # DOSBox 창 찾기
            window_handle = self._find_dosbox()
            if not window_handle:
                return {'success': False, 'error': 'DOSBox window not found'}
            
            # 현재 포커스 저장
            try:
                original_focus = win32gui.GetForegroundWindow()
            except:
                original_focus = None
            
            # 키 매핑 (확장된)
            key_map = {
                'left': win32con.VK_LEFT,
                'right': win32con.VK_RIGHT,
                'up': win32con.VK_UP,
                'down': win32con.VK_DOWN,
                'space': win32con.VK_SPACE,
                'enter': win32con.VK_RETURN,
                'z': ord('Z'),
                'x': ord('X'),
                'a': ord('A'),
                's': ord('S'),
                'q': ord('Q'),
                'w': ord('W'),
                'e': ord('E'),
                '1': ord('1'),
                '2': ord('2'),
                '3': ord('3'),
                'esc': win32con.VK_ESCAPE,
                'tab': win32con.VK_TAB
            }
            
            if action in key_map:
                vk_code = key_map[action]
                
                # 고속 PostMessage 전송 (포커스 변경 절대 없음)
                WM_KEYDOWN = 0x0100
                WM_KEYUP = 0x0101
                
                # 키 이벤트 전송 (더 빠른 타이밍)
                win32gui.PostMessage(window_handle, WM_KEYDOWN, vk_code, 0)
                time.sleep(0.008)  # 더 빠른 타이밍
                win32gui.PostMessage(window_handle, WM_KEYUP, vk_code, 0)
                
                # 포커스 보존 확인 (필요시에만)
                focus_preserved = True
                if original_focus and original_focus != window_handle:
                    try:
                        current_focus = win32gui.GetForegroundWindow()
                        if current_focus != original_focus:
                            win32gui.SetForegroundWindow(original_focus)
                        focus_preserved = True
                    except:
                        focus_preserved = False
                
                return {
                    'success': True, 
                    'action': action, 
                    'focus_preserved': focus_preserved,
                    'window_handle': window_handle,
                    'response_time': time.time()
                }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Unknown action'}
    
    def send_rapid_action(self, action: str) -> bool:
        """고속 액션 전송"""
        if not self.running:
            return False
        
        try:
            self.action_queue.put({'action': action}, timeout=0.1)
            return True
        except queue.Full:
            return False
    
    def get_result_fast(self, timeout: float = 0.05) -> Optional[Dict]:
        """고속 결과 받기"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        return self.performance_stats.copy()


class MasterRAGAI:
    """마스터 RAG AI 시스템 - 최고 성능"""
    
    def __init__(self):
        """초기화"""
        self.rag_db = AdvancedRAGDatabase()
        self.controller = SuperIsolatedController()
        self.model_name = CONFIG['model_name']
        self.ollama_url = CONFIG['ollama_url']
        self.enable_llm = CONFIG['enable_llm']
        
        # 고급 상태 추적
        self.step_count = 0
        self.battle_count = 0
        self.session_battle_count = 0
        self.action_history = deque(maxlen=100)
        self.session_start = time.time()
        self.session_id = f"session_{int(time.time())}"
        
        # 고급 학습 통계
        self.advanced_stats = {
            'total_experiences': 0,
            'successful_actions': 0,
            'battle_discoveries': 0,
            'rag_queries': 0,
            'model_decisions': 0,
            'learning_episodes': 0,
            'avg_confidence': 0.0,
            'success_rate': 0.0,
            'battle_rate': 0.0,
            'decision_speed': 0.0
        }
        
        # 고급 학습 상태
        self.current_episode = 0
        self.episode_start_time = time.time()
        self.episode_actions = []
        self.episode_rewards = []
        self.recent_performance = deque(maxlen=50)
        
        # 동적 학습 파라미터
        self.learning_params = {
            'exploration_rate': 0.3,
            'confidence_threshold': 0.7,
            'rag_influence_weight': 0.6,
            'speed_mode': True,
            'battle_focus_mode': True
        }
        
        print("🚀 마스터 RAG AI 시스템 초기화")
        print("💾 고급 경험 데이터베이스 연결")
        print("🔒 완전 격리 컨트롤러 준비")
        print("📊 고급 학습 시스템 활성화")
        print("⚡ 최고 성능 모드 설정")
    
    async def master_rag_thinking(self, screen_data: Dict) -> Dict:
        """마스터 RAG 사고 과정"""
        
        thinking_start = time.time()
        
        # 1. 고급 상황 분류
        situation_type = self._advanced_classify_situation(screen_data)
        
        # 2. 고급 RAG 컨텍스트 생성
        rag_context = self.rag_db.get_advanced_context(screen_data, situation_type, limit=10)
        
        # 3. 동적 학습 파라미터 적용
        exploration_bonus = ""
        if self.learning_params['exploration_rate'] > 0.2:
            exploration_bonus = "탐험적 행동도 고려하세요."
        
        battle_focus = ""
        if self.learning_params['battle_focus_mode']:
            battle_focus = f"전투 발견이 최우선! 현재 {self.session_battle_count}회 달성."
        
        # 4. 고도화된 프롬프트
        prompt = f"""영웅전설4 마스터 AI. 에피소드 {self.current_episode}, 스텝 {self.step_count}, 전투 {self.session_battle_count}회.

화면분석: {screen_data.get('description', '')[:150]}
상황분류: {situation_type}

{rag_context}

{battle_focus}
{exploration_bonus}

성능목표: 속도 최적화, 무제한 전투, 완전 학습

행동옵션: left/right/up/down/space/enter/z/x/a/s/q/w/e/1/2/3/esc/tab

RAG 데이터 + 실시간 학습으로 최적 결정:
{{
    "thoughts": "RAG분석+실시간판단",
    "action": "행동",
    "reasoning": "상세이유",
    "confidence": 0.85,
    "situation_type": "{situation_type}",
    "rag_influence": 0.7,
    "exploration": false,
    "battle_potential": 0.5
}}"""

        if self.enable_llm:
            try:
                # 고속 LLM 요청
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.15,  # 더 결정적
                            "max_tokens": 120,    # 더 빠르게
                            "num_ctx": 1536,      # 더 효율적
                            "top_k": 10,
                            "top_p": 0.9
                        }
                    }
                    
                    async with session.post(f"{self.ollama_url}/api/generate", 
                                          json=payload, timeout=aiohttp.ClientTimeout(total=3.0)) as response:
                        if response.status == 200:
                            result = await response.json()
                            ai_response = result.get('response', '')
                            
                            # 고속 JSON 파싱
                            try:
                                json_start = ai_response.find('{')
                                json_end = ai_response.rfind('}') + 1
                                
                                if json_start >= 0 and json_end > json_start:
                                    json_str = ai_response[json_start:json_end]
                                    ai_decision = json.loads(json_str)
                                    
                                    # 메타데이터 추가
                                    ai_decision['situation_type'] = situation_type
                                    ai_decision['decision_time'] = time.time() - thinking_start
                                    ai_decision['rag_context_length'] = len(rag_context)
                                    
                                    return ai_decision
                            except json.JSONDecodeError:
                                pass
                                
            except Exception as e:
                print(f"❌ AI 연결 오류: {e}")
        
        # 고급 폴백 (RAG 기반)
        fallback_decision = self._generate_rag_fallback(situation_type, screen_data)
        fallback_decision['decision_time'] = time.time() - thinking_start
        fallback_decision['is_fallback'] = True
        
        return fallback_decision
    
    def _advanced_classify_situation(self, screen_data: Dict) -> str:
        """고급 상황 분류"""
        brightness = screen_data.get('brightness', 0)
        red_ratio = screen_data.get('red_ratio', 0)
        blue_ratio = screen_data.get('blue_ratio', 0)
        green_ratio = screen_data.get('green_ratio', 0)
        
        # 더 정교한 분류
        if blue_ratio > 0.15 and brightness > 50:
            return 'menu_interface'
        elif red_ratio > 0.08 or (red_ratio > 0.03 and brightness < 80):
            return 'battle_scene'
        elif green_ratio > 0.1 and brightness > 70:
            return 'field_exploration'
        elif brightness < 20:
            return 'dark_dungeon'
        elif brightness > 120:
            return 'bright_outdoor'
        elif blue_ratio > 0.05:
            return 'ui_interaction'
        elif 40 < brightness < 80:
            return 'indoor_area'
        else:
            return 'general_exploration'
    
    def _generate_rag_fallback(self, situation_type: str, screen_data: Dict) -> Dict:
        """RAG 기반 폴백 결정"""
        
        # 상황별 기본 행동
        situation_actions = {
            'battle_scene': ['z', 'x', 'a', 's', 'space'],
            'menu_interface': ['enter', 'space', 'z'],
            'field_exploration': ['right', 'left', 'up', 'down'],
            'dark_dungeon': ['right', 'left', 'up'],
            'bright_outdoor': ['right', 'left', 'down', 'up'],
            'ui_interaction': ['enter', 'z', 'esc'],
            'indoor_area': ['right', 'left', 'up', 'down'],
            'general_exploration': ['right', 'left']
        }
        
        actions = situation_actions.get(situation_type, ['right', 'left'])
        
        # 최근 행동 고려하여 다양성 추가
        recent_actions = list(self.action_history)[-5:]
        if recent_actions and len(set(recent_actions)) < 2:
            # 반복적이면 다른 행동 시도
            alternative_actions = [a for a in actions if a not in recent_actions]
            if alternative_actions:
                actions = alternative_actions
        
        # 전투 우선 모드
        if self.learning_params['battle_focus_mode'] and 'battle' not in situation_type:
            # 전투를 찾기 위한 적극적 탐험
            if self.step_count % 20 < 15:
                actions = ['right', 'down', 'left', 'up']
            else:
                actions = ['z', 'x', 'space', 'enter']
        
        selected_action = actions[self.step_count % len(actions)]
        
        return {
            'thoughts': f"RAG 기반 {situation_type} 대응",
            'action': selected_action,
            'reasoning': f'{situation_type}에 최적화된 행동 선택',
            'confidence': 0.6,
            'situation_type': situation_type,
            'rag_influence': 0.8,
            'exploration': True,
            'battle_potential': 0.7 if 'battle' in situation_type else 0.3
        }
    
    async def run_master_ai_session(self):
        """마스터 AI 세션 실행"""
        print("\n🚀 마스터 RAG AI 세션 시작!")
        print("⚡ 최고 성능 모드 활성화")
        print("🔒 완전 격리 + 무제한 실행")
        print("💾 고급 경험 축적 시스템\n")
        
        # 컨트롤러 시작
        self.controller.start_super_isolated_control()
        
        try:
            while True:
                loop_start = time.time()
                self.step_count += 1
                
                # 에피소드 관리
                if self.step_count % 100 == 0:
                    self._start_new_episode()
                
                # 고속 화면 분석
                screen_data = self._fast_screen_analysis()
                
                # 마스터 AI 추론
                ai_decision = await self.master_rag_thinking(screen_data)
                
                # 가시성 높은 로그 (주기적으로)
                if self.step_count % max(1, CONFIG['log_every_steps']) == 0:
                    try:
                        situation = ai_decision.get('situation_type', 'unknown')
                        action = ai_decision.get('action', 'none')
                        conf = ai_decision.get('confidence', 0.0)
                        print(f"🧭 S{self.step_count} | 상황 {situation} -> 행동 {action} (신뢰도 {conf:.2f})")
                    except Exception:
                        pass
                
                # 고속 행동 실행
                success = self.controller.send_rapid_action(ai_decision['action'])
                
                if success:
                    result = self.controller.get_result_fast(timeout=0.03)
                    if result and result.get('success'):
                        
                        # 행동 기록
                        self.action_history.append(ai_decision['action'])
                        
                        # 결과 평가 및 학습
                        evaluation = self._evaluate_advanced_result(screen_data, ai_decision)
                        
                        # 고급 경험 저장 (비동기)
                        self.rag_db.add_advanced_experience(
                            screen_data, ai_decision, evaluation, 
                            self.current_episode, self.session_id
                        )
                        
                        # 통계 업데이트
                        self._update_advanced_stats(ai_decision, evaluation)
                        
                        # 동적 파라미터 조정
                        self._adjust_learning_parameters(evaluation)
                        
                        # 진행 상황 (고속 모드)
                        if self.step_count % 25 == 0:
                            self._print_fast_progress()
                
                # 고속 루프 (더 빠른 실행)
                loop_time = time.time() - loop_start
                if loop_time < 0.05:
                    await asyncio.sleep(0.05 - loop_time)
                
        except KeyboardInterrupt:
            print("\n⏹️ 사용자 중단")
        finally:
            self.controller.stop_super_isolated_control()
            self._print_final_stats()
    
    def _start_new_episode(self):
        """새 에피소드 시작"""
        self.current_episode += 1
        
        # 이전 에피소드 성능 평가
        if self.episode_actions:
            episode_performance = {
                'episode': self.current_episode - 1,
                'duration': time.time() - self.episode_start_time,
                'actions': len(self.episode_actions),
                'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'battle_count': sum(1 for r in self.episode_rewards if r > 0.5)
            }
            
            self.recent_performance.append(episode_performance)
        
        # 새 에피소드 초기화
        self.episode_start_time = time.time()
        self.episode_actions = []
        self.episode_rewards = []
        
        print(f"\n🔄 에피소드 {self.current_episode} 시작")
    
    def _fast_screen_analysis(self) -> Dict:
        """고속 화면 분석"""
        try:
            hwnd = self.controller._find_dosbox()
            if not hwnd:
                return {'error': 'No window'}
            
            rect = win32gui.GetWindowRect(hwnd)
            screenshot = ImageGrab.grab(rect)
            
            # 축소하여 고속 처리
            small_image = screenshot.resize((160, 120))
            image = np.array(small_image)
            
            # 고속 색상 분석
            brightness = np.mean(image)
            
            # HSV 변환 (축소된 이미지로)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 색상 마스크 (더 빠른 방식)
            h, s, v = cv2.split(hsv)
            
            red_pixels = np.sum((h < 10) | (h > 170)) + np.sum((h >= 0) & (h <= 10))
            blue_pixels = np.sum((h >= 100) & (h <= 130))
            green_pixels = np.sum((h >= 40) & (h <= 80))
            
            total_pixels = image.shape[0] * image.shape[1]
            
            red_ratio = red_pixels / total_pixels
            blue_ratio = blue_pixels / total_pixels
            green_ratio = green_pixels / total_pixels
            
            return {
                'brightness': brightness,
                'red_ratio': red_ratio,
                'blue_ratio': blue_ratio,
                'green_ratio': green_ratio,
                'description': f"B{brightness:.0f} R{red_ratio:.3f} G{green_ratio:.3f} B{blue_ratio:.3f}",
                'analysis_speed': 'fast'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _evaluate_advanced_result(self, screen_data: Dict, ai_decision: Dict) -> Dict:
        """고급 결과 평가"""
        
        # 기본 보상
        base_reward = 0.1
        
        # 신뢰도 보너스
        confidence = ai_decision.get('confidence', 0.5)
        confidence_bonus = confidence * 0.2
        
        # 상황 적합성 보너스
        situation_type = ai_decision.get('situation_type', 'unknown')
        situation_bonus = 0.1 if situation_type != 'unknown' else 0
        
        # 전투 탐지 (고급)
        battle_detected = 0
        battle_bonus = 0
        
        red_ratio = screen_data.get('red_ratio', 0)
        blue_ratio = screen_data.get('blue_ratio', 0)
        brightness = screen_data.get('brightness', 0)
        
        # 더 정교한 전투 탐지
        if (red_ratio > 0.05 or 
            (blue_ratio > 0.08 and brightness < 100) or
            'battle' in situation_type):
            battle_detected = 1
            self.session_battle_count += 1
            battle_bonus = 1.5  # 높은 전투 보상
            
            print(f"⚔️ 전투 발견! (총 {self.session_battle_count}회)")
        
        # 탐험 보너스
        exploration_bonus = 0
        if ai_decision.get('exploration', False):
            exploration_bonus = 0.15
        
        # 최종 점수 계산
        total_reward = (base_reward + confidence_bonus + situation_bonus + 
                       battle_bonus + exploration_bonus)
        
        success_score = min(total_reward / 2.0, 1.0)  # 정규화
        
        return {
            'success_score': success_score,
            'battle_detected': battle_detected,
            'battle_count': self.session_battle_count,
            'reward': total_reward,
            'confidence_bonus': confidence_bonus,
            'battle_bonus': battle_bonus,
            'exploration_bonus': exploration_bonus
        }
    
    def _update_advanced_stats(self, ai_decision: Dict, evaluation: Dict):
        """고급 통계 업데이트"""
        
        self.advanced_stats['total_experiences'] += 1
        
        if evaluation['success_score'] > 0.5:
            self.advanced_stats['successful_actions'] += 1
        
        if evaluation['battle_detected']:
            self.advanced_stats['battle_discoveries'] += 1
        
        self.advanced_stats['model_decisions'] += 1
        
        # 평균 신뢰도 업데이트
        current_avg = self.advanced_stats['avg_confidence']
        total_decisions = self.advanced_stats['model_decisions']
        new_confidence = ai_decision.get('confidence', 0.5)
        self.advanced_stats['avg_confidence'] = (current_avg * (total_decisions - 1) + new_confidence) / total_decisions
        
        # 성공률 업데이트
        self.advanced_stats['success_rate'] = self.advanced_stats['successful_actions'] / self.advanced_stats['total_experiences']
        
        # 전투율 업데이트
        self.advanced_stats['battle_rate'] = self.advanced_stats['battle_discoveries'] / self.advanced_stats['total_experiences']
        
        # 에피소드 기록
        self.episode_actions.append(ai_decision['action'])
        self.episode_rewards.append(evaluation['reward'])
    
    def _adjust_learning_parameters(self, evaluation: Dict):
        """동적 학습 파라미터 조정"""
        
        # 성공률에 따른 탐험율 조정
        if self.advanced_stats['success_rate'] > 0.7:
            self.learning_params['exploration_rate'] = max(0.1, self.learning_params['exploration_rate'] - 0.01)
        else:
            self.learning_params['exploration_rate'] = min(0.5, self.learning_params['exploration_rate'] + 0.01)
        
        # 전투율에 따른 포커스 모드 조정
        if self.advanced_stats['battle_rate'] < 0.1:
            self.learning_params['battle_focus_mode'] = True
        elif self.advanced_stats['battle_rate'] > 0.3:
            self.learning_params['battle_focus_mode'] = False
        
        # 신뢰도에 따른 임계값 조정
        if self.advanced_stats['avg_confidence'] > 0.8:
            self.learning_params['confidence_threshold'] = 0.75
        else:
            self.learning_params['confidence_threshold'] = 0.65
    
    def _print_fast_progress(self):
        """고속 진행 상황 출력"""
        elapsed = time.time() - self.session_start
        speed = self.step_count / elapsed
        
        stats = self.advanced_stats
        perf = self.controller.get_performance_stats()
        
        print(f"🚀 S{self.step_count} | 전투{self.session_battle_count} | {speed:.1f}sps | "
              f"성공률{stats['success_rate']:.2f} | 신뢰도{stats['avg_confidence']:.2f} | "
              f"제어성공률{perf['actions_successful']/max(1, perf['actions_sent']):.2f}")
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        elapsed = time.time() - self.session_start
        
        print(f"\n📊 마스터 AI 세션 완료")
        print(f"⏱️ 총 시간: {elapsed:.0f}초")
        print(f"🎯 총 스텝: {self.step_count}")
        print(f"⚔️ 전투 발견: {self.session_battle_count}회")
        print(f"📈 성공률: {self.advanced_stats['success_rate']:.2%}")
        print(f"🧠 평균 신뢰도: {self.advanced_stats['avg_confidence']:.2f}")
        print(f"⚡ 처리 속도: {self.step_count / elapsed:.1f} SPS")
        
        controller_stats = self.controller.get_performance_stats()
        print(f"🎮 제어 성공률: {controller_stats['actions_successful'] / max(1, controller_stats['actions_sent']):.2%}")
        print(f"🔒 포커스 보존: {controller_stats['focus_preserved_count']}회")


# 실행
if __name__ == "__main__":
    async def main():
        print("🔥 영웅전설4 마스터 RAG AI")
        print("=" * 60)
        print("⚡ 최고 성능 + 완전 격리 + 무제한 학습")
        print("🎯 목표: 좌우 이동하며 전투 10회 이상 달성")
        print("🚀 속도: 최대한 빠르게")
        print("🔒 독립성: 윈도우 포커스 절대 방해 안함")
        print("💾 학습: 모든 경험을 RAG에 축적")
        print("=" * 60)
        
        ai = MasterRAGAI()
        await ai.run_master_ai_session()
    
    asyncio.run(main())