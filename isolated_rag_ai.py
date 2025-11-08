#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영웅전설4 독립 실행 AI + RAG 시스템
- 별도 프로세스로 독립 실행
- 윈도우 컨트롤 격리
- RAG 데이터 축적 및 활용
- 경험 기반 학습
"""

import asyncio
import time
import json
import sqlite3
import numpy as np
import cv2
import aiohttp
import subprocess
import threading
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import win32ui
import win32gui
import win32con
import win32api
import win32process
import multiprocessing
from queue import Queue
import pickle
import os
from src.screen_pipeline import build_capture_pipeline  # 신규 파이프라인 통합
try:
    from src.state_classifier import StateClassifier
except Exception:
    StateClassifier = None
import ctypes

# DPI 인식 설정: 좌표/크기 불일치(고DPI 스케일링) 방지
try:
    # Per Monitor v2 (Windows 10+)
    _DPI_CONTEXT_PER_MONITOR_AWARE_V2 = -4  # HWND-상관 없음, 상수 값
    ctypes.windll.user32.SetProcessDpiAwarenessContext(_DPI_CONTEXT_PER_MONITOR_AWARE_V2)
except Exception:
    try:
        # Windows 8.1 API
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

# 실행 목표/로그/스냅샷 및 타깃 창 설정 (환경변수 주입 가능)
CONFIG = {
    'goal': os.environ.get('HERO4_GOAL', 'move_field_and_battle'),  # move_field_and_battle | battle | explore
    'log_every_steps': int(os.environ.get('HERO4_LOG_STEPS', '5')),
    'snapshot_dir': os.environ.get('HERO4_SNAPSHOT_DIR', 'snapshots'),
    'snapshot_every_steps': int(os.environ.get('HERO4_SNAPSHOT_EVERY', '1')),
    'snapshot_annotate': bool(int(os.environ.get('HERO4_SNAPSHOT_ANNOT', '1'))),
    # 타깃 창 추적 필터
    'win_title_substr': os.environ.get('HERO4_WIN_TITLE', 'dosbox'),  # 예: 'DOSBox', 'ED4'
    'win_class': os.environ.get('HERO4_WIN_CLASS', ''),               # 예: 'SDL_app' 등
    'proc_exe_substr': os.environ.get('HERO4_PROC_EXE', 'dosbox'),    # 예: 'dosbox', 'ed4'
    'strict_target_only': bool(int(os.environ.get('HERO4_STRICT_ONLY', '1'))),
    # 캡처 모드: client(클라이언트 영역) | window(GetWindowRect) | frame(확장 프레임 포함)
    'capture_mode': os.environ.get('HERO4_CAPTURE_MODE', 'window').lower(),
    # OBS 가상카메라로 입력 받기 옵션
    'obs_device_index': int(os.environ.get('HERO4_OBS_DEVICE_INDEX', '0'))
}

class WindowTracker:
    """특정 윈도우만 '정확하게' 추적/고정하여 캡처/입력을 보장하는 트래커"""

    def __init__(self,
                 title_substr: str = CONFIG['win_title_substr'],
                 class_name: str = CONFIG['win_class'],
                 exe_substr: str = CONFIG['proc_exe_substr']):
        self.title_substr = (title_substr or '').lower()
        self.class_name = class_name or ''
        self.exe_substr = (exe_substr or '').lower()
        self.main_hwnd: Optional[int] = None
        self.child_hwnd: Optional[int] = None
        self.locked_pid: Optional[int] = None
        self._last_refind = 0.0
        # DWM 확장 프레임 바운즈 조회용 구조체 준비
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        self._RECT = RECT
        self._DWMWA_EXTENDED_FRAME_BOUNDS = 9
        # WGC(Windows Graphics Capture) 가용 여부 확인
        self._wgc = None
        try:
            import windows_capture  # 경량 래퍼 라이브러리(선택)
            self._wgc = windows_capture
        except Exception:
            try:
                import winrt.windows.graphics.capture as _try_wgc  # 공식 WinRT 경로(구현 난도 높음)
                self._wgc = _try_wgc  # 플래그 용도
            except Exception:
                self._wgc = None

    def _window_matches(self, hwnd: int) -> bool:
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return False
            title = win32gui.GetWindowText(hwnd) or ''
            cls = win32gui.GetClassName(hwnd) or ''
            if self.title_substr and self.title_substr not in title.lower():
                return False
            if self.class_name and self.class_name != cls:
                return False
            # 프로세스 경로 검사
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                handle = win32api.OpenProcess(0x0400 | 0x0010, False, pid)  # QUERY_INFORMATION | VM_READ
                try:
                    exe = win32process.GetModuleFileNameEx(handle, 0) or ''
                except Exception:
                    exe = ''
                finally:
                    win32api.CloseHandle(handle)
                if self.exe_substr and self.exe_substr not in exe.lower():
                    return False
            except Exception:
                # 프로세스 경로 확인이 실패해도, 타이틀/클래스가 맞으면 허용
                pass
            return True
        except Exception:
            return False

    def _pick_largest(self, hwnds: List[int]) -> Optional[int]:
        best = None
        best_area = -1
        for h in hwnds:
            try:
                l, t, r, b = win32gui.GetClientRect(h)
                area = max(0, r - l) * max(0, b - t)
                if area > best_area:
                    best_area = area
                    best = h
            except Exception:
                continue
        return best

    def find_and_lock(self, force: bool = False) -> Optional[int]:
        """조건에 맞는 창을 찾아 고정. 이미 고정돼있으면 그대로 유지."""
        now = time.time()
        if not force and self.main_hwnd and win32gui.IsWindow(self.main_hwnd) and (now - self._last_refind) < 1.0:
            return self.main_hwnd

        candidates: List[int] = []
        def enum_cb(h, _):
            if self._window_matches(h):
                candidates.append(h)
            return True
        win32gui.EnumWindows(enum_cb, None)
        if not candidates:
            self.main_hwnd = None
            self.child_hwnd = None
            self.locked_pid = None
            return None

        self.main_hwnd = self._pick_largest(candidates) or candidates[0]
        try:
            _, pid = win32process.GetWindowThreadProcessId(self.main_hwnd)
        except Exception:
            pid = None
        self.locked_pid = pid
        self.child_hwnd = self._find_best_child(self.main_hwnd)
        self._last_refind = now
        return self.main_hwnd

    def _find_best_child(self, main_hwnd: int) -> int:
        """가장 큰 자식 창(렌더 표면일 가능성)을 선택"""
        best = main_hwnd
        best_area = -1
        try:
            def enum_child(h, _):
                nonlocal best, best_area
                try:
                    l, t, r, b = win32gui.GetClientRect(h)
                    area = max(0, r - l) * max(0, b - t)
                    if area > best_area:
                        best_area = area
                        best = h
                except Exception:
                    pass
                return True
            win32gui.EnumChildWindows(main_hwnd, enum_child, None)
        except Exception:
            pass
        return best

    def get_handles(self) -> Tuple[Optional[int], Optional[int]]:
        if not self.main_hwnd or not win32gui.IsWindow(self.main_hwnd):
            self.find_and_lock(force=True)
        # 자식 핸들은 변동 가능 → 매번 최신화
        if self.main_hwnd:
            self.child_hwnd = self._find_best_child(self.main_hwnd)
        return self.main_hwnd, self.child_hwnd

    def client_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """클라이언트 영역을 스크린 좌표 bbox로 반환"""
        main, child = self.get_handles()
        if not child:
            return None
        try:
            l, t, r, b = win32gui.GetClientRect(child)
            x, y = win32gui.ClientToScreen(child, (0, 0))
            return (x, y, x + (r - l), y + (b - t))
        except Exception:
            return None

    def window_bbox(self, use_extended: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """윈도우 전체 bbox. use_extended=True이면 DWM 확장 프레임까지 포함"""
        main, _ = self.get_handles()
        if not main:
            return None
        if use_extended:
            try:
                rect = self._RECT()
                hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
                    ctypes.wintypes.HWND(main),
                    ctypes.wintypes.DWORD(self._DWMWA_EXTENDED_FRAME_BOUNDS),
                    ctypes.byref(rect),
                    ctypes.sizeof(rect)
                )
                if hr == 0:  # S_OK
                    return (rect.left, rect.top, rect.right, rect.bottom)
            except Exception:
                pass
        try:
            l, t, r, b = win32gui.GetWindowRect(main)
            return (l, t, r, b)
        except Exception:
            return None

    def sizes(self) -> Dict[str, Optional[Tuple[int, int]]]:
        """클라이언트/윈도우 크기 반환"""
        main, child = self.get_handles()
        client = None
        window = None
        try:
            if child:
                l, t, r, b = win32gui.GetClientRect(child)
                client = (max(0, r - l), max(0, b - t))
        except Exception:
            pass
        try:
            if main:
                l, t, r, b = win32gui.GetWindowRect(main)
                window = (max(0, r - l), max(0, b - t))
        except Exception:
            pass
        return { 'client': client, 'window': window }

    def grab_image(self, mode: str = None) -> Tuple[Optional[int], Optional[Tuple[int,int,int,int]], Optional[Image.Image]]:
        """모드에 따른 이미지 캡처: client | window | frame"""
        mode = (mode or CONFIG.get('capture_mode', 'window')).lower()
        main, child = self.get_handles()
        if not main:
            return None, None, None

        # OBS와 유사한 캡처: Windows Graphics Capture (가능한 경우)
        if mode == 'wgc':
            img = self._capture_with_wgc(main)
            if img is not None:
                bbox = self.window_bbox(use_extended=True) or self.window_bbox(False)
                return main, bbox, img
            # 실패 시 다른 모드로 폴백

        if mode == 'client' and child:
            # 1) ImageGrab by client bbox
            bbox = self.client_bbox()
            if bbox and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
                try:
                    return main, bbox, ImageGrab.grab(bbox)
                except Exception:
                    pass
            # 2) PrintWindow PW_CLIENTONLY
            try:
                l, t, r, b = win32gui.GetClientRect(child)
                w, h = (r - l), (b - t)
                if w > 0 and h > 0:
                    hwndDC = win32gui.GetWindowDC(child)
                    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                    saveDC = mfcDC.CreateCompatibleDC()
                    saveBitMap = win32ui.CreateBitmap()
                    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
                    saveDC.SelectObject(saveBitMap)
                    ok = win32gui.PrintWindow(child, saveDC.GetSafeHdc(), 1)
                    if ok == 1:
                        bmpinfo = saveBitMap.GetInfo()
                        bmpstr = saveBitMap.GetBitmapBits(True)
                        shot = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)
                        x, y = win32gui.ClientToScreen(child, (0, 0))
                        bbox = (x, y, x + w, y + h)
                        win32gui.DeleteObject(saveBitMap.GetHandle())
                        saveDC.DeleteDC()
                        mfcDC.DeleteDC()
                        win32gui.ReleaseDC(child, hwndDC)
                        return main, bbox, shot
                    win32gui.DeleteObject(saveBitMap.GetHandle())
                    saveDC.DeleteDC()
                    mfcDC.DeleteDC()
                    win32gui.ReleaseDC(child, hwndDC)
            except Exception:
                pass

        # window/frame 모드: 윈도우 전체 사각형 캡처
        bbox = self.window_bbox(use_extended=(mode == 'frame')) or self.window_bbox(False)
        if bbox and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
            try:
                return main, bbox, ImageGrab.grab(bbox)
            except Exception:
                pass
        return None, None, None

    def _capture_with_wgc(self, hwnd: int) -> Optional[Image.Image]:
        """Windows Graphics Capture 기반 단일 프레임 캡처 시도.
        - 우선 windows-capture 패키지가 있으면 그것을 사용
        - 없으면 WinRT가 있는지 플래그만 확인하고 실패 반환
        """
        try:
            if self._wgc is None:
                return None
            # 1) windows-capture 경로
            try:
                import windows_capture as wc
                capturer = wc.WindowCapture(hwnd)
                frame = capturer.get_latest_frame(timeout=100)
                if frame is None:
                    return None
                # frame을 numpy(HxWxBGRA) 라고 가정
                import numpy as _np
                arr = _np.asarray(frame)
                if arr.shape[-1] == 4:
                    from PIL import Image as _Image
                    return _Image.fromarray(arr[:, :, :3].copy())
                return _Image.fromarray(arr.copy())
            except Exception:
                pass
            # 2) WinRT만 감지된 경우: 구현 복잡도 때문에 안내 후 None
            return None
        except Exception:
            return None

class RAGDatabase:
    """RAG (Retrieval-Augmented Generation) 데이터베이스"""
    
    def __init__(self, db_path: str = "hero4_rag.db"):
        """초기화"""
        self.db_path = db_path
        self.experience_cache = deque(maxlen=1000)
        self.write_queue = Queue()  # 비동기 쓰기 큐
        self.db_lock = threading.Lock()  # 데이터베이스 락
        self._init_database()
        self._start_db_writer()
    
    def _start_db_writer(self):
        """데이터베이스 쓰기 스레드 시작"""
        self.db_writer_thread = threading.Thread(target=self._db_writer_worker, daemon=True)
        self.db_writer_thread.start()
        
    def _get_connection(self):
        """안전한 데이터베이스 연결"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # WAL 모드로 동시 접근 개선
        conn.execute("PRAGMA synchronous=NORMAL")  # 성능 개선
        conn.execute("PRAGMA cache_size=10000")  # 캐시 크기 증가
        return conn
    
    def _init_database(self):
        """데이터베이스 초기화"""
        with self._get_connection() as conn:
            # 화면 상황별 행동 패턴
            conn.execute("""
                CREATE TABLE IF NOT EXISTS screen_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    screen_hash TEXT NOT NULL,
                    screen_description TEXT,
                    action_taken TEXT NOT NULL,
                    result_success INTEGER,
                    battle_discovered INTEGER,
                    reward_score REAL,
                    timestamp TEXT,
                    ai_reasoning TEXT
                )
            """)
            
            # AI 추론 패턴 저장
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    situation_type TEXT,
                    ai_thoughts TEXT,
                    action_chosen TEXT,
                    confidence_level REAL,
                    curiosity_level REAL,
                    success_rate REAL,
                    usage_count INTEGER DEFAULT 1,
                    last_used TEXT
                )
            """)
            
            # 성공/실패 패턴 분석
            conn.execute("""
                CREATE TABLE IF NOT EXISTS success_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_description TEXT,
                    success_actions TEXT,
                    failure_actions TEXT,
                    pattern_frequency INTEGER,
                    effectiveness_score REAL
                )
            """)
            
    def store_experience(self, screen_data: Dict, ai_decision: Dict, result: Dict):
        """경험 저장 (비동기 큐 방식)"""
        try:
            # 큐에 데이터 추가 (비동기 처리)
            experience_data = {
                'type': 'experience',
                'screen_hash': self._generate_screen_hash(screen_data),
                'screen_description': screen_data.get('description', '')[:500],
                'action_taken': ai_decision.get('action', ''),
                'result_success': result.get('success', 0),
                'battle_discovered': result.get('battle_found', 0),
                'reward_score': result.get('reward', 0.0),
                'timestamp': datetime.now().isoformat(),
                'ai_reasoning': json.dumps(ai_decision),
                'ai_decision': ai_decision,
                'result': result
            }
            
            self.write_queue.put(experience_data)
            
        except Exception as e:
            print(f"⚠️ 경험 큐잉 실패: {e}")
    
    def _db_writer_worker(self):
        """데이터베이스 쓰기 워커 (별도 스레드)"""
        batch_size = 10
        batch_data = []
        
        while True:
            try:
                # 배치 데이터 수집
                while len(batch_data) < batch_size:
                    try:
                        data = self.write_queue.get(timeout=1.0)
                        batch_data.append(data)
                    except:
                        break  # 타임아웃 시 현재 배치 처리
                
                if batch_data:
                    self._process_batch_data(batch_data)
                    batch_data.clear()
                
            except Exception as e:
                print(f"⚠️ DB 쓰기 워커 오류: {e}")
                time.sleep(1)
    
    def _process_batch_data(self, batch_data: List[Dict]):
        """배치 데이터 처리"""
        with self.db_lock:
            try:
                with self._get_connection() as conn:
                    for data in batch_data:
                        if data['type'] == 'experience':
                            # 경험 데이터 저장
                            conn.execute("""
                                INSERT INTO screen_actions 
                                (screen_hash, screen_description, action_taken, result_success, 
                                 battle_discovered, reward_score, timestamp, ai_reasoning)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                data['screen_hash'],
                                data['screen_description'],
                                data['action_taken'],
                                data['result_success'],
                                data['battle_discovered'],
                                data['reward_score'],
                                data['timestamp'],
                                data['ai_reasoning']
                            ))
                            
                            # 추론 패턴 업데이트 (배치)
                            self._batch_update_reasoning_pattern(
                                conn, data['ai_decision'], data['result']
                            )
                    
                    conn.commit()
                    
            except Exception as e:
                print(f"⚠️ 배치 처리 실패: {e}")
    
    def _generate_screen_hash(self, screen_data: Dict) -> str:
        """화면 데이터를 해시로 변환"""
        # 주요 특성들을 조합해서 해시 생성
        key_features = [
            screen_data.get('brightness', 0),
            screen_data.get('red_ratio', 0),
            screen_data.get('blue_ratio', 0),
            screen_data.get('green_ratio', 0)
        ]
        
        # 값들을 구간으로 나누어 해시 생성
        hash_parts = []
        for feature in key_features:
            if isinstance(feature, (int, float)):
                bucket = int(feature * 10) // 2  # 0.2 단위로 그룹화
                hash_parts.append(str(bucket))
        
        return "_".join(hash_parts)
    
    def _batch_update_reasoning_pattern(self, conn, ai_decision: Dict, result: Dict):
        """추론 패턴 업데이트 (배치 처리용)"""
        situation = ai_decision.get('situation_type', 'general')
        action = ai_decision.get('action', '')
        confidence = ai_decision.get('confidence', 0.5)
        success = result.get('success', 0)
        
        try:
            # 기존 패턴 확인
            cursor = conn.execute("""
                SELECT id, success_rate, usage_count FROM reasoning_patterns 
                WHERE situation_type = ? AND action_chosen = ?
            """, (situation, action))
            
            existing = cursor.fetchone()
            
            if existing:
                # 기존 패턴 업데이트
                old_success_rate = existing[1]
                usage_count = existing[2]
                new_success_rate = (old_success_rate * usage_count + success) / (usage_count + 1)
                
                conn.execute("""
                    UPDATE reasoning_patterns 
                    SET success_rate = ?, usage_count = usage_count + 1, last_used = ?
                    WHERE id = ?
                """, (new_success_rate, datetime.now().isoformat(), existing[0]))
            else:
                # 새 패턴 생성
                conn.execute("""
                    INSERT INTO reasoning_patterns 
                    (situation_type, ai_thoughts, action_chosen, confidence_level, 
                     success_rate, last_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    situation,
                    ai_decision.get('thoughts', '')[:200],
                    action,
                    confidence,
                    success,
                    datetime.now().isoformat()
                ))
        except Exception as e:
            print(f"⚠️ 패턴 업데이트 실패: {e}")
    
    def get_similar_experiences(self, current_screen: Dict, limit: int = 5) -> List[Dict]:
        """유사한 경험 검색"""
        screen_hash = self._generate_screen_hash(current_screen)
        
        with self.db_lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        SELECT screen_description, action_taken, result_success, 
                               battle_discovered, reward_score, ai_reasoning
                        FROM screen_actions 
                        WHERE screen_hash = ? OR screen_hash LIKE ?
                        ORDER BY reward_score DESC, timestamp DESC
                        LIMIT ?
                    """, (screen_hash, f"{screen_hash[:3]}%", limit))
                    
                    experiences = []
                    for row in cursor:
                        experiences.append({
                            'description': row[0],
                            'action': row[1],
                            'success': row[2],
                            'battle': row[3],
                            'reward': row[4],
                            'reasoning': json.loads(row[5]) if row[5] else {}
                        })
                    
                    return experiences
                    
            except Exception as e:
                print(f"⚠️ 경험 검색 실패: {e}")
                return []
    
    def get_best_actions_for_situation(self, situation_type: str) -> List[Dict]:
        """상황별 최적 행동 추천"""
        with self.db_lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        SELECT action_chosen, success_rate, usage_count, confidence_level
                        FROM reasoning_patterns 
                        WHERE situation_type = ? OR situation_type = 'general'
                        ORDER BY success_rate DESC, usage_count DESC
                        LIMIT 3
                    """, (situation_type,))
                    
                    recommendations = []
                    for row in cursor:
                        recommendations.append({
                            'action': row[0],
                            'success_rate': row[1],
                            'usage_count': row[2],
                            'confidence': row[3]
                        })
                    
                    return recommendations
                    
            except Exception as e:
                print(f"⚠️ 추천 검색 실패: {e}")
                return []
    
    def get_rag_context(self, current_screen: Dict, situation: str) -> str:
        """RAG 컨텍스트 생성"""
        # 유사 경험 가져오기
        similar_experiences = self.get_similar_experiences(current_screen, 3)
        
        # 상황별 최적 행동 가져오기
        best_actions = self.get_best_actions_for_situation(situation)
        
        # RAG 컨텍스트 구성
        context = "과거 경험 참조:\n"
        
        if similar_experiences:
            context += "유사한 상황에서의 행동:\n"
            for exp in similar_experiences:
                context += f"- {exp['action']} → {'성공' if exp['success'] else '실패'} (보상: {exp['reward']:.2f})\n"
        
        if best_actions:
            context += "\n검증된 효과적 행동:\n"
            for action in best_actions:
                context += f"- {action['action']}: 성공률 {action['success_rate']:.1%} (사용 {action['usage_count']}회)\n"
        
        return context

class IsolatedGameController:
    """격리된 게임 컨트롤러"""
    
    def __init__(self, tracker: Optional[WindowTracker] = None):
        """초기화"""
        self.game_process = None
        self.control_thread = None
        self.action_queue = Queue()
        self.result_queue = Queue()
        self.running = False
        self.tracker = tracker or WindowTracker()
        
    def start_isolated_control(self):
        """격리된 컨트롤 시작"""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_worker, daemon=True)
        self.control_thread.start()
        print("🔒 격리된 게임 컨트롤러 시작")
    
    def stop_isolated_control(self):
        """격리된 컨트롤 중지"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2)
    
    def _control_worker(self):
        """컨트롤 워커 (별도 스레드)"""
        dosbox_window = self._find_dosbox()
        last_refind = time.time()
        
        while self.running:
            try:
                # 주기적으로 또는 창 핸들 유효성 검사 후 재탐색 (더 짧은 주기)
                if not dosbox_window or not win32gui.IsWindow(dosbox_window) or (time.time() - last_refind > 1.0):
                    # 트래커 기반 재탐색/고정
                    self.tracker.find_and_lock(force=True)
                    dosbox_window, _ = self.tracker.get_handles()
                    if not dosbox_window:
                        dosbox_window = self._find_dosbox()
                    last_refind = time.time()
                
                try:
                    action_data = self.action_queue.get(timeout=0.02)
                    result = self._execute_isolated_action(dosbox_window, action_data)
                    self.result_queue.put(result)
                except Exception:
                    # 큐가 비어있으면 아주 짧게 휴식
                    time.sleep(0.003)
            except Exception as e:
                print(f"⚠️ 컨트롤 워커 오류: {e}")
                time.sleep(0.1)
    
    def _find_dosbox(self):
        """DOSBox 창 찾기"""
        # 우선 트래커 사용
        self.tracker.find_and_lock(force=True)
        h, _ = self.tracker.get_handles()
        if h:
            return h
        # 폴백: 타이틀 단순 검색
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if 'dosbox' in window_text.lower() or 'ed4' in window_text.lower():
                    windows.append(hwnd)
            return True
        windows: List[int] = []
        win32gui.EnumWindows(enum_callback, windows)
        return windows[0] if windows else None
    
    def _get_window_class(self, hwnd: int) -> str:
        try:
            return win32gui.GetClassName(hwnd)
        except Exception:
            return ""
    
    def _get_best_target_window(self, main_hwnd: int) -> int:
        """입력 메시지를 보낼 최적의 대상 핸들 선택 (자식창 우선)"""
        if not main_hwnd:
            return None
        
        best = main_hwnd
        best_area = 0
        
        def enum_child(hwnd, data):
            nonlocal best, best_area
            try:
                rect = win32gui.GetClientRect(hwnd)
                area = max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])
                if area > best_area:
                    best_area = area
                    best = hwnd
            except Exception:
                pass
            return True
        
        try:
            win32gui.EnumChildWindows(main_hwnd, enum_child, None)
        except Exception:
            pass
        
        return best
    
    def _execute_isolated_action(self, window_handle, action_data):
        """완전 격리된 액션 실행 (포커스 변경 없음)"""
        if not window_handle:
            return {'success': False, 'error': 'No window'}
        
        try:
            action = action_data['action']
            
            # 대상 창 결정 (트래커 자식창 우선)
            _, target_hwnd = (self.tracker.get_handles() if self.tracker else (None, None))
            if not target_hwnd:
                target_hwnd = self._get_best_target_window(window_handle)
            
            # 현재 포커스 저장 (단, 변경/복원은 하지 않음 = 완전 격리)
            try:
                original_focus = win32gui.GetForegroundWindow()
            except Exception:
                original_focus = None
            
            # 키 입력 (PostMessage 사용으로 포커스 변경 없이)
            key_map = {
                'left': 0x25, 'right': 0x27, 'up': 0x26, 'down': 0x28,
                'space': 0x20, 'enter': 0x0D, 'z': 0x5A, 'x': 0x58,
                'a': 0x41, 's': 0x53, '1': 0x31, '2': 0x32,
                'esc': 0x1B
            }
            
            if action in key_map:
                vk_code = key_map[action]
                
                # 메시지 상수
                WM_KEYDOWN = 0x0100
                WM_KEYUP = 0x0101
                
                # ScanCode 및 확장키 플래그 구성
                scancode = win32api.MapVirtualKey(vk_code, 0) & 0xFF
                extended_keys = {0x25, 0x27, 0x26, 0x28}  # 방향키는 확장키
                is_extended = 1 if vk_code in extended_keys else 0
                
                # lParam 구성 함수
                def make_lparam(down: bool) -> int:
                    repeat = 1
                    l = (repeat & 0xFFFF) | ((scancode & 0xFF) << 16)
                    if is_extended:
                        l |= (1 << 24)
                    if not down:
                        l |= (1 << 30) | (1 << 31)  # 이전/업 트랜지션
                    return l
                
                # SendMessageTimeout 사용 (대상 스레드에 동기 전달, 포커스 영향 없음)
                try:
                    SMTO_ABORTIFHUNG = 0x0002
                    win32gui.SendMessageTimeout(target_hwnd, WM_KEYDOWN, vk_code, make_lparam(True), SMTO_ABORTIFHUNG, 20)
                    time.sleep(0.01)
                    win32gui.SendMessageTimeout(target_hwnd, WM_KEYUP, vk_code, make_lparam(False), SMTO_ABORTIFHUNG, 20)
                except Exception:
                    # 타임아웃 API 미지원/오류 시 일반 SendMessage로 폴백
                    win32gui.SendMessage(target_hwnd, WM_KEYDOWN, vk_code, make_lparam(True))
                    time.sleep(0.01)
                    win32gui.SendMessage(target_hwnd, WM_KEYUP, vk_code, make_lparam(False))

                # 엄격 모드: 지정 타깃에만 입력 전달
                if not CONFIG.get('strict_target_only', True):
                    try:
                        win32gui.PostMessage(window_handle, WM_KEYDOWN, vk_code, make_lparam(True))
                        win32gui.PostMessage(window_handle, WM_KEYUP, vk_code, make_lparam(False))
                    except Exception:
                        pass

                # 문자 키는 WM_CHAR도 전달 (z/x/a/s/space/enter/숫자)
                char_keys = {
                    'z': ord('z'), 'x': ord('x'), 'a': ord('a'), 's': ord('s'),
                    '1': ord('1'), '2': ord('2'), 'space': 32, 'enter': 13
                }
                if action in char_keys:
                    WM_CHAR = 0x0102
                    ch = char_keys[action]
                    try:
                        win32gui.PostMessage(target_hwnd, WM_CHAR, ch, 1)
                    except Exception:
                        pass
                
                # 디버그 정보
                target_class = self._get_window_class(target_hwnd)
                return {
                    'success': True,
                    'action': action,
                    'focus_preserved': bool(original_focus is None or original_focus != target_hwnd),
                    'target_class': target_class,
                    'target_hwnd': hex(target_hwnd)
                }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Unknown action'}
    
    def send_action(self, action: str) -> bool:
        """액션 전송 (비동기)"""
        if not self.running:
            return False
            
        self.action_queue.put({'action': action})
        return True
    
    def get_result(self, timeout: float = 0.1) -> Optional[Dict]:
        """결과 받기"""
        try:
            return self.result_queue.get(timeout=timeout)
        except:
            return None

class RAGEnhancedAI:
    """RAG 강화 AI 시스템 - 완전 독립 실행"""
    
    def __init__(self):
        """초기화"""
        # 타깃 창 트래커: 정확하게 한 창만 고정 추적
        self.window_tracker = WindowTracker()
        self.rag_db = RAGDatabase()
        self.controller = IsolatedGameController(tracker=self.window_tracker)
        self.model_name = "qwen2.5-coder:7b"
        self.ollama_url = "http://localhost:11434"
        self.obs_cap = None  # OBS 가상카메라 소스
        
        # 상태 추적
        self.step_count = 0
        self.battle_count = 0
        self.action_history = deque(maxlen=50)
        self.session_start = time.time()
        
        # 목표 주입 및 단계 상태
        self.goal = CONFIG['goal']
        self.goal_phase = 'seek_field'  # seek_field -> seek_battle
        self.map_changed = False
        self.prev_signature = None
        
        # 학습 통계
        self.learning_stats = {
            'total_experiences': 0,
            'successful_actions': 0,
            'battle_discoveries': 0,
            'rag_queries': 0,
            'model_decisions': 0
        }
        
        # 경험 추적
        self.current_screen_state = {}
        self.last_ai_decision = {}
        self.learning_episode = 0
        
        # 언스턱 상태
        self.last_action = None
        self.action_repeat = 0
        self.last_sig = None
        self.unstuck_index = 0
        
        # 이동/움직임 추적
        self.prev_small_frame = None
        # 통합 캡처 파이프라인 구성 (window_tracker와 별개로 안정 캡처용)
        try:
            self.locator, self.capture_chain, self.frame_analyzer = build_capture_pipeline()
        except Exception:
            self.locator = None
            self.capture_chain = None
            self.frame_analyzer = None
        # 씬 분류기(선택)
        self.state_clf = None
        try:
            if StateClassifier is not None:
                self.state_clf = StateClassifier.from_env()
        except Exception:
            self.state_clf = None
        self.no_movement_steps = 0
        self.move_dir = 'right'
        self.menu_steps = 0

        # 스냅샷 저장 설정
        try:
            self.snapshot_every = max(1, int(CONFIG['snapshot_every_steps']))
        except Exception:
            self.snapshot_every = 1
        self.snapshot_root = os.path.join(CONFIG['snapshot_dir'], datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.snapshot_root, exist_ok=True)
        
        print("🧠 RAG 강화 AI 시스템 초기화")
        print("💾 경험 데이터베이스 연결") 
        print("🔒 완전 격리된 컨트롤러 준비")
        print("📊 학습 통계 시스템 활성화")
        print(f"🎯 목표: {self.goal} (초기 단계: {self.goal_phase})")
    
    async def rag_enhanced_thinking(self, screen_data: Dict) -> Dict:
        """RAG 강화 사고 과정"""
        
        # 1. 상황 분류
        situation_type = self._classify_situation(screen_data)
        
        # 목표 컨텍스트 텍스트 구성
        goal_text = ""
        if self.goal == 'move_field_and_battle':
            phase_text = '필드이동' if self.goal_phase == 'seek_field' else '전투탐색'
            goal_text = f"목표: 다른 필드로 이동 후 전투 유도 (현재단계: {phase_text})\n"
        elif self.goal == 'battle':
            goal_text = "목표: 전투 화면 진입 및 유지\n"
        elif self.goal == 'explore':
            goal_text = "목표: 맵 탐험 및 UI/경로 학습\n"
        
        # 2. RAG 컨텍스트 생성
        rag_context = self.rag_db.get_rag_context(screen_data, situation_type)
        
        # 3. AI에게 보낼 강화된 프롬프트
        prompt = f"""영웅전설4 AI. 스텝 {self.step_count}, 전투 {self.battle_count}회.

화면: {screen_data.get('description', '')[:200]}

{goal_text}
{rag_context}

행동: left/right/up/down/space/enter/z/x/a/s/1/2

RAG 경험을 참고하여 최적 행동 선택:
{{
    "thoughts": "분석과 RAG 참조",
    "action": "행동",
    "reason": "이유", 
    "confidence": 0.8,
    "situation_type": "{situation_type}"
}}"""

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "max_tokens": 150,
                        "num_ctx": 2048
                    }
                }
                
                async with session.post(f"{self.ollama_url}/api/generate", 
                                      json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get('response', '')
                        
                        # JSON 파싱
                        try:
                            json_start = ai_response.find('{')
                            json_end = ai_response.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = ai_response[json_start:json_end]
                                ai_decision = json.loads(json_str)
                                ai_decision['situation_type'] = situation_type
                                return ai_decision
                        except:
                            pass
        except Exception as e:
            print(f"❌ AI 연결 오류: {e}")
        
        # 실패시 RAG + 목표 바이어스 기반 기본 응답
        best_actions = self.rag_db.get_best_actions_for_situation(situation_type)
        fallback_action = best_actions[0]['action'] if best_actions else 'right'
        
        # 메뉴에서는 탈출/선택 우선
        if situation_type == 'menu_ui':
            # esc -> enter -> z 순환
            cyc = self.step_count % 6
            if cyc < 2:
                fallback_action = 'esc'
            elif cyc < 4:
                fallback_action = 'enter'
            else:
                fallback_action = 'z'

        if self.goal == 'move_field_and_battle':
            if not self.map_changed:
                # 필드 이동 우선: 우→상→좌→하 스윕 패턴
                cycle = self.step_count % 60
                if cycle < 25:
                    fallback_action = 'right'
                elif cycle < 35:
                    fallback_action = 'up'
                elif cycle < 55:
                    fallback_action = 'left'
                else:
                    fallback_action = 'down'
            else:
                # 전투 유도: 좌우 왕복 + 간헐적 공격키
                fallback_action = 'right' if (self.step_count % 10) < 5 else 'left'
                if (self.step_count % 15) == 0:
                    fallback_action = 'x'
        elif self.goal == 'battle':
            if situation_type != 'battle_scene':
                fallback_action = 'right' if (self.step_count % 8) < 4 else 'left'
        elif self.goal == 'explore':
            # 탐험 강화: 사분면 탐색
            cycle = self.step_count % 40
            if cycle < 10:
                fallback_action = 'right'
            elif cycle < 20:
                fallback_action = 'down'
            elif cycle < 30:
                fallback_action = 'left'
            else:
                fallback_action = 'up'
        
        return {
            "thoughts": "RAG 데이터 기반 안전 선택",
            "action": fallback_action,
            "reason": "과거 성공 경험 활용",
            "confidence": 0.6,
            "situation_type": situation_type
        }
    
    def _classify_situation(self, screen_data: Dict) -> str:
        """상황 분류"""
        brightness = screen_data.get('brightness', 0)
        blue_ratio = screen_data.get('blue_ratio', 0)
        red_ratio = screen_data.get('red_ratio', 0)
        edge_h = screen_data.get('edge_h', 0)
        edge_v = screen_data.get('edge_v', 0)
        movement = screen_data.get('movement', 0)
        
        # 메뉴: 파랑이 두드러지고 수평 에지가 강하며 움직임 거의 없음
        if (blue_ratio > 0.10 and edge_h > edge_v * 1.2) or (blue_ratio > 0.15 and movement < 1.0):
            return 'menu_ui'
        elif red_ratio > 0.06:
            return 'battle_scene'
        elif brightness < 30:
            return 'dark_area'
        elif brightness > 100:
            return 'bright_field'
        else:
            return 'exploration'
    
    async def run_rag_ai_session(self):
        """RAG AI 세션 실행"""
        print("\n🚀 RAG 강화 AI 세션 시작!")
        print("🔒 격리 모드로 독립 실행")
        print("💾 경험 데이터 축적 및 활용\n")
        
        # 격리 컨트롤러 시작
        self.controller.start_isolated_control()
        # 타깃 창 고정 및 정보 출력
        self.window_tracker.find_and_lock(force=True)
        main, child = self.window_tracker.get_handles()
        if main:
            title = win32gui.GetWindowText(main)
            cls = win32gui.GetClassName(main)
            sizes = self.window_tracker.sizes()
            cap_mode = CONFIG.get('capture_mode','window')
            print(f"🎯 타깃 창 고정: hwnd={hex(main)} child={hex(child) if child else 'None'} title='{title}' class='{cls}' cap={cap_mode} size(win={sizes.get('window')}, cli={sizes.get('client')})")
        else:
            print("⚠️ 타깃 창을 찾지 못했습니다. 필터 설정(HERO4_WIN_TITLE/HERO4_WIN_CLASS/HERO4_PROC_EXE)을 확인하세요.")
        
        # 초기 워밍업: 메뉴/대화 상자에 갇힌 경우 탈출 시도 (포커스 불요)
        for key in ['esc', 'enter', 'right', 'left']:
            self.controller.send_action(key)
            time.sleep(0.05)
        
        try:
            while True:
                self.step_count += 1
                
                # 화면 캡처 및 분석 (고속)
                screen_data = self._capture_and_analyze()
                
                # 필드 변경 감지 로직 (간단 시그니처 비교)
                sig = self._screen_signature(screen_data) if screen_data else None
                if sig is not None:
                    if self.prev_signature is not None and self._detect_map_change(self.prev_signature, sig):
                        if not self.map_changed:
                            self.map_changed = True
                            if self.goal == 'move_field_and_battle' and self.goal_phase == 'seek_field':
                                self.goal_phase = 'seek_battle'
                            print(f"🗺️ 필드 변경 감지 → 단계 전환: {self.goal_phase}")
                    self.prev_signature = sig
                
                # RAG 강화 AI 사고
                ai_decision = await self.rag_enhanced_thinking(screen_data)
                
                # 추론/행동 로그 (짧고 명확하게)
                if ai_decision:
                    thoughts = ai_decision.get('thoughts') or ai_decision.get('reason') or ai_decision.get('reasoning') or ''
                    if thoughts:
                        thoughts = (thoughts if isinstance(thoughts, str) else str(thoughts))[:80]
                    print(f"🧠 {ai_decision.get('situation_type','?')} -> {ai_decision.get('action','?')} | 신뢰도 {ai_decision.get('confidence',0):.2f} | 생각 {thoughts}")
                    # 메뉴 스텝 카운트 업데이트
                    if ai_decision.get('situation_type') == 'menu_ui':
                        self.menu_steps += 1
                    else:
                        self.menu_steps = 0

                # 스냅샷 저장 (현재 화면 + 분석/결정 메타)
                if (self.step_count % self.snapshot_every) == 0:
                    try:
                        hwnd, rect, shot = self._grab_window_image()
                        if shot is not None:
                            self._save_step_snapshot(self.step_count, shot, screen_data, ai_decision)
                    except Exception as e:
                        # 스냅샷 실패는 무시하고 계속 진행
                        pass
                
                # 계획형(휴리스틱) 의사결정: 메뉴/정지 상태 우선 적용
                planner_action, planner_reason = self._planner_decision(screen_data)
                use_planner = False
                if ai_decision.get('situation_type') == 'menu_ui':
                    use_planner = True
                else:
                    mv = screen_data.get('movement', 0)
                    if mv < 1.0:
                        self.no_movement_steps += 1
                    else:
                        self.no_movement_steps = 0
                    if self.no_movement_steps >= 3:
                        use_planner = True
                
                chosen_action = planner_action if use_planner else ai_decision['action']
                if use_planner:
                    print(f"🧭 플래너 적용: {planner_action} | {planner_reason}")
                
                # 언스턱 판단 및 오버라이드
                override = self._maybe_unstuck(sig, chosen_action) if screen_data else None
                action_to_send = override or chosen_action
                if override:
                    print(f"🧩 언스턱: {ai_decision.get('action')} → {override}")

                # 행동 실행 (격리된 방식)
                success = self.controller.send_action(action_to_send)
                
                if success:
                    # 즉시 다음 루프로 진행: 결과는 짧게만 폴링
                    result = self.controller.get_result(timeout=0.02)
                    if result and result.get('success'):
                        self.action_history.append(action_to_send)
                        
                        # 결과 평가 및 RAG 저장
                        experience_result = self._evaluate_result(screen_data, ai_decision)
                        self.rag_db.store_experience(screen_data, ai_decision, experience_result)
                        
                        # 진행 상황 출력 (간단히)
                        if self.step_count % max(1, CONFIG['log_every_steps']) == 0:
                            elapsed = time.time() - self.session_start
                            tg = f"{result.get('target_class','?')}@{result.get('target_hwnd','?')}"
                            phase = self.goal_phase
                            print(f"🎮 S{self.step_count} | {ai_decision['situation_type']} -> {action_to_send} (신뢰도 {ai_decision.get('confidence',0):.2f}) | 단계 {phase} | 대상 {tg} | 경과 {elapsed:.0f}s")
                    elif result and not result.get('success'):
                        if self.step_count % 10 == 0:
                            print(f"⚠️ 입력 실패: {result.get('error','unknown')} | DOSBox 창 탐지 불가 가능성")
                
                # CPU 폭주 방지용 최소 휴식
                await asyncio.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n⏹️ 사용자 중단")
        finally:
            self.controller.stop_isolated_control()
            print("🔒 격리된 컨트롤러 중지")

    def _maybe_unstuck(self, sig, action: str) -> Optional[str]:
        """같은 액션 반복 + 시그니처 변화 없으면 언스턱 액션 반환"""
        if action == self.last_action and sig == self.last_sig:
            self.action_repeat += 1
        else:
            self.action_repeat = 0
        self.last_action = action
        self.last_sig = sig
        
        if self.action_repeat >= 6:
            seq = ['esc', 'enter', 'z', 'x', 'left', 'right']
            a = seq[self.unstuck_index % len(seq)]
            self.unstuck_index += 1
            # 리셋하여 과도한 언스턱 방지
            self.action_repeat = 0
            return a
        return None
    
    def _capture_and_analyze(self) -> Dict:
        """새 파이프라인 기반 캡처 + 분석 (FrameAnalyzer 활용)"""
        try:
            # 1) 통합 캡처 체인 우선 사용
            if self.capture_chain is not None:
                res = self.capture_chain.get_frame()
                img = res.image
                meta = res.meta or {}
            else:
                # 폴백: 기존 window_tracker 경로
                _, _, img = self.window_tracker.grab_image('window')
                meta = {'fallback': True}
            if img is None:
                return {}
            # 2) FrameAnalyzer로 특성 추출
            features = self.frame_analyzer.analyze(img) if self.frame_analyzer else {}
            # (선택) 씬 분류 예측 추가 → RAG 저장/플래너 힌트로 사용
            if self.state_clf is not None:
                try:
                    pred = self.state_clf.predict(img)
                    features['scene_pred'] = pred.label
                    features['scene_conf'] = pred.confidence
                    for k, v in pred.probs.items():
                        features[f'scene_{k}'] = v
                except Exception:
                    pass
            brightness = features.get('brightness', 0.0)
            red_ratio = features.get('red_ratio', 0.0)
            blue_ratio = features.get('blue_ratio', 0.0)
            edge_h = features.get('edge_h', 0.0)
            edge_v = features.get('edge_v', 0.0)
            movement = features.get('movement', 0.0)
            desc = f"밝기 {brightness:.0f}, 빨강 {red_ratio:.2f}, 파랑 {blue_ratio:.2f} mv{movement:.1f}"
            # 3) 메타 통합
            out = {
                'brightness': brightness,
                'red_ratio': red_ratio,
                'blue_ratio': blue_ratio,
                'edge_h': edge_h,
                'edge_v': edge_v,
                'movement': movement,
                'description': desc,
                'capture_mode': meta.get('mode'),
                'capture_error': meta.get('error'),
            }
            return out
        except Exception as e:
            return {'error': str(e), 'capture_mode': None}
    
    def _evaluate_result(self, screen_data: Dict, ai_decision: Dict) -> Dict:
        """결과 평가"""
        # 간단한 보상 계산
        reward = 0.1  # 기본
        
        if ai_decision.get('confidence', 0) > 0.8:
            reward += 0.2
        
        # 전투 감지 (간단한 방식)
        battle_found = 0
        if (screen_data.get('red_ratio', 0) > 0.05 or 
            screen_data.get('blue_ratio', 0) > 0.1):
            battle_found = 1
            self.battle_count += 1
            reward += 1.0
        
        return {
            'success': 1,
            'battle_found': battle_found,
            'reward': reward
        }

    def _screen_signature(self, sd: Dict) -> Optional[tuple]:
        """화면 시그니처 (간단 요약)"""
        if not sd or 'brightness' not in sd:
            return None
        b = float(sd.get('brightness', 0))
        r = float(sd.get('red_ratio', 0))
        bl = float(sd.get('blue_ratio', 0))
        # 라운딩으로 노이즈 감소
        return (round(b, 0), round(r, 2), round(bl, 2))

    def _detect_map_change(self, prev_sig: tuple, curr_sig: tuple) -> bool:
        """필드(맵) 변경 탐지: 밝기/색 비율 급변으로 추정"""
        if not prev_sig or not curr_sig:
            return False
        db = abs(curr_sig[0] - prev_sig[0])
        dr = abs(curr_sig[1] - prev_sig[1])
        dbl = abs(curr_sig[2] - prev_sig[2])
        # 간단 기준: 밝기 25 이상 또는 색 비율 0.12 이상 급변
        return (db >= 25) or (dr >= 0.12) or (dbl >= 0.12)

    def _planner_decision(self, sd: Dict) -> Tuple[str, str]:
        """상황/목표 기반 휴리스틱 플래너 결정"""
        if not sd:
            return 'right', '기본 우측 이동'
        st = self._classify_situation(sd)
        mv = sd.get('movement', 0)
        reason = ''
        
        if st == 'menu_ui':
            # 누적 메뉴 스텝 기반 빠른 탈출 시퀀스
            seq = ['esc', 'x', 'z', 'enter']
            a = seq[self.menu_steps % len(seq)] if hasattr(self, 'menu_steps') else 'esc'
            reason = f'메뉴 탈출/선택 시퀀스({a})'
            return a, reason
        
        if st == 'battle_scene':
            return ('x' if (self.step_count % 4) < 2 else 'z'), '전투 중 공격/선택'
        
        # 필드 탐색/이동
        # 움직임 없으면 방향 전환
        if mv < 1.0:
            self.no_movement_steps += 1
        else:
            self.no_movement_steps = 0
        
        if self.no_movement_steps >= 2:
            # 간단 방향 전환 시퀀스
            self.move_dir = 'left' if self.move_dir == 'right' else 'right'
            reason = '정체 상태 해소를 위한 방향 전환'
            self.no_movement_steps = 0
        
        # 목표 단계 반영
        if self.goal == 'move_field_and_battle' and self.goal_phase == 'seek_field':
            # 우측 위주 + 가끔 위/아래로 스윕
            if (self.step_count % 20) in (15, 16):
                return 'up', '필드 이동 경로 탐색(위)'
            if (self.step_count % 20) in (17, 18):
                return 'down', '필드 이동 경로 탐색(아래)'
            return self.move_dir, f'필드 이동({self.move_dir})'
        
        # 전투 유도 단계
        if self.goal_phase == 'seek_battle':
            if (self.step_count % 10) < 5:
                return 'right', '전투 유도 좌우 스윙(우)'
            else:
                return 'left', '전투 유도 좌우 스윙(좌)'
        
        # 일반 탐색
        return self.move_dir, f'일반 탐색({self.move_dir})'

    def _grab_window_image(self) -> Tuple[Optional[int], Optional[Tuple[int,int,int,int]], Optional[Image.Image]]:
        """WindowTracker를 사용하여 설정된 모드로 정확히 캡처"""
        mode = CONFIG.get('capture_mode', 'window')
        if mode == 'obs':
            img = self._grab_obs_frame()
            return None, None, img
        return self.window_tracker.grab_image(mode)

    def _grab_obs_frame(self) -> Optional[Image.Image]:
        """OBS 가상 카메라에서 한 프레임 가져오기(BGR→RGB→PIL)."""
        try:
            if self.obs_cap is None:
                # DirectShow 장치 인덱스 사용
                self.obs_cap = cv2.VideoCapture(CONFIG.get('obs_device_index', 0), cv2.CAP_DSHOW)
                # 낮은 지연을 위해 버퍼 줄이기
                self.obs_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = self.obs_cap.read()
            if not ok or frame is None:
                return None
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        except Exception:
            return None

    def _save_step_snapshot(self, step: int, shot: Image.Image, sd: Dict, decision: Dict):
        """스냅샷 저장: 원본 + 주석이미지 + 메타 JSON"""
        try:
            base = os.path.join(self.snapshot_root, f"step_{step:06d}")
            raw_path = base + "_raw.png"
            ann_path = base + "_annot.png"
            json_path = base + ".json"
            # 원본 저장
            shot.save(raw_path)
            if CONFIG.get('snapshot_annotate', True):
                # 주석 이미지 생성
                ann = shot.copy()
                draw = ImageDraw.Draw(ann)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                lines = []
                lines.append(f"step {step}")
                # 크기 정보
                try:
                    sizes = self.window_tracker.sizes()
                    lines.append(f"size win={sizes.get('window')} cli={sizes.get('client')}")
                except Exception:
                    pass
                # 화면 분석 요약
                if sd:
                    lines.append(sd.get('description', ''))
                    lines.append(f"mv {sd.get('movement',0):.1f} edgeH {sd.get('edge_h',0):.1f} edgeV {sd.get('edge_v',0):.1f}")
                # 결정 요약
                if decision:
                    lines.append(f"situation {decision.get('situation_type','?')} action {decision.get('action','?')} conf {decision.get('confidence',0):.2f}")
                    thoughts = decision.get('thoughts') or decision.get('reason') or decision.get('reasoning') or ''
                    if thoughts:
                        if not isinstance(thoughts, str):
                            thoughts = str(thoughts)
                        if len(thoughts) > 120:
                            thoughts = thoughts[:117] + '...'
                        lines.append(f"reason {thoughts}")
                # 텍스트 렌더링
                x, y = 8, 8
                for ln in lines:
                    # 윤곽선 효과로 가독성↑
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        draw.text((x+dx,y+dy), ln, fill=(0,0,0), font=font)
                    draw.text((x,y), ln, fill=(255,255,255), font=font)
                    y += 14
                ann.save(ann_path)
            # 메타 저장
            meta = {
                'step': step,
                'screen': {
                    'description': sd.get('description') if sd else None,
                    'brightness': sd.get('brightness') if sd else None,
                    'red_ratio': sd.get('red_ratio') if sd else None,
                    'blue_ratio': sd.get('blue_ratio') if sd else None,
                    'edge_h': sd.get('edge_h') if sd else None,
                    'edge_v': sd.get('edge_v') if sd else None,
                    'movement': sd.get('movement') if sd else None,
                    'sizes': self.window_tracker.sizes() if hasattr(self, 'window_tracker') else None,
                    'capture_mode': CONFIG.get('capture_mode', 'window'),
                },
                'decision': {
                    'situation_type': decision.get('situation_type') if decision else None,
                    'action': decision.get('action') if decision else None,
                    'confidence': decision.get('confidence') if decision else None,
                    'thoughts': (decision.get('thoughts') or decision.get('reason') or decision.get('reasoning')) if decision else None
                },
                'goal': self.goal,
                'phase': self.goal_phase
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

# 실행
if __name__ == "__main__":
    async def main():
        ai = RAGEnhancedAI()
        await ai.run_rag_ai_session()
    
    print("🔒 독립 실행 RAG AI 시스템")
    print("=" * 50)
    print("💾 경험 축적 + 윈도우 격리 + 무제한 실행")
    
    asyncio.run(main())