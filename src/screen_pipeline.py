# -*- coding: utf-8 -*-
"""
게임 화면 인식 1단계: 안전/안정 캡처 + 전처리 + 특성추출 파이프라인
- 다른 윈도우에 영향 없음 (입력/포커스 변경 X)
- 가려져도 가능한 백엔드 우선(WGC/OBS) + 폴백(PrintWindow/GDI)
- 모듈은 캡처만 담당: 키보드/마우스 입력 없음

환경 변수(대부분 선택):
- HERO4_WIN_TITLE / HERO4_WIN_CLASS / HERO4_PROC_EXE : 타깃 창 필터
- HERO4_CAPTURE_MODE : wgc | client | window | frame | obs
- HERO4_OBS_DEVICE_INDEX : OBS 가상카메라 인덱스 (기본 0)
"""
from __future__ import annotations
import os
import time
import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image, ImageGrab

# 선택 의존성: pywin32가 없으면 일부 백엔드 비활성화
try:
    import win32gui, win32api, win32ui, win32process
except Exception:  # pragma: no cover
    win32gui = None
    win32api = None
    win32ui = None
    win32process = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# 선택 의존성: mss(멀티 모니터/음수 좌표 안전 캡처)
try:
    import mss
except Exception:
    mss = None

CONFIG = {
    'win_title': os.environ.get('HERO4_WIN_TITLE', 'dosbox'),
    'win_class': os.environ.get('HERO4_WIN_CLASS', ''),
    'proc_exe': os.environ.get('HERO4_PROC_EXE', 'dosbox'),
    'capture_mode': os.environ.get('HERO4_CAPTURE_MODE', 'window').lower(),  # + wgc_native
    'obs_device_index': int(os.environ.get('HERO4_OBS_DEVICE_INDEX', '0')),
    'black_skip': bool(int(os.environ.get('HERO4_BLACK_SKIP', '1'))),
    'black_threshold': float(os.environ.get('HERO4_BLACK_THRESHOLD', '1.0')),
}

# DPI 인식(좌표 불일치 방지)
try:
    _DPI_CTX_PM_V2 = -4
    ctypes.windll.user32.SetProcessDpiAwarenessContext(_DPI_CTX_PM_V2)
except Exception:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

@dataclass
class CaptureResult:
    image: Optional[Image.Image]
    bbox: Optional[Tuple[int, int, int, int]]
    meta: Dict

class GameWindowLocator:
    """타깃 창 고정 추적(포커스/입력 영향 없음)"""
    def __init__(self, title_substr: str, class_name: str, exe_substr: str):
        self.title_substr = (title_substr or '').lower()
        self.class_name = class_name or ''
        self.exe_substr = (exe_substr or '').lower()
        self.main_hwnd = None
        self.child_hwnd = None
        self._last_refind = 0.0

    def _window_matches(self, hwnd: int) -> bool:
        if not win32gui:
            return False
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return False
            title = (win32gui.GetWindowText(hwnd) or '').lower()
            cls = win32gui.GetClassName(hwnd) or ''
            if self.title_substr and self.title_substr not in title:
                return False
            if self.class_name and self.class_name != cls:
                return False
            # 프로세스 exe 필터
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                handle = win32api.OpenProcess(0x0400 | 0x0010, False, pid)
                try:
                    exe = (win32process.GetModuleFileNameEx(handle, 0) or '').lower()
                finally:
                    win32api.CloseHandle(handle)
                if self.exe_substr and self.exe_substr not in exe:
                    return False
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _pick_largest_child(self, main: int) -> int:
        best = main
        best_area = -1
        try:
            def enum_child(h, _):
                nonlocal best, best_area
                try:
                    l, t, r, b = win32gui.GetClientRect(h)
                    area = max(0, r-l) * max(0, b-t)
                    if area > best_area:
                        best_area = area
                        best = h
                except Exception:
                    pass
                return True
            win32gui.EnumChildWindows(main, enum_child, None)
        except Exception:
            pass
        return best

    def get_handles(self):
        now = time.time()
        if not self.main_hwnd or not win32gui or not win32gui.IsWindow(self.main_hwnd) or (now - self._last_refind) > 1.0:
            self._refind()
        if self.main_hwnd and win32gui:
            self.child_hwnd = self._pick_largest_child(self.main_hwnd)
        return self.main_hwnd, self.child_hwnd

    def _refind(self):
        if not win32gui:
            return
        candidates = []
        def enum_cb(h, _):
            if self._window_matches(h):
                candidates.append(h)
            return True
        win32gui.EnumWindows(enum_cb, None)
        self.main_hwnd = candidates[0] if candidates else None
        self._last_refind = time.time()

    def client_bbox(self) -> Optional[Tuple[int,int,int,int]]:
        if not win32gui:
            return None
        main, child = self.get_handles()
        if not child:
            return None
        l, t, r, b = win32gui.GetClientRect(child)
        x, y = win32gui.ClientToScreen(child, (0, 0))
        w, h = (r - l), (b - t)
        if w <= 0 or h <= 0:
            return None
        return (x, y, x + w, y + h)

    def window_bbox(self) -> Optional[Tuple[int,int,int,int]]:
        if not win32gui:
            return None
        main, _ = self.get_handles()
        if not main:
            return None
        l, t, r, b = win32gui.GetWindowRect(main)
        w, h = (r - l), (b - t)
        if w <= 0 or h <= 0:
            return None
        return (l, t, r, b)

class CaptureBackend:
    def capture(self) -> CaptureResult:
        raise NotImplementedError

class GDIClientBackend(CaptureBackend):
    def __init__(self, locator: GameWindowLocator):
        self.locator = locator
    def capture(self) -> CaptureResult:
        bbox = self.locator.client_bbox()
        if not bbox:
            return CaptureResult(None, None, {'mode': 'client'})
        x1, y1, x2, y2 = bbox
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return CaptureResult(None, None, {'mode': 'client'})
        try:
            img = ImageGrab.grab(bbox)
            return CaptureResult(img, bbox, {'mode': 'client'})
        except Exception:
            return CaptureResult(None, None, {'mode': 'client', 'error': 'grab-failed'})

class PrintWindowClientBackend(CaptureBackend):
    def __init__(self, locator: GameWindowLocator):
        self.locator = locator
    def capture(self) -> CaptureResult:
        if not win32gui or not win32ui:
            return CaptureResult(None, None, {'mode': 'printwindow'})
        main, child = self.locator.get_handles()
        if not child:
            return CaptureResult(None, None, {'mode': 'printwindow'})
        try:
            l, t, r, b = win32gui.GetClientRect(child)
            w, h = (r-l), (b-t)
            if w <= 0 or h <= 0:
                return CaptureResult(None, None, {'mode': 'printwindow'})
            hwndDC = win32gui.GetWindowDC(child)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            saveBmp = win32ui.CreateBitmap()
            saveBmp.CreateCompatibleBitmap(mfcDC, w, h)
            saveDC.SelectObject(saveBmp)
            ok = win32gui.PrintWindow(child, saveDC.GetSafeHdc(), 1)
            if ok == 1:
                info = saveBmp.GetInfo()
                bits = saveBmp.GetBitmapBits(True)
                img = Image.frombuffer('RGB', (info['bmWidth'], info['bmHeight']), bits, 'raw', 'BGRX', 0, 1)
                x, y = win32gui.ClientToScreen(child, (0, 0))
                bbox = (x, y, x+w, y+h)
                win32gui.DeleteObject(saveBmp.GetHandle())
                saveDC.DeleteDC(); mfcDC.DeleteDC(); win32gui.ReleaseDC(child, hwndDC)
                return CaptureResult(img, bbox, {'mode': 'printwindow'})
            win32gui.DeleteObject(saveBmp.GetHandle())
            saveDC.DeleteDC(); mfcDC.DeleteDC(); win32gui.ReleaseDC(child, hwndDC)
        except Exception:
            pass
        return CaptureResult(None, None, {'mode': 'printwindow'})

class MSSClientBackend(CaptureBackend):
    def __init__(self, locator: GameWindowLocator):
        self.locator = locator
        try:
            self.sct = mss.mss() if mss else None
        except Exception:
            self.sct = None
    def capture(self) -> CaptureResult:
        if self.sct is None:
            return CaptureResult(None, None, {'mode': 'mss_client'})
        bbox = self.locator.client_bbox()
        if not bbox:
            return CaptureResult(None, None, {'mode': 'mss_client'})
        x1, y1, x2, y2 = bbox
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w <= 0 or h <= 0:
            return CaptureResult(None, None, {'mode': 'mss_client'})
        try:
            shot = self.sct.grab({'left': x1, 'top': y1, 'width': w, 'height': h})
            img = Image.frombytes('RGB', shot.size, shot.bgra, 'raw', 'BGRX')
            return CaptureResult(img, bbox, {'mode': 'mss_client'})
        except Exception:
            return CaptureResult(None, None, {'mode': 'mss_client'})

class WindowRectBackend(CaptureBackend):
    def __init__(self, locator: GameWindowLocator):
        self.locator = locator
    def capture(self) -> CaptureResult:
        bbox = self.locator.window_bbox()
        if not bbox:
            return CaptureResult(None, None, {'mode': 'window'})
        x1, y1, x2, y2 = bbox
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return CaptureResult(None, None, {'mode': 'window'})
        try:
            img = ImageGrab.grab(bbox)
            return CaptureResult(img, bbox, {'mode': 'window'})
        except Exception:
            return CaptureResult(None, None, {'mode': 'window', 'error': 'grab-failed'})
class FullscreenBackend(CaptureBackend):
    """윈도우가 없을 때 전체 화면 캡처 폴백."""
    def capture(self) -> CaptureResult:
        try:
            img = ImageGrab.grab()
            # bbox는 알 수 없으므로 None
            return CaptureResult(img, None, {'mode': 'fullscreen'})
        except Exception:
            return CaptureResult(None, None, {'mode': 'fullscreen', 'error': 'grab-failed'})

class MSSWindowBackend(CaptureBackend):
    def __init__(self, locator: GameWindowLocator):
        self.locator = locator
        try:
            self.sct = mss.mss() if mss else None
        except Exception:
            self.sct = None
    def capture(self) -> CaptureResult:
        if self.sct is None:
            return CaptureResult(None, None, {'mode': 'mss_window'})
        bbox = self.locator.window_bbox()
        if not bbox:
            return CaptureResult(None, None, {'mode': 'mss_window'})
        x1, y1, x2, y2 = bbox
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w <= 0 or h <= 0:
            return CaptureResult(None, None, {'mode': 'mss_window'})
        try:
            shot = self.sct.grab({'left': x1, 'top': y1, 'width': w, 'height': h})
            img = Image.frombytes('RGB', shot.size, shot.bgra, 'raw', 'BGRX')
            return CaptureResult(img, bbox, {'mode': 'mss_window'})
        except Exception:
            return CaptureResult(None, None, {'mode': 'mss_window'})

class ObsVirtualCamBackend(CaptureBackend):
    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None
    def capture(self) -> CaptureResult:
        if cv2 is None:
            return CaptureResult(None, None, {'mode': 'obs'})
        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return CaptureResult(None, None, {'mode': 'obs'})
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            return CaptureResult(img, None, {'mode': 'obs'})
        except Exception:
            return CaptureResult(None, None, {'mode': 'obs'})

class WGCCaptureBackend(CaptureBackend):
    """Windows Graphics Capture(선택 설치). 구현 단순화를 위해 windows-capture 래퍼 의존.
    설치가 없으면 자동 비활성화.
    """
    def __init__(self, locator: GameWindowLocator):
        self.locator = locator
        try:
            import windows_capture as wc  # type: ignore
            self.wc = wc
        except Exception:
            self.wc = None
        self.capturer = None
        self._cap_control = None
        self._last_frame = None
    def _on_frame(self, frame):
        # windows_capture.Frame
        try:
            self._last_frame = frame
        except Exception:
            pass
    def capture(self) -> CaptureResult:
        # 모듈 미존재
        if self.wc is None:
            return CaptureResult(None, None, {'mode': 'wgc', 'error': 'import-failed'})
        # HWND 탐색
        main, child = self.locator.get_handles()
        if not main:
            return CaptureResult(None, None, {'mode': 'wgc', 'error': 'no-hwnd'})
        try:
            # 이벤트 기반 캡처 세팅
            if self.capturer is None:
                cap_cls = getattr(self.wc, 'WindowsCapture', None) or getattr(self.wc, 'NativeWindowsCapture', None)
                if cap_cls is None:
                    return CaptureResult(None, None, {'mode': 'wgc', 'error': 'no-ctor'})
                self.capturer = cap_cls()
                # 콜백 등록
                if hasattr(self.capturer, 'on_frame_arrived'):
                    self.capturer.on_frame_arrived(self._on_frame)
                if hasattr(self.capturer, 'on_closed'):
                    self.capturer.on_closed(lambda: None)
                # 시작
                start = getattr(self.capturer, 'start', None)
                if start is None:
                    return CaptureResult(None, None, {'mode': 'wgc', 'error': 'no-start'})
                try:
                    # 키워드 인자로 창 핸들 전달
                    self._cap_control = start(window=main, cursor_capture=False)
                except TypeError:
                    try:
                        self._cap_control = start(main)
                    except Exception as e:
                        return CaptureResult(None, None, {'mode': 'wgc', 'error': f'start-failed:{e.__class__.__name__}', 'detail': str(e)[:160]})
                # 초기 프레임 유입 대기 (짧은 블록)
                try:
                    if hasattr(self._cap_control, 'wait'):
                        self._cap_control.wait(100)
                except Exception:
                    pass
            # 프레임 수신 여부 확인
            frame = self._last_frame
            if frame is None:
                return CaptureResult(None, None, {'mode': 'wgc', 'error': 'no-frame'})
            # Frame.convert_to_bgr() -> numpy ndarray(BGR)
            arr = None
            if hasattr(frame, 'convert_to_bgr'):
                arr = frame.convert_to_bgr()
            if arr is None:
                return CaptureResult(None, None, {'mode': 'wgc', 'error': 'no-array'})
            if arr.ndim == 3:
                # BGR -> RGB
                arr = arr[:, :, ::-1]
                img = Image.fromarray(arr.copy())
                return CaptureResult(img, None, {'mode': 'wgc'})
            return CaptureResult(None, None, {'mode': 'wgc', 'error': 'unexpected-shape', 'shape': getattr(arr, 'shape', None)})
        except Exception as e:
            return CaptureResult(None, None, {
                'mode': 'wgc',
                'error': f'exception:{e.__class__.__name__}',
                'detail': str(e)[:200]
            })

class WGCPureNativeBackend(CaptureBackend):
    """WinRT 기반 네이티브 WGC (프레임 미구현 시 None 반환하여 폴백 유도).
    capture_mode=wgc_native 에서 최우선 시도.
    """
    def __init__(self, locator: GameWindowLocator):
        self.locator = locator
        try:
            from .wgc_native import WgcSession, HAVE_WINRT  # relative import
            self.session_cls = WgcSession
            self.have_winrt = HAVE_WINRT
        except Exception:
            self.session_cls = None
            self.have_winrt = False
        self.session = None
        self._started = False
    def capture(self) -> CaptureResult:
        if not self.have_winrt or self.session_cls is None:
            return CaptureResult(None, None, {'mode': 'wgc_native', 'error': 'winrt-unavailable'})
        if self.session is None:
            self.session = self.session_cls()
            ok = self.session.ensure_item()
            if not ok:
                return CaptureResult(None, None, {'mode': 'wgc_native', 'error': 'picker-failed'})
            if not self.session.start():
                return CaptureResult(None, None, {'mode': 'wgc_native', 'error': 'start-failed'})
        img = self.session.get_frame()
        if img is None:
            return CaptureResult(None, None, {'mode': 'wgc_native'})
        return CaptureResult(img, None, {'mode': 'wgc_native'})

class GameCapture:
    """백엔드 체인으로 안정 캡처 제공"""
    def __init__(self, locator: GameWindowLocator, mode: str):
        self.locator = locator
        self.mode = (mode or 'window').lower()
        self.backends = self._build_backends()
        self._black_thresh = CONFIG.get('black_threshold', 1.0)
        self._black_skip = CONFIG.get('black_skip', True)
    def _is_black(self, img: Image.Image) -> bool:
        if not self._black_skip or img is None:
            return False
        try:
            arr = np.asarray(img)
            # 지원 형식: HxW, HxWx3/4
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            mean = float(np.mean(arr))
            return mean <= self._black_thresh
        except Exception:
            return False
    def _build_backends(self):
        order = []
        if self.mode == 'obs':
            order.append(ObsVirtualCamBackend(CONFIG['obs_device_index']))
        elif self.mode == 'wgc':
            order.append(WGCCaptureBackend(self.locator))
        elif self.mode == 'wgc_native':
            order.append(WGCPureNativeBackend(self.locator))
        elif self.mode == 'client':
            order.append(PrintWindowClientBackend(self.locator))
            order.append(GDIClientBackend(self.locator))
            order.append(MSSClientBackend(self.locator))
        elif self.mode == 'frame':
            order.append(WindowRectBackend(self.locator))
            order.append(MSSWindowBackend(self.locator))
        else:  # window 기본
            # 다중 모니터/좌표 이슈 회피를 위해 PrintWindow 우선 시도
            order.append(PrintWindowClientBackend(self.locator))
            order.append(GDIClientBackend(self.locator))
            order.append(WindowRectBackend(self.locator))
            order.append(MSSClientBackend(self.locator))
            order.append(MSSWindowBackend(self.locator))
            order.append(WGCCaptureBackend(self.locator))
            order.append(WGCPureNativeBackend(self.locator))
            order.append(FullscreenBackend())
        return order
    def get_frame(self) -> CaptureResult:
        last_meta: Dict = {'mode': self.mode, 'error': 'all-backends-failed'}
        for b in self.backends:
            res = b.capture()
            if res.meta:
                last_meta = res.meta
            if res.image is not None:
                if self._is_black(res.image):
                    # 검은 프레임은 무시하고 다음 백엔드로 폴백
                    last_meta = {**(last_meta or {}), 'black_skip': True}
                    continue
                return res
        return CaptureResult(None, None, last_meta)

class FrameAnalyzer:
    """간단 전처리/특성 추출기 (입력 점유 無)"""
    def __init__(self, size=(160,120)):
        self.size = size
        self.prev = None
    def analyze(self, img: Image.Image) -> Dict:
        if img is None:
            return {}
        im = img.resize(self.size)
        arr = np.asarray(im)
        meta: Dict = {}
        if cv2 is None:
            meta['brightness'] = float(np.mean(arr))
            return meta
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        brightness = float(np.mean(arr))
        sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        edge_h = float(np.mean(np.abs(sobelx)))
        edge_v = float(np.mean(np.abs(sobely)))
        red_mask = cv2.inRange(hsv, (0,50,50), (10,255,255))
        blue_mask = cv2.inRange(hsv, (100,50,50), (130,255,255))
        total = arr.shape[0]*arr.shape[1]
        red_ratio = float(np.sum(red_mask))/total
        blue_ratio = float(np.sum(blue_mask))/total
        movement = 0.0
        if self.prev is not None and self.prev.shape == gray.shape:
            movement = float(np.mean(np.abs(gray.astype(np.float32)-self.prev.astype(np.float32))))
        self.prev = gray
        meta.update({
            'brightness': brightness,
            'edge_h': edge_h,
            'edge_v': edge_v,
            'red_ratio': red_ratio,
            'blue_ratio': blue_ratio,
            'movement': movement
        })
        return meta

def build_capture_pipeline() -> Tuple[GameWindowLocator, GameCapture, FrameAnalyzer]:
    locator = GameWindowLocator(CONFIG['win_title'], CONFIG['win_class'], CONFIG['proc_exe'])
    capture = GameCapture(locator, CONFIG['capture_mode'])
    analyzer = FrameAnalyzer()
    return locator, capture, analyzer
