# -*- coding: utf-8 -*-
"""
Windows Graphics Capture (네이티브) 백엔드 - OBS 의존 제거용.

요구 사항:
  pip install winrt pillow numpy
(선택) pip install pywin32  (창 핸들 탐색은 기존 locator가 처리)

주의:
  - 첫 사용 시 GraphicsCapturePicker를 띄워 사용자가 DOSBox 창을 직접 선택해야 함.
  - 선택된 GraphicsCaptureItem은 세션 라이프타임 동안 재사용.
  - 최소화/가림 상태에서도 프레임 공급.

이 파일은 간단한 pull 방식 구현:
  WgcSession.start() 호출 -> get_frame()으로 최근 프레임 반환.
고성능을 위해선 FrameArrived 이벤트 + GPU → CPU 복사 최적화 추가 가능.

제한:
  Python에서 Direct3D11 디바이스 핸들 생성/관리 복잡성 때문에 여기서는
  winrt CaptureFrameSurface를 시스템이 BGRA8 Surface로 준다고 가정하고
  WinRT 인터페이스의 CopyFromMemory 접근 대신 임시 Win2D 생략.
  완전한 구현을 위해서는 추가 dxinterop(pybind11) 레이어 필요.

현재 구현은 "동작 스텁" 형태이며, 실제 프레임이 None이면 상위 폴백(backends 체인)이 계속 시도.

사용:
  from wgc_native import WgcSession
  sess = WgcSession()
  if sess.ensure_item():
      sess.start()
      img = sess.get_frame()

"""
from __future__ import annotations
import time
from typing import Optional

try:
    from winrt.windows.graphics.capture import GraphicsCapturePicker
    from winrt.windows.graphics.capture import GraphicsCaptureSession
    from winrt.windows.graphics.capture import Direct3D11CaptureFramePool
    from winrt.windows.graphics.capture import GraphicsCaptureItem
    from winrt.windows.graphics.directx import DirectXPixelFormat
    from winrt.windows.graphics.directx.direct3d11 import IDirect3DDevice
    from winrt.windows.foundation import TypedEventHandler
    HAVE_WINRT = True
except Exception:
    HAVE_WINRT = False

try:
    import numpy as np
    from PIL import Image
except Exception:
    np = None
    Image = None

# 간단한 D3D 디바이스 생성 헬퍼 (WinRT가 요구)
# 완전 구현은 ctypes로 D3D11CreateDevice를 호출해야 하나 여기서는 생략/스텁

def _create_d3d_device_stub() -> Optional[IDirect3DDevice]:
    # 실제 구현을 위해서는 winrt.d3ddevice 인터롭 필요.
    # windows-capture 패키지를 쓸 경우 그쪽 래퍼를 대신 사용하는 것이 편함.
    return None  # 스텁 반환 -> FramePool 생성 실패 시 폴백 유도

class WgcSession:
    def __init__(self):
        self.item: Optional[GraphicsCaptureItem] = None
        self.frame_pool: Optional[Direct3D11CaptureFramePool] = None
        self.session: Optional[GraphicsCaptureSession] = None
        self.running = False
        self.last_frame = None
        self._last_get = 0.0
        self._fps_limit = 30
        self._device = None

    def ensure_item(self) -> bool:
        if not HAVE_WINRT:
            return False
        if self.item:
            return True
        try:
            picker = GraphicsCapturePicker()
            self.item = picker.PickSingleItemAsync().get()  # block until selected
            return self.item is not None
        except Exception:
            return False

    def start(self) -> bool:
        if not HAVE_WINRT or not self.item:
            return False
        if self.running:
            return True
        # 디바이스 생성 (스텁: 실패 시 False 반환)
        self._device = _create_d3d_device_stub()
        if self._device is None:
            # 실제 Direct3D11CaptureFramePool 생성 불가 -> 사용 불가
            return False
        try:
            size = self.item.Size
            self.frame_pool = Direct3D11CaptureFramePool.Create(
                self._device,
                DirectXPixelFormat.B8G8R8A8UIntNormalized,
                2,
                size
            )
            self.session = self.frame_pool.CreateCaptureSession(self.item)
            self.session.StartCapture()
            self.running = True
            return True
        except Exception:
            return False

    def get_frame(self) -> Optional[Image.Image]:
        """최근 프레임 획득. 스텁 구현: frame_pool TryGetNextFrame 미사용."""
        if not self.running or not self.frame_pool:
            return None
        now = time.time()
        if (now - self._last_get) < (1.0 / self._fps_limit):
            return self.last_frame
        self._last_get = now
        try:
            # 실제 구현 필요: frame_pool.TryGetNextFrame() 호출 후 Direct3D 표면 → CPU 배열 변환
            # 여기서는 아직 미구현이므로 None 유지
            # self.last_frame = Image.fromarray(...)
            return self.last_frame
        except Exception:
            return None

    def stop(self):
        if self.session:
            try:
                self.session.Close()
            except Exception:
                pass
        if self.frame_pool:
            try:
                self.frame_pool.Close()
            except Exception:
                pass
        self.session = None
        self.frame_pool = None
        self.running = False
        self.item = None

__all__ = ["WgcSession", "HAVE_WINRT"]
