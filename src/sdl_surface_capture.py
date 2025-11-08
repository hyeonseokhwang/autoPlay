# -*- coding: utf-8 -*-
"""
SDL Surface Direct Capture 연구 스텁
===================================
목표: DOSBox 내부 SDL 서피스(프레임버퍼)에 직접 접근하여 창 가림/최소화 상태에서도 저오버헤드 픽셀 획득.

접근 후보:
1) DLL 인젝션 + SDL 함수 후킹
   - 대상: SDL 1.x (DOSBox 고전 빌드)에서 사용되는 SDL_UpdateRect / SDL_Flip / SDL_SetVideoMode.
   - 방식: CreateToolhelp32Snapshot로 dosbox.exe PID 확인 → OpenProcess → VirtualAllocEx + WriteProcessMemory로 injector DLL 경로 전달 → CreateRemoteThread(LoadLibraryW).
   - DLL 내부에서 GetProcAddress로 SDL 관련 심볼 찾고, MinHook(외부 라이브러리) 또는 수작업 JMP 패치로 후킹.
   - 후킹 함수에서 원래 서피스(SDL_Surface *screen)의 pixels(uchar*) 버퍼를 복사하여 공유 메모리(Memory Mapped File)로 노출.
   - Python 측: mmap 열고 numpy.frombuffer로 매 프레임 읽기 → PIL.Image.fromarray.

2) BitBlt/GDI 후킹
   - DOSBox가 GDI BitBlt를 사용해 윈도우 DC로 복사한 뒤 최종 화면 출력 시 이를 후킹.
   - EasyHook/Detours 기반으로 BitBlt 또는 StretchDIBits 인라인 패치.
   - 장점: SDL 내부 구조 몰라도 됨. 단점: 창 최소화 시 BitBlt 호출 감소/중단 가능.

3) Windows Graphics Capture PID Surface
   - WGC API(WindowGraphicsCaptureItem.CreateFromWindow) 이미 사용 중이나 최소화 시 실패 가능.
   - WinRT 경로에서 Direct3D duplication 레벨 최적화 연구(현재 wrapper 실패).

4) Desktop Duplication (IDXGIOutputDuplication)
   - 전체 화면 복제 후 대상 창 bbox로 crop.
   - 최소화/가림 시에도 백버퍼 픽셀 유지되는지 실험 필요 (일반적으로 occluded면 컴포지터가 placeholder 제공 → 게임 내부 픽셀은 잃을 수 있음).

5) DOSBox 소스 기반 IPC
   - DOSBox를 커스텀 빌드하여 SDL Flip 직전에 raw framebuffer를 공유 메모리로 투사.
   - 가장 안정적이지만 바이너리 교체 필요.

리스크 및 고려 사항:
- DLL 인젝션은 보안/안티치트 환경에서 위험. 오프라인 싱글플레이 용도라면 허용 가능.
- MinHook/Detours 외부 의존성 추가 필요.
- 64비트/32비트 불일치(주로 DOSBox는 32비트 빌드 가능) → 인젝션 DLL 아키텍처 맞춰야 함.
- 성능: 매 프레임 복사 시 memcpy 비용. 640x480 RGB 약 900KB/frame → 60fps ≈ 54MB/s, 허용 범위.

프로토타입 단계 제안 순서:
A. 화면 표본화 빈도 낮춤(10~15fps)으로 안정성 검증.
B. GDI 후킹으로 구현 난이도 낮은 버전 측정.
C. DLL 인젝션 + SDL_Surface 추출 버전 제작.
D. 필요시 커스텀 DOSBox 빌드.

현재 스텁은 API 형태만 정의하고 항상 None 반환 → 상위 파이프라인에서 폴백 처리.
"""
from __future__ import annotations
from typing import Optional
from PIL import Image

class SDLSurfaceCapture:
    def __init__(self):
        # 실제 구현 시: 공유 메모리 핸들 / 후킹 초기화 상태 등 저장
        self.initialized = False
        self.last_error: Optional[str] = None

    def available(self) -> bool:
        """현재 SDL 직접 캡처 경로 사용 가능 여부.
        구현 전이므로 False 고정."""
        return False

    def initialize(self) -> bool:
        """후킹/공유메모리 초기화 (미구현)."""
        self.last_error = "not-implemented"
        self.initialized = False
        return False

    def capture_frame(self) -> Optional[Image.Image]:
        """SDL 서피스에서 한 프레임을 PIL.Image로 반환.
        미구현 상태이므로 항상 None.
        구현 시 RGB565/Indexed 변환 처리 필요 가능."""
        return None

__all__ = ["SDLSurfaceCapture"]
