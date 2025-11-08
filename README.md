# 영웅전설4 자동 플레이 AI (Windows/DOSBox)

Windows 환경에서 DOSBox로 실행되는 영웅전설4(ED4)를 자동으로 플레이하기 위한 AI 기반 캡처/인식/행동 시스템의 단계적 구현 저장소입니다. 현재는 안정적인 멀티 백엔드 화면 캡처와 간단 특성 추출, 원샷 인식 스크립트까지 완료된 상태입니다.

## 포함된 기능

- 멀티 백엔드 캡처 (PrintWindow → GDI → MSS → WGC 시도) + 블랙 프레임 자동 폴백
- 윈도우 핸들 추적 (타이틀/클래스/프로세스 필터)
- 특성 추출 (밝기/엣지/색 비율/움직임)
- 원샷 캡처 스크립트(`scripts/recognize_once.py`) + 씬 분류기(옵션, 핫 리로드 지원)
- 격리된 입력 전송(포커스 탈취 없이 SendMessage 기반) - 메인 AI 코드에서 사용

## 설치

필수: Python 3.8+

```cmd
pip install -r requirements.txt
```

## 개발 진행 단계

완료: 캡처 파이프라인, 특성 추출, 원샷 스크립트, 분류기 핫 리로드, AI 루프 내 분류 예측 주입, 행동 정책(ActionPolicy) 통합, SDL 서피스 연구 스텁.

예정:

- SDL 표면 직접 접근 구현 단계(DLL 인젝션/후킹) 본격화
- 분류기 실제 학습/검증 데이터 축적

## 빠른 사용법

환경 변수로 대상 DOSBox 창 힌트 제공 후 한 프레임 캡처:

```cmd
set HERO4_WIN_TITLE=DOSBox
set HERO4_PROC_EXE=dosbox
python scripts\recognize_once.py
```

출력: 저장된 PNG 경로, 사용 백엔드, 주요 특성값, (활성화된 경우) 씬 분류 확률.

## 핵심 파일

- `src/screen_pipeline.py` : 캡처 백엔드 체인 + FrameAnalyzer
- `src/recognition.py` : 단일 호출 인식 래퍼
- `src/state_classifier.py` : 경량 CNN + 핫 리로드
- `isolated_rag_ai.py` : RAG + 장기 세션 + 격리 입력 + 분류기 통합
- `scripts/recognize_once.py` : 원샷 캡처 CLI

## 추가 파일

- `src/action_policy.py` : 행동 성공률 기반 가중치 정책
- `src/sdl_surface_capture.py` : SDL 서피스 직접 캡처 연구 스텁

## 환경 변수(일부)

| 변수 | 의미 | 기본 |
|------|------|------|
| HERO4_WIN_TITLE | 창 타이틀 포함 문자열 | dosbox |
| HERO4_CAPTURE_MODE | window/client/wgc/obs | window |
| HERO4_STATE_MODEL | 분류기 가중치 경로 | models/state_cnn.pt |
| HERO4_USE_CLASSIFIER | 1이면 분류기 사용 | 1 |
| HERO4_BLACK_SKIP | 검은 프레임 스킵 | 1 |

## 주의사항

- DOSBox 창이 최소화되면 일부 백엔드 실패 가능 (MSS/전체화면 폴백 권장)
- 초기 분류기는 랜덤 가중치이므로 실제 학습 전까지 신뢰도 낮음
- 관리자 권한 없이도 작동하나, 일부 환경에서 핸들/PrintWindow 접근 실패 시 권한 상승 필요할 수 있음

## 라이선스

추후 명시 예정.

---

지속적인 자동 학습(RAG + 재훈련) 및 행동 정책 최적화를 위한 기반을 확장 중입니다.
