# 아키텍처 & 로드맵 개요

## 목표

DOSBox에서 실행되는 영웅전설4 화면을 실시간으로 분석하고 로컬 LLM/비전 모델을 결합하여 자동 플레이를 수행하는 완전 로컬 AI 파이프라인.

## 상위 컴포넌트

| 컴포넌트 | 책임 | 입력 | 출력 |
|----------|------|------|------|
| Capture Layer | DOSBox 창 픽셀 획득 | Win32 HWND | BGR Frame (numpy) |
| Frame Bus | 최신 프레임 버퍼 유지, 구독 지원 | Frames | Latest frame ref |
| Vision Perception | 객체/커서/OCR/씬 분류 | Frame | JSON 상태 기술 |
| Policy (LLM) | 목표 + 상태 → 행동 결정 | State JSON + Memory | Action JSON |
| Action Executor | 행동 JSON을 키/마우스/메시지로 변환 | Action JSON | OS 이벤트 / 결과 |
| Memory Manager | 최근 요약 + 장기 진행 저장 | Actions/State | 압축 요약 / 퀘스트 상태 |
| Logging/Telemetry | 성능/오류 기록 | 모든 | 로그 파일/metrics |

## 데이터 계약 (초기 버전)

### 1. Vision 상태 JSON (예시)

```jsonc
{
  "timestamp": 1731080000.123,
  "scene": "battle",               // 분류 결과
  "objects": [                      // YOLO 등 검출
    {"cls": "enemy", "bbox": [120,60,40,40], "conf": 0.91},
    {"cls": "menu_item", "bbox": [500,420,120,32], "conf": 0.78}
  ],
  "cursor": {"x": 300, "y": 220, "conf": 0.88},
  "texts": [                        // OCR 결과 ROI별
    {"text": "HP 73/100", "bbox": [10,450,120,20], "conf": 0.95},
    {"text": "공격", "bbox": [520,420,60,20], "conf": 0.92}
  ],
  "metrics": {"brightness": 0.54, "edge_density": 0.31}
}
```

### 2. Policy 입력 구조

```jsonc
{
  "frame_state": { /* Vision 상태 JSON 축약 */ },
  "goal": "승리 후 HP 50% 이상 유지",
  "recent_actions": ["move_up","move_up","attack"],
  "memory_summary": {
    "quest_phase": "early_forest",
    "hp_trend": "stable",
    "mana_trend": "low"
  },
  "format": "JSON_ACTIONS"
}
```

### 3. Policy 출력(Action JSON)

```jsonc
{
  "actions": [
    {"type": "key_hold", "key": "up", "ms": 220},
    {"type": "tap", "key": "space"},
    {"type": "tap", "key": "enter"}
  ],
  "confidence": 0.81,
  "notes": "Approach enemy then confirm skill"
}
```

### 4. Executor 내부 변환

| Action type | 처리 방식 | 상세 |
|-------------|-----------|------|
| tap | WM_KEYDOWN + WM_KEYUP | 짧은 키 입력 |
| key_hold | WM_KEYDOWN 후 sleep → WM_KEYUP | ms 기준 유지 |
| sequence | 여러 액션 확장 | 매크로처럼 직렬 실행 |
| noop | 무시 | 안전 대기 |

## 쓰레드 & 큐 구조(초기)

```text
[Capture Thread] --> (Latest Frame Var) --> [Vision Thread] --> (State Queue) --> [Policy Thread] --> (Action Queue) --> [Executor Thread]
```

- 캡처는 고정 FPS (예: 12fps)
- Vision Thread는 가장 최신 프레임만 소비 (큐 과잉시 이전 프레임 폐기)
- Policy는 새 상태 도착 시 이전 요청 취소(또는 스킵) 후 최신만 처리
- Executor는 Action Queue를 직렬 처리 (중복 hold 융합 가능)

## 로드맵 단계별 (요약)

| 단계 | 목표 | 조건 | 결과 |
|------|------|------|------|
| P0 | 캡처 + 랜덤 움직임 | 화면 캡처 안정 | 캐릭터 이동 확인 |
| P1 | 커서/메뉴 템플릿 탐지 | 템플릿 확보 | 메뉴 열림 인식 |
| P2 | 씬 분류(ResNet18) | 라벨 300장 | 자동 장면 구분 |
| P3 | YOLOv8s UI/적/아이템 | 라벨 600~800장 | 객체 기반 정책 |
| P4 | OCR 대화/메뉴 | ROI 안정 | 텍스트 해석 행동 반영 |
| P5 | LLM(3B→8B) 정책 | JSON 출력 안정 | 전략성 행동 |
| P6 | 메모리 요약 + 장기 목표 | 요약 성능 측정 | 컨텍스트 유지 개선 |
| P7 | TensorRT/양자화 최적화 | 모델 정확도 유지 | FPS/지연 감소 |
| P8 | 음성/웹 인터페이스 | 정책 안정 | 멀티 입력 모드 |
| P9 | RLHF/강화학습 실험 | Rollout infra | 학습형 정책 |

## 에러 모드 & 복구 전략

| 에러 | 원인 | 복구 |
|------|------|------|
| 캡처 None 연속 | HWND 변경/창 최소화 | 윈도우 재검색 + 지연 재시도 |
| Vision timeout | 과부하/모델 정지 | 프레임 스킵 + watchdog 재기동 |
| LLM JSON 파싱 실패 | 출력 포맷 붕괴 | 마지막 유효 액션 반복 + 경고 로그 |
| Executor 키 실패 | HWND 권한 문제 | 포커스 강제/대체 pyautogui 모드 |
| 메모리 누수 | 축적된 텐서 | 주기적 GPU 캐시 flush + 프로파일 로그 |

## 성능 KPI (목표)

| 항목 | 목표값 |
|------|--------|
| 평균 캡처→비전 지연 | < 45 ms (FP16, 640x480) |
| LLM 응답 시간(8B Q4) | < 500 ms (정책 프롬프트) |
| 전체 루프 주기(액션 반영) | < 750 ms |
| 안정 실행 시간 | 8h+ 무중단 |
| 프레임 드롭률 | < 3% |

## 추후 확장 아이디어

- 멀티모달(이미지+텍스트) 정책: CLIP 임베딩을 LLM 프롬프트에 주입
- 행동 품질 평가 모델(Reward Model) 추가
- 작업 스케줄러: 저빈도 태스크(OCR 전체 스캔, 장기 메모리 압축) 시간 슬라이스 처리
- 자동 라벨링 지원(전이 학습된 YOLO 결과 수동 검수 UI)

## 결론

이 구조는 최소 기능(MVP)부터 고도화(RLHF, 음성, 멀티모달)까지 단계적 확장을 가능하게 하는 모듈식 설계를 제공한다. 문서화된 데이터 계약을 엄격히 유지하면 모델 교체/업그레이드가 용이해지고, 고성능 하드웨어 자원을 효율적으로 사용하여 실시간에 가까운 게임 자동화가 가능하다.
