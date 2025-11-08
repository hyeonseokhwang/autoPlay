# 영웅전설4 AI 플레이어

도스박스에서 실행되는 영웅전설4를 자동으로 플레이하는 AI 시스템입니다.

## 주요 기능

- **화면 캡처**: 도스박스 윈도우 실시간 캡처
- **게임 상태 인식**: 컴퓨터 비전을 통한 맵, 캐릭터, UI 요소 분석
- **AI 결정 시스템**: 게임 상황에 따른 최적 행동 선택
- **자동 입력**: 방향키 및 액션 키 자동 입력

## 시스템 요구사항

- Python 3.8+
- Windows 10/11
- DOSBox 설치 및 영웅전설4 게임

## 설치 방법

1. 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 게임 실행:
```bash
python main.py
```

## 프로젝트 구조

```
src/
├── screen_capture.py   # 화면 캡처 모듈
├── game_vision.py      # 게임 화면 분석
├── ai_player.py        # AI 플레이어 로직
├── input_controller.py # 키보드 입력 제어
└── main.py            # 메인 실행 파일
config/
└── settings.json      # 설정 파일
```

## 사용 방법

1. DOSBox에서 영웅전설4 실행
2. AI 플레이어 스크립트 시작
3. 게임 창이 인식되면 자동 플레이 시작

## 주의사항

- 게임 해상도와 창 크기 설정 필요
- DOSBox 윈도우가 활성 상태여야 함
- 관리자 권한으로 실행 권장