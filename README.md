# 영웅전설4 스크린샷 도구 + 코치 챗(RAG) (Windows/DOSBox)

이 저장소는 DOSBox 윈도우 화면을 PNG로 저장하는 최소 구성에 더해,
간단한 코치 챗 콘솔(대화 + RAG 메모리)을 포함합니다.

## 구성

- `screenshot.py` : 스크린샷 CLI
- `src/screen_capture.py` : DOSBox 창 캡처 구현(MSS 기반)
- `config/settings.json` : 기본 윈도우 타이틀 등 설정
- `screenshot/` : 캡처 이미지 저장 폴더(기본)
- `chat_console.py` : 코치 챗 콘솔(대화 기록을 SQLite FTS로 저장/검색)
- `src/memory/memory_store.py` : RAG 메모리(FTS5)
- `src/chat/chat_session.py` : 대화 세션 관리

## 설치

Python 3.8+ 권장

```cmd
pip install -r requirements.txt
```

## 사용법

한 장 캡처(기본 경로: `screenshot/`):

```cmd
python screenshot.py --count 1
```

여러 장/간격/영역 지정 예시:

```cmd
python screenshot.py --count 10 --interval 0.5 --prefix hero4
python screenshot.py --region 100,100,320,240 --count 1
python screenshot.py --window-title "DOSBox" --count 3
python screenshot.py --count 1 --open
```

옵션 도움말은 `python screenshot.py -h` 참고.

## 챗 콘솔(코치/RAG)

간단한 훈수/설명 챗을 터미널에서 실행합니다. 과거 대화와 이벤트를
SQLite FTS5(`data/memory.db`)에 저장하고, 질의와 관련된 히스토리를 검색해
컨텍스트로 주입합니다.

```cmd
python chat_console.py --session coach
```

선택: 로컬 LLM(DeepSeek gguf) 경로를 지정할 수 있습니다. `llama_cpp`가 설치되지 않으면
더미 모드로 동작하며 대화는 기록되지만 고급 추론은 제한됩니다.

```cmd
python chat_console.py --model models\llm\deepseek-r1-8b-q4_k_m.gguf --max-tokens 128
```

## 문서

- 모델 선택 가이드: `docs/model_selection.md`
- 아키텍처 & 로드맵: `docs/architecture.md`
- 모델 프로파일 예시: `config/model_profiles.example.yaml`

## 주의

- DOSBox 창이 꺼져 있거나 제목이 다르면 자동 재시도 후 종료합니다.
- 캡처 실패 시 최대 3회 재시도합니다.
