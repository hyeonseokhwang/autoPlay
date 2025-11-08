# 모델 선택 가이드 (High-End Desktop 기준)

> 시스템 스펙: i9-14900KF (24C/32T), RTX 4090 24GB, DDR5 128GB, NVMe SSD.  
> 목표: 영웅전설4(DOSBox) 화면 기반 실시간(준실시간) 자동 플레이 + 점진적 학습.  
> 원칙: 전부 로컬 실행, 네트워크 호출 없음, 확장 가능 아키텍처.

## 전체 파이프라인 역할별 모델

| 역할 | 1차(빠른 구축) | 2차(정밀) | 3차(고도화/연구) |
|------|----------------|-----------|------------------|
| GUI/오브젝트 검출 | YOLOv8n (초경량) | YOLOv8s / YOLOv5s (균형) | YOLOv8m + TensorRT FP16 / INT8 |
| 커서/특수 픽셀 | 템플릿 매칭(OpenCV) | 커스텀 YOLO 클래스 추가 | 세그멘테이션(DeepLab v3) |
| 텍스트(OCR) | PaddleOCR (KO+EN) | PaddleOCR + 경량 후처리 (반복 제거) | 자체 미세튜닝 OCR 헤드 |
| 상태/씬 분류 | 경량 CNN (ResNet18) | ViT-small / ConvNeXt-tiny | CLIP 기반 멀티모달 재사용 |
| 액션 정책(LLM) | Llama 3.2 3B Instruct (Q4) | Llama 3.2 8B Instruct (Q4_K_M) | Llama 3.2 70B(부분 offloading) 연구 용 |
| 전략/메모리 | 간단 요약 버퍼(JSON) | LLM + 벡터메모리(FAISS local) | 장기 persistence + RLHF loop |
| 음성(향후) | Vosk small KO/EN | Whisper medium Q5 | Whisper large-v3 + GPU 가속 |

---

## 세부 선택 근거

### 1. 비전 검출 (YOLO 계열)

- 왜 YOLOv8s?: 4090 메모리 충분 → 속도/정확도 균형. 640x480 해상도에서 Batch=1 실시간 150~200 FPS 가능.
- 최초 학습 데이터 부족 → v8n으로 빠르게 프로토타입, 이후 클래스 안정되면 v8s 재학습.
- TensorRT 변환: FP16으로 레이턴시 추가 30~40% 감소. 추후 INT8 Calibration 데이터(200~300 라벨 샘플) 확보 후 적용.

추론 백엔드 옵션:

| 백엔드 | 장점 | 예상 FPS (4090, 640x640) |
|--------|------|-------------------------|
| PyTorch FP32 | 가장 단순 | v8s ~140 |
| PyTorch FP16 | CUDA 메모리 절감 | v8s ~170 |
| TensorRT FP16 | 최고속도/최적화 | v8s ~200 |
| TensorRT INT8 | 더 낮은 지연 | v8s ~210 (정밀도 손실 경미) |

### 2. 커서 검출

- 화살표/손모양 등 고정 → 템플릿 매칭(CV2 matchTemplate + 이진 마스크)로 처음 구현. > YOLO 커서 클래스는 후순위.
- 프레임 노이즈 적음 → threshold 기반 거리 < 0.1이면 확정.

### 3. OCR

- PaddleOCR: 한국어/영어 혼용 지원 + GPU 가속. Tesseract는 한글 정확도/속도 낮음.
- 전략: 대화창/메뉴 ROI만 잘라서 OCR 실행(전체 프레임 OCR 금지). 평균문자수 적으므로 레이턴시 < 30ms 기대.
- 후처리: 반복 행 제거, 특수문자 필터.

### 4. 씬/상태 분류

- 1단계: ResNet18 (pretrain 사용 후 마지막 FC 재학습) → 640x480 입력을 224 리사이즈.
- 클래스: town / field / battle / menu / dialog / transition (6개 시작).
- 2단계: ViT-small (더 잘 일반화). 4090에서 224 입력 실시간 충분.
- 3단계: CLIP image encoder + 텍스트 프롬프트("battle scene", "in town") Zero-shot 개선.

### 5. 액션 정책 LLM

모델 레벨별 프로파일:

| 모델 | 형식 | 메모리(Q4) | 1토큰 추론(ms, 대략) | 용도 |
|------|------|-----------|----------------------|------|
| Llama 3.2 3B | Q4_K_M | ~2.2GB | ~15-25 | 빠른 루프/테스트 |
| Gemma 3 4B | Q4_K_M | ~2.6GB | ~20-30 | 대안(긴 컨텍스트) |
| Llama 3.2 8B | Q4_K_M | ~4.5GB | ~30-45 | 메인 정책 추론 |
| Mistral 7B Instruct | Q4_K_M | ~4.0GB | ~35-50 | 대체 비교 |
| Llama 3.2 70B | Q4_0 (mixed) | >40GB | >120 | 연구/비실시간 |

선택: **Llama 3.2 8B (Q4_K_M)** 메인.  
Fallback: 3B (고속 모드) / Gemma 3 4B (긴 컨텍스트 필요 시).  
On-GPU full load (24GB) 가능, 필요 시 일부 레이어 CPU offload.

프롬프트 설계 핵심:

```json
{
  "frame_summary": {"scene": "battle", "hp": 70, "mp": 20, "enemies": 3},
  "recent_actions": ["move_up","move_up","attack"],
  "goal": "conserve hp and win",
  "constraints": ["avoid low mana skills"],
  "format": "JSON_ACTIONS"
}
```

LLM 출력 예:

```json
{"actions": [
  {"type": "key_hold", "key": "up", "ms": 180},
  {"type": "tap", "key": "space"},
  {"type": "tap", "key": "enter"}
]}
```

파싱 실패 시: 최근 유효 정책 반복 + 로그 기록.

### 6. 메모리/전략 관리

- 단기 버퍼: 최근 50 프레임 요약(씬, 주요 이벤트, HP 변동)
- 행동 히스토리 압축: run-length + 중복 제거 → LLM 프롬프트 토큰 절약
- 장기 목표: 캐릭터 진행(퀘스트 단계) 간략 JSON 스토어(`memory_state.json`)

### 7. 음성(후순위)

- Vosk KO/EN 소형 모델로 명령 인식("다음 뭐해", "아이템 사용")
- Whisper 빠른 모드(Q5)로 업그레이드 → 정확도 향상
- 파이프라인: 음성 → 텍스트 → LLM 정책 → 액션

### 8. Quantization / 최적화 우선순위

1. LLM: Q4_K_M (4090에서 충분 속도) → 필요 시 Q6_K로 약간 품질↑
2. YOLO: FP16 TensorRT 변환; INT8는 충분 라벨 후
3. OCR: PaddleOCR 기본 FP32 후 점진적 FP16
4. 분류기: ResNet18 FP32 → ONNX → TensorRT FP16

### 9. 단계별 도입 순서 (실행 시퀀스)

1. 캡처 루프 + 더미 정책 (랜덤 이동) 안정화
2. 템플릿 커서 + 메뉴 ROI 추출 → 상태 분류(ResNet18)
3. YOLOv8s 객체 검출 학습 적용 (아이템/메뉴/대화박스)
4. PaddleOCR 통합(대화/메뉴 텍스트)
5. LLM(3B)로 행동 JSON 생성 → 파서/신뢰도 체크
6. LLM 교체(8B) + 메모리 요약 버퍼 추가
7. TensorRT 최적화 / INT8 Calibration
8. 음성 입력 + 웹 UI 연동

### 10. 위험/대응

| 위험 | 설명 | 대응 |
|------|------|------|
| OCR 잡음 | 구형 폰트/저해상 | ROI 확대 + 이진화 전처리 |
| LLM 헛소리 | 잘못된 키 시퀀스 생성 | JSON 스키마 강제 + 화이트리스트 |
| 프레임 지연 | 동시 처리 병목 | 캡처/인식/정책 분리 큐 구조 |
| 데이터 부족 | YOLO 학습 한계 | 자동 캡처 + 반자동 라벨 툴 추가 |
| 메모리 누수 | 장기 실행 자원 증가 | 주기적 torch.cuda.empty_cache + 프로파일링 |

### 11. 파일/모듈 권장 구조 (추가 예정)

```
src/
  capture/          # screen_capture.py, frame_bus
  vision/           # yolo_detector.py, cursor_finder.py, ocr.py, scene_classifier.py
  llm/              # policy_llm.py, prompt_builder.py, memory.py
  actuation/        # key_sender.py, action_executor.py
  pipeline/         # main_loop.py, scheduler.py
  utils/            # timing.py, logging.py, config_loader.py
models/
  yolo/             # weights
  ocr/              # ocr model files
  llm/              # gguf or llama.cpp model files
  classifier/       # resnet18.ckpt
```

---

## 결정 요약 (최종 추천 세트)

- Vision: YOLOv8s (PyTorch FP16 → 이후 TensorRT)
- Cursor: 템플릿 매칭 초기 / 필요 시 YOLO 클래스 추가
- OCR: PaddleOCR (KO+EN) ROI 기반 호출
- Scene: ResNet18 → ViT-small 승급
- Policy LLM: Llama 3.2 8B Instruct (Q4_K_M), Fallback 3B
- Memory: 요약 JSON + 주기적 축약
- 최적화: TensorRT(FP16) + LLM Q4_K_M, 추후 INT8 실험

추가 요구사항/변경 요청 있으면 이 문서 갱신.
