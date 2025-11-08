# 두뇌(Brain) 모듈 설계

## 개요

게임 상태(JSON 요약)를 받아 키 입력 행동(JSON)을 생성하는 로컬 추론 엔진. 초기에는 DeepSeek-R1 스타일 LLM(gguf) 또는 Llama 계열 사용, 출력은 반드시 JSON 스키마 강제.

## 선택: DeepSeek 계열 사용 이유

- 합리적 추론(Chain-of-Thought)을 잘하지만 여기서는 CoT를 내부적으로 숨기고 최종 JSON만 사용.
- 8B 규모(Q4_K_M)면 RTX 4090에서 1토큰 수십 ms 수준.
- 향후 Llama 3.2 8B, Gemma 3 4B와 교체/비교 용이 (동일 인터페이스 유지).

## 인터페이스 계약

BrainBase.decide(state: dict) -> {"actions": [ {...}, ... ]}

Action 항목 스키마:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| type | string | ✔ | tap/key_hold/sequence/noop 타입 지정 |
| key  | string | 조건 | tap/key_hold에서 필요 (up, down, left, right, enter, space 등) |
| ms   | int    | 선택 | key_hold 지속 시간(ms) |

## 프롬프트 최소 형태 예시

```text
You are a game automation policy generator.
Return ONLY valid JSON with key 'actions'.
State:
{"scene":"battle","hp":73,"enemies":2,"recent":["move_up","attack"]}
JSON:
```

## 출력 파싱 전략

1. 모델 전체 텍스트에서 첫 '{'와 마지막 '}' 사이 추출
2. json.loads 실패 시 fallback 정책 (enter + up)
3. actions 배열 검증 (list 여부, 각 요소 type/key 유효성)

## 안전장치

- 토큰 폭주 방지: max_tokens 제한 (예: 128)
- stop 토큰 사용 ('\n') → 모델이 추가 비서 텍스트 길게 생성하지 않도록
- 잘못된 키: 화이트리스트 {up,down,left,right,enter,space,esc} 이외는 무시(추후 controller 확장)

## 향후 개선

- 온톨로지 추가: 전략 태그(explore/combat/loot) → LLM이 상황별 정책 선택
- CoT 출력 허용 + 후처리로 JSON만 추출 (성능 비교 실험)
- RLHF 또는 행동 결과 보상에 따른 미세튜닝

## 대체 모델 비교 간단표

| 모델 | 장점 | 단점 | 비고 |
|------|------|------|------|
| DeepSeek-R1 8B | 추론 안정성, CoT 좋음 | 양자화 전 메모리 큼 | Q4_K_M 권장 |
| Llama 3.2 8B | 생태계 풍부 | 추론 성향 일반적 | 동일 프롬프트 재사용 |
| Gemma 3 4B | 긴 컨텍스트(128K) | 소형이라 복잡 전략 한계 | 메모리 절약 모드 |
| Mistral 7B Instruct | 빠른 응답 | 일부 정책 일관성 낮음 | 대체 실험용 |

## 파일 구조

```text
src/llm/
  brain_base.py       # 인터페이스
  brain_deepseek.py   # DeepSeekBrain 구현 (llama_cpp 로드)
```
