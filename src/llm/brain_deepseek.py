# -*- coding: utf-8 -*-
"""
DeepSeek 기반 두뇌 스텁
- 실제 deepseek(예: DeepSeek-R1 7B/8B) 모델을 로컬 gguf/llama.cpp로 로드한다고 가정.
- 현재는 의존성 최소화를 위해 import 실패 시 graceful degrade.
"""
from __future__ import annotations
import json
from typing import Dict, Any, Optional, List
from .brain_base import BrainBase

try:
    import llama_cpp  # type: ignore
except ImportError:  # 로컬 환경에 아직 설치 안된 경우
    llama_cpp = None  # 타입 회피용

PROMPT_TEMPLATE = """You are a game automation policy generator.\nReturn ONLY valid JSON with key 'actions'.\nExample:\n{"actions":[{"type":"tap","key":"enter"}]}\nState:\n{state}\nJSON:"""

CHAT_TEMPLATE = (
    "You are a helpful coach for The Legend of Heroes IV (DOSBox).\n"
    "Always answer in Korean, concise and practical.\n"
    "If a '컨텍스트' block was provided earlier as a system message, use it.\n"
    "Conversation so far:\n{history}\n"
    "User: {user}\n"
    "Assistant:"
)

class DeepSeekBrain(BrainBase):
    def __init__(
        self,
        model_path: str = "models/llm/deepseek-r1-8b-q4_k_m.gguf",
        max_tokens: int = 128,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self._model = None
        if llama_cpp:
            try:
                llm_kwargs = {
                    "model_path": model_path,
                    "n_ctx": n_ctx,
                    "logits_all": False,
                    "n_gpu_layers": n_gpu_layers,
                }
                if n_threads is not None:
                    llm_kwargs["n_threads"] = n_threads
                self._model = llama_cpp.Llama(**llm_kwargs)
                print(f"[DeepSeekBrain] 모델 로드 완료: {model_path} (n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})")
            except Exception as e:
                print(f"[DeepSeekBrain] 모델 로드 실패: {e}")
                self._model = None
        else:
            print("[DeepSeekBrain] llama_cpp 미설치 - 더미 모드 동작")

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # 더미/백업 경로
        if not self._model:
            return {"actions": [
                {"type": "key_hold", "key": "up", "ms": 180},
                {"type": "tap", "key": "space"}
            ]}

        prompt = PROMPT_TEMPLATE.replace("{state}", json.dumps(state, ensure_ascii=False))
        output = self._model(prompt=prompt, max_tokens=self.max_tokens, stop=["\n"], temperature=0.4)
        text = output.get("choices", [{}])[0].get("text", "")
        # JSON 파싱 시도
        try:
            # 잘못된 잡음 제거를 위해 첫 { ... } 추출
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1:
                raise ValueError("no json braces")
            candidate = text[start:end+1]
            data = json.loads(candidate)
            # 스키마 검사
            if "actions" not in data or not isinstance(data["actions"], list):
                raise ValueError("invalid schema")
            return data
        except Exception:
            return {"actions": [
                {"type": "tap", "key": "enter"},
                {"type": "key_hold", "key": "up", "ms": 120}
            ]}

    # 자연어 챗 응답
    def chat(self, messages: List[Dict[str, str]], system: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        # messages: [{role: system|user|assistant, content: str}, ...]
        # 최근 대화 요약 텍스트 구성
        history_lines: List[str] = []
        for m in messages[-20:]:
            r = m.get("role", "").lower()
            c = m.get("content", "")
            if r == "user":
                history_lines.append(f"User: {c}")
            elif r == "assistant":
                history_lines.append(f"Assistant: {c}")
            # system은 히스토리에 직접 포함하지 않음(이미 별도 system으로 주입 가능)
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        if not self._model:
            # 더미 응답(간단 규칙 기반): 최근 사용자 질문을 요약, 히스토리 길이에 따라 코칭 포인트 제공
            tips = []
            if "전투" in last_user:
                tips.append("전투 전 HP/MP 점검 후 버프 사용을 고려하세요")
            if "이동" in last_user or "길" in last_user:
                tips.append("불필요한 횡이동 줄이고 목적지까지 직선 루트 탐색")
            if "자원" in last_user or "골드" in last_user:
                tips.append("상점 방문 주기 최적화로 소모품 과잉 구매 피하기")
            if not tips:
                tips.append("상황을 조금 더 구체적으로 말하면 세부 전략을 줄 수 있어요")
            joined = " • ".join(tips)
            return f"요약: {last_user[:60]} | 코칭: {joined}"

        prompt = CHAT_TEMPLATE.format(
            history="\n".join(history_lines),
            user=last_user,
        )
        if system:
            prompt = system.strip() + "\n\n" + prompt
        output = self._model(prompt=prompt, max_tokens=(max_tokens or self.max_tokens), temperature=0.4, stop=["\nUser:"])
        text = output.get("choices", [{}])[0].get("text", "").strip()
        return text
