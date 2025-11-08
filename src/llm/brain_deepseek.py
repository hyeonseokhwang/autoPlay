# -*- coding: utf-8 -*-
"""
DeepSeek 기반 두뇌 스텁
- 실제 deepseek(예: DeepSeek-R1 7B/8B) 모델을 로컬 gguf/llama.cpp로 로드한다고 가정.
- 현재는 의존성 최소화를 위해 import 실패 시 graceful degrade.
"""
from __future__ import annotations
import json
from typing import Dict, Any
from .brain_base import BrainBase

try:
    import llama_cpp  # type: ignore
except ImportError:  # 로컬 환경에 아직 설치 안된 경우
    llama_cpp = None  # 타입 회피용

PROMPT_TEMPLATE = """You are a game automation policy generator.\nReturn ONLY valid JSON with key 'actions'.\nExample:\n{"actions":[{"type":"tap","key":"enter"}]}\nState:\n{state}\nJSON:"""

class DeepSeekBrain(BrainBase):
    def __init__(self, model_path: str = "models/llm/deepseek-r1-8b-q4_k_m.gguf", max_tokens: int = 128):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self._model = None
        if llama_cpp:
            try:
                self._model = llama_cpp.Llama(model_path=model_path, n_ctx=8192, logits_all=False)
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
