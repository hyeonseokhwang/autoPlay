# -*- coding: utf-8 -*-
"""
대화 세션(훈수/설명/학습 질의)
- DeepSeek(또는 동일 인터페이스 LLM)과의 대화를 관리하고 MemoryStore에 로그를 남김.
- 시스템/유저/어시스턴트 역할 메시지 기록.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
from src.memory.memory_store import MemoryStore

class ChatSession:
    def __init__(self, brain, memory: Optional[MemoryStore] = None, session_name: str = "default"):
        self.brain = brain
        self.memory = memory or MemoryStore()
        self.session_name = session_name
        self.history: List[Dict[str, str]] = []  # {role: system|user|assistant, content: str}

    def system(self, text: str):
        self.history.append({"role": "system", "content": text})
        self.memory.add_event("system", text, {"session": self.session_name})

    def user(self, text: str):
        self.history.append({"role": "user", "content": text})
        self.memory.add_event("user", text, {"session": self.session_name})

    def assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self.memory.add_event("assistant", text, {"session": self.session_name})

    def ask(self, prompt: str) -> str:
        # 간단화를 위해 brain.decide를 재사용하지만, 실제로는 별도 chat API를 쓰는 것이 낫다.
        self.user(prompt)
        state = {
            "chat_history": self.history[-20:],
            "goal": "coach player and explain decisions",
        }
        result = self.brain.decide(state)  # JSON actions가 아닌 경우도 있을 수 있으므로 안전 처리
        if isinstance(result, dict) and "actions" in result:
            text = json.dumps(result, ensure_ascii=False)
        else:
            text = str(result)
        self.assistant(text)
        return text
