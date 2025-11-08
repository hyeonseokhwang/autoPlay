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
    def __init__(self, brain, memory: Optional[MemoryStore] = None, session_name: str = "default", mode: str = "actions"):
        """
        mode: "actions" -> brain.decide() 사용(JSON 액션)
              "chat"    -> brain.chat() 사용(자연어)
        """
        self.brain = brain
        self.memory = memory or MemoryStore()
        self.session_name = session_name
        self.mode = mode
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
        self.user(prompt)
        if self.mode == "chat":
            # RAG 컨텍스트 구성: 최근 이벤트 + 키워드 검색
            recent_summary = self.memory.summarize_recent(6)
            # 간단 키워드 추출(공백 기준, 길이>=2)
            tokens = [t for t in prompt.replace('\n', ' ').split(' ') if len(t.strip()) >= 2][:4]
            hits_blocks = []
            for t in tokens:
                hits = self.memory.search(t, limit=2)
                if hits:
                    # 각 토큰별 첫 hit만 표시
                    first = hits[0]
                    hits_blocks.append(f"[{t}] {first['content'][:70]}")
            rag_context = "최근 요약:\n" + recent_summary
            if hits_blocks:
                rag_context += "\n검색 스니펫:\n" + "\n".join(hits_blocks)
            # 시스템 메시지로 주입(메모리에도 남김)
            self.system("RAG컨텍스트:\n" + rag_context)
            system_texts = [m["content"] for m in self.history if m["role"] == "system"]
            system_joined = "\n\n".join(system_texts[-3:])  # 최근 3개 시스템 메시지만 사용
            answer = self.brain.chat(self.history, system=system_joined)
            self.assistant(answer)
            return answer
        else:
            # 액션 모드: brain.decide
            state = {
                "chat_history": self.history[-20:],
                "goal": "coach player and explain decisions",
            }
            result = self.brain.decide(state)
            if isinstance(result, dict) and "actions" in result:
                text = json.dumps(result, ensure_ascii=False)
            else:
                text = str(result)
            self.assistant(text)
            return text
