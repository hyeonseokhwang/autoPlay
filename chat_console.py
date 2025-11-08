# -*- coding: utf-8 -*-
"""
LLM 상담(훈수) 콘솔
- 간단한 터미널 기반 대화 인터페이스
- RAG: 최근/검색 결과를 프롬프트에 주입(초기 단순 구현)
"""
from __future__ import annotations
import argparse
import sys
import os
import sys
# Ensure project root on sys.path when running as a script
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.llm.brain_deepseek import DeepSeekBrain
from src.memory.memory_store import MemoryStore
from src.chat.chat_session import ChatSession

SYSTEM_PROMPT = (
    "당신은 영웅전설4 DOSBox 플레이를 돕는 코치입니다. "
    "사용자 질문에 전략/전투/이동/자원 관리 관점에서 조언을 주세요. "
    "필요하면 과거 대화/이벤트를 참조해 일관성을 유지하세요."
)

def build_context(memory: MemoryStore, query: str, k: int = 5) -> str:
    recent = memory.summarize_recent(8)
    snippets = []
    if query.strip():
        hits = memory.search(query, limit=k)
        for h in hits:
            meta_part = f" meta={h['meta']}" if h.get('meta') else ""
            snippets.append(f"HIT: {h['content']}{meta_part}")
    ctx = (
        "=== 최근 이벤트 ===\n" + recent + "\n\n" +
        ("=== 검색 결과 ===\n" + "\n".join(snippets) if snippets else "")
    )
    return ctx

def main():
    parser = argparse.ArgumentParser(description="영웅전설4 코치 챗 콘솔")
    parser.add_argument("--model", default="models/llm/deepseek-r1-8b-q4_k_m.gguf", help="GGUF 모델 경로")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--session", default="coach")
    parser.add_argument("--mode", choices=["actions", "chat"], default="chat", help="대화 모드 (chat=자연어, actions=JSON 정책)")
    args = parser.parse_args()

    brain = DeepSeekBrain(model_path=args.model, max_tokens=args.max_tokens)
    memory = MemoryStore()
    chat = ChatSession(brain=brain, memory=memory, session_name=args.session, mode=args.mode)
    chat.system(SYSTEM_PROMPT)

    print("[챗 시작] 종료하려면 /exit 입력. 검색은 /s <쿼리>")
    while True:
        try:
            user_in = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            break
        if not user_in:
            continue
        if user_in == "/exit":
            print("[종료]")
            break
        query_override = None
        if user_in.startswith("/s "):
            # 검색 전용
            query_override = user_in[3:].strip()
            print(f"[검색] {query_override}")

        ctx = build_context(memory, query_override or user_in)
        # 컨텍스트를 system으로 임시 주입
        chat.system("컨텍스트:\n" + ctx)
        # 검증 편의를 위해 /s 사용 시 컨텍스트를 화면에도 출력
        if query_override is not None:
            print("\n[컨텍스트 미리보기]\n" + ctx + "\n")
        answer = chat.ask(user_in)
        print("답변>", answer)

    memory.close()

if __name__ == "__main__":
    main()
