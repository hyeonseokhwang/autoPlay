# -*- coding: utf-8 -*-
"""
챗/RAG 메모리 자동 테스트 스크립트
1) 고유 토큰 포함 사용자 질문을 남김
2) 답변 생성(더미 또는 실제 모델)
3) 새 세션을 열어 검색(/s 기능 없이 직접 메서드로)으로 토큰 HIT 확인
"""
from __future__ import annotations
import uuid
from src.llm.brain_deepseek import DeepSeekBrain
from src.memory.memory_store import MemoryStore
from src.chat.chat_session import ChatSession


def run_test():
    token = "TESTTOKEN-" + uuid.uuid4().hex[:8]
    print(f"[테스트] 고유 토큰: {token}")

    brain = DeepSeekBrain()  # 더미 또는 실제 모델
    memory = MemoryStore()
    chat = ChatSession(brain=brain, memory=memory, session_name="coach")
    chat.system("시스템: 챗/RAG 메모리 테스트")

    user_question = f"이 토큰을 기억해줘: {token} 그리고 전투 이동 전략은?"
    print("[사용자 입력]", user_question)
    answer = chat.ask(user_question)
    print("[모델 답변]", answer)

    # 세션 종료 후 새 세션 열기 (메모리 지속성 확인)
    memory.close()
    memory2 = MemoryStore()  # 같은 DB 파일 재열기
    hits = memory2.search(token, limit=5)
    print(f"[검색] '{token}' 결과 개수: {len(hits)}")
    for h in hits:
        print("  HIT:", h["content"])
    assert hits, "토큰 검색 실패: 메모리 저장 문제"  # 없으면 실패
    print("[성공] 토큰이 메모리에 유지됨 → RAG 기본 기능 OK")
    memory2.close()

if __name__ == "__main__":
    run_test()
