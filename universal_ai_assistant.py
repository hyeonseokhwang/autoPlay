"""
🤖 범용 자율학습 AI 비서
게임 플레이 → 인터넷 서핑 → 일반 업무 도우미로 진화하는 AI
"""

import requests
import json
import time
import sqlite3
import asyncio
import aiohttp
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import quote, urljoin
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import schedule
import threading

class UniversalLearningAssistant:
    """범용 자율학습 AI 비서"""
    
    def __init__(self):
        # LLM 설정
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.model = "qwen2.5-coder:7b"
        
        # 임베딩 모델 (GPU 가속)
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            device='cuda' if self.check_gpu_available() else 'cpu'
        )
        
        # 지식 데이터베이스
        self.init_knowledge_database()
        
        # 학습 영역들
        self.learning_domains = {
            "game_strategy": "게임 공략 및 전략",
            "web_research": "인터넷 정보 수집", 
            "task_management": "업무 관리",
            "knowledge_synthesis": "정보 종합 분석",
            "conversation": "대화 및 소통",
            "problem_solving": "문제 해결"
        }
        
        # 자율 학습 스케줄러
        self.learning_schedule = {}
        self.setup_autonomous_learning()
        
        print("🤖 범용 자율학습 AI 비서 초기화 완료")
        print(f"🧠 임베딩 모델: {'GPU' if self.embedding_model.device.type == 'cuda' else 'CPU'} 사용")
    
    def check_gpu_available(self):
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def init_knowledge_database(self):
        """지식 데이터베이스 초기화"""
        self.conn = sqlite3.connect('universal_ai_knowledge.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        # 웹 정보 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                url TEXT,
                title TEXT,
                content TEXT,
                summary TEXT,
                embedding_vector TEXT,
                relevance_score REAL,
                last_updated TIMESTAMP,
                source_type TEXT
            )
        """)
        
        # 대화 컨텍스트 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                ai_response TEXT,
                context_embedding TEXT,
                satisfaction_score REAL,
                timestamp TIMESTAMP,
                domain TEXT
            )
        """)
        
        # 학습 진행도 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                skill_name TEXT,
                proficiency_level REAL,
                learning_count INTEGER,
                last_practice TIMESTAMP,
                next_review TIMESTAMP
            )
        """)
        
        # 작업 기록 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_description TEXT,
                execution_method TEXT,
                success_rate REAL,
                time_taken REAL,
                learned_optimization TEXT,
                timestamp TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    async def autonomous_web_research(self, topic, depth=3):
        """자율적 웹 리서치"""
        print(f"🔍 '{topic}' 주제로 자율 웹 리서치 시작...")
        
        research_results = {
            'topic': topic,
            'sources': [],
            'key_insights': [],
            'related_topics': [],
            'confidence_score': 0.0
        }
        
        # 검색 쿼리 생성 및 다각화
        search_queries = await self.generate_search_queries(topic)
        
        async with aiohttp.ClientSession() as session:
            for query in search_queries[:depth]:
                try:
                    # 다양한 검색 엔진 활용
                    results = await self.search_multiple_sources(session, query)
                    
                    for result in results:
                        # 콘텐츠 분석 및 요약
                        analyzed_content = await self.analyze_web_content(
                            session, result['url'], result['title']
                        )
                        
                        if analyzed_content:
                            research_results['sources'].append(analyzed_content)
                            
                            # 지식 베이스에 저장
                            self.store_web_knowledge(
                                domain=topic,
                                url=result['url'],
                                title=result['title'],
                                content=analyzed_content['content'],
                                summary=analyzed_content['summary']
                            )
                
                except Exception as e:
                    print(f"⚠️ 리서치 오류: {e}")
        
        # 수집된 정보 종합 분석
        research_results['key_insights'] = await self.synthesize_insights(research_results['sources'])
        research_results['related_topics'] = await self.find_related_topics(topic, research_results['sources'])
        research_results['confidence_score'] = self.calculate_research_confidence(research_results)
        
        print(f"✅ 웹 리서치 완료: {len(research_results['sources'])}개 소스 수집")
        return research_results
    
    async def generate_search_queries(self, topic):
        """주제에 대한 다양한 검색 쿼리 생성"""
        
        prompt = f"""
주제: {topic}

이 주제에 대해 포괄적으로 연구하기 위한 다양한 검색 쿼리를 5개 생성해주세요.
각 쿼리는 서로 다른 관점이나 세부 영역을 다뤄야 합니다.

JSON 형태로:
{{
    "queries": [
        "기본 쿼리 1",
        "심화 쿼리 2", 
        "실용적 쿼리 3",
        "전문가 관점 쿼리 4",
        "최신 동향 쿼리 5"
    ]
}}
"""
        
        try:
            response = await self.call_llm_async(prompt)
            parsed = self.parse_llm_json(response)
            return parsed.get('queries', [topic])
        except:
            # 기본 쿼리들
            return [
                topic,
                f"{topic} 가이드",
                f"{topic} 팁",
                f"{topic} 전략",
                f"{topic} 최신"
            ]
    
    async def search_multiple_sources(self, session, query):
        """여러 검색 소스에서 검색"""
        
        search_engines = [
            f"https://www.google.com/search?q={quote(query)}",
            f"https://www.bing.com/search?q={quote(query)}",
            f"https://duckduckgo.com/?q={quote(query)}"
        ]
        
        results = []
        
        for engine_url in search_engines[:1]:  # 일단 구글만
            try:
                # 실제 구현에서는 검색 API 사용 권장
                # 여기서는 간단한 예시
                mock_results = [
                    {
                        'url': f"https://example.com/article1?q={quote(query)}",
                        'title': f"{query} 관련 정보 1",
                        'snippet': f"{query}에 대한 기본 정보"
                    },
                    {
                        'url': f"https://example.com/article2?q={quote(query)}",
                        'title': f"{query} 상세 가이드",
                        'snippet': f"{query} 심화 학습 자료"
                    }
                ]
                results.extend(mock_results)
                
            except Exception as e:
                print(f"검색 오류: {e}")
        
        return results[:5]  # 상위 5개만
    
    async def analyze_web_content(self, session, url, title):
        """웹 콘텐츠 분석 및 요약"""
        
        try:
            # 실제 구현에서는 웹페이지 크롤링
            # 여기서는 모의 데이터
            mock_content = f"""
            {title}에 대한 상세한 내용입니다.
            
            주요 포인트:
            1. 기본 개념과 정의
            2. 실용적인 활용 방법
            3. 전문가 팁과 조언
            4. 최신 동향 및 업데이트
            
            이 정보는 {url}에서 수집되었습니다.
            """
            
            # LLM으로 요약 생성
            summary_prompt = f"""
다음 웹 콘텐츠를 간결하게 요약해주세요:

제목: {title}
URL: {url}
내용: {mock_content}

핵심 정보만 3-4줄로 요약:
"""
            
            summary = await self.call_llm_async(summary_prompt)
            
            return {
                'url': url,
                'title': title,
                'content': mock_content,
                'summary': summary.strip(),
                'word_count': len(mock_content.split()),
                'relevance': 0.8  # 관련성 점수
            }
            
        except Exception as e:
            print(f"콘텐츠 분석 오류: {e}")
            return None
    
    def store_web_knowledge(self, domain, url, title, content, summary):
        """웹 지식을 데이터베이스에 저장"""
        
        # 임베딩 생성
        combined_text = f"{title} {summary} {content}"
        embedding = self.embedding_model.encode(combined_text)
        embedding_json = json.dumps(embedding.tolist())
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO web_knowledge 
            (domain, url, title, content, summary, embedding_vector, 
             relevance_score, last_updated, source_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            domain, url, title, content, summary, embedding_json,
            0.8, datetime.now().isoformat(), 'web_research'
        ))
        
        self.conn.commit()
    
    async def synthesize_insights(self, sources):
        """수집된 소스들로부터 핵심 인사이트 추출"""
        
        if not sources:
            return []
        
        combined_summaries = "\n\n".join([s['summary'] for s in sources])
        
        synthesis_prompt = f"""
다음 정보들을 종합하여 핵심 인사이트 5개를 추출해주세요:

{combined_summaries}

JSON 형태로:
{{
    "insights": [
        "핵심 인사이트 1",
        "핵심 인사이트 2",
        "핵심 인사이트 3",
        "핵심 인사이트 4", 
        "핵심 인사이트 5"
    ]
}}
"""
        
        try:
            response = await self.call_llm_async(synthesis_prompt)
            parsed = self.parse_llm_json(response)
            return parsed.get('insights', [])
        except:
            return ["정보 종합 중 오류 발생"]
    
    async def find_related_topics(self, main_topic, sources):
        """관련 주제 발견"""
        
        content_text = " ".join([s['content'] for s in sources])
        
        related_prompt = f"""
주요 주제: {main_topic}

다음 내용에서 관련된 하위 주제나 연관 주제들을 찾아주세요:

{content_text[:1000]}...

JSON 형태로:
{{
    "related_topics": [
        "관련 주제 1",
        "관련 주제 2",
        "관련 주제 3"
    ]
}}
"""
        
        try:
            response = await self.call_llm_async(related_prompt)
            parsed = self.parse_llm_json(response)
            return parsed.get('related_topics', [])
        except:
            return []
    
    def calculate_research_confidence(self, research_results):
        """리서치 결과의 신뢰도 계산"""
        
        source_count = len(research_results['sources'])
        avg_relevance = np.mean([s.get('relevance', 0.5) for s in research_results['sources']])
        insight_count = len(research_results['key_insights'])
        
        confidence = (
            min(source_count / 5, 1.0) * 0.4 +  # 소스 다양성
            avg_relevance * 0.4 +               # 관련성
            min(insight_count / 5, 1.0) * 0.2   # 인사이트 품질
        )
        
        return round(confidence, 2)
    
    async def intelligent_conversation(self, user_input, context_history=None):
        """지능적 대화 처리"""
        
        print(f"💬 사용자: {user_input}")
        
        # 사용자 입력 의도 분석
        intent = await self.analyze_user_intent(user_input)
        
        # 관련 지식 검색
        relevant_knowledge = self.search_relevant_knowledge(user_input)
        
        # 컨텍스트 구성
        context = {
            'user_input': user_input,
            'intent': intent,
            'relevant_knowledge': relevant_knowledge,
            'conversation_history': context_history or []
        }
        
        # AI 응답 생성
        response = await self.generate_contextual_response(context)
        
        # 대화 기록 저장
        self.store_conversation(user_input, response, intent['domain'])
        
        print(f"🤖 AI: {response}")
        return response
    
    async def analyze_user_intent(self, user_input):
        """사용자 입력 의도 분석"""
        
        intent_prompt = f"""
사용자 입력을 분석하여 의도를 파악해주세요:

입력: "{user_input}"

JSON 형태로:
{{
    "intent_type": "question/request/conversation/task",
    "domain": "game_strategy/web_research/task_management/general",
    "urgency": "low/medium/high",
    "requires_research": true/false,
    "specific_action": "구체적인 행동 (있다면)"
}}
"""
        
        try:
            response = await self.call_llm_async(intent_prompt)
            return self.parse_llm_json(response)
        except:
            return {
                "intent_type": "conversation",
                "domain": "general", 
                "urgency": "medium",
                "requires_research": False
            }
    
    def search_relevant_knowledge(self, query, top_k=3):
        """관련 지식 검색 (벡터 유사도 기반)"""
        
        query_embedding = self.embedding_model.encode(query)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT title, summary, content, relevance_score, source_type 
            FROM web_knowledge 
            ORDER BY last_updated DESC LIMIT 50
        """)
        
        knowledge_items = cursor.fetchall()
        
        if not knowledge_items:
            return []
        
        # 유사도 계산
        similarities = []
        for item in knowledge_items:
            try:
                # 실제 구현에서는 저장된 임베딩 사용
                item_text = f"{item[0]} {item[1]}"
                item_embedding = self.embedding_model.encode(item_text)
                
                similarity = cosine_similarity([query_embedding], [item_embedding])[0][0]
                
                similarities.append({
                    'title': item[0],
                    'summary': item[1], 
                    'content': item[2][:500],  # 첫 500자만
                    'similarity': similarity,
                    'source_type': item[4]
                })
            except:
                continue
        
        # 유사도 순 정렬 후 상위 k개 반환
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    async def generate_contextual_response(self, context):
        """컨텍스트 기반 응답 생성"""
        
        user_input = context['user_input']
        intent = context['intent']
        relevant_knowledge = context['relevant_knowledge']
        
        # 지식 컨텍스트 구성
        knowledge_context = ""
        if relevant_knowledge:
            knowledge_context = "\n참고 정보:\n"
            for i, item in enumerate(relevant_knowledge, 1):
                knowledge_context += f"{i}. {item['title']}: {item['summary']}\n"
        
        response_prompt = f"""
당신은 지능적인 AI 비서입니다.

사용자 입력: {user_input}
의도 분석: {json.dumps(intent, ensure_ascii=False)}

{knowledge_context}

위 정보를 바탕으로 도움이 되는 응답을 생성해주세요.
- 구체적이고 실용적인 답변
- 필요시 추가 질문 제안
- 친근하고 전문적인 톤

응답:
"""
        
        try:
            response = await self.call_llm_async(response_prompt)
            
            # 추가 연구가 필요한 경우 자동 트리거
            if intent.get('requires_research') and intent.get('domain') != 'general':
                research_topic = self.extract_research_topic(user_input)
                if research_topic:
                    print(f"🔍 '{research_topic}'에 대한 자동 리서치 시작...")
                    asyncio.create_task(self.autonomous_web_research(research_topic))
            
            return response.strip()
            
        except Exception as e:
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {e}"
    
    def extract_research_topic(self, user_input):
        """사용자 입력에서 리서치 주제 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 필요)
        keywords = re.findall(r'\b[가-힣]{2,}\b', user_input)
        return " ".join(keywords[:3]) if keywords else None
    
    def store_conversation(self, user_input, ai_response, domain):
        """대화 기록 저장"""
        
        context_text = f"User: {user_input} AI: {ai_response}"
        context_embedding = self.embedding_model.encode(context_text)
        embedding_json = json.dumps(context_embedding.tolist())
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conversation_history 
            (user_input, ai_response, context_embedding, satisfaction_score, timestamp, domain)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_input, ai_response, embedding_json, 
            0.8, datetime.now().isoformat(), domain
        ))
        
        self.conn.commit()
    
    def setup_autonomous_learning(self):
        """자율 학습 스케줄 설정"""
        
        # 매일 자율 학습 세션
        schedule.every().day.at("09:00").do(self.daily_learning_session)
        schedule.every().day.at("18:00").do(self.evening_research_session)
        
        # 주기적 지식 정리
        schedule.every().week.do(self.weekly_knowledge_synthesis)
        
        # 스케줄러 백그라운드 실행
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        print("📅 자율 학습 스케줄 설정 완료")
    
    async def daily_learning_session(self):
        """일일 자율 학습 세션"""
        print("📚 일일 자율 학습 세션 시작...")
        
        # 최근 대화에서 자주 언급된 주제 분석
        trending_topics = self.analyze_trending_topics()
        
        for topic in trending_topics[:2]:  # 상위 2개 주제
            await self.autonomous_web_research(topic, depth=2)
        
        print("✅ 일일 학습 완료")
    
    async def evening_research_session(self):
        """저녁 리서치 세션"""
        print("🌙 저녁 심화 리서치 세션...")
        
        # 지식 격차 분석 및 보완
        knowledge_gaps = self.identify_knowledge_gaps()
        
        for gap in knowledge_gaps[:1]:  # 가장 큰 격차 1개
            await self.autonomous_web_research(gap, depth=3)
    
    def analyze_trending_topics(self):
        """최근 대화에서 트렌딩 주제 분석"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT user_input FROM conversation_history 
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC LIMIT 20
        """)
        
        recent_inputs = [row[0] for row in cursor.fetchall()]
        
        # 간단한 키워드 빈도 분석
        all_text = " ".join(recent_inputs)
        keywords = re.findall(r'\b[가-힣]{2,}\b', all_text)
        
        from collections import Counter
        keyword_counts = Counter(keywords)
        
        return [word for word, count in keyword_counts.most_common(5)]
    
    def identify_knowledge_gaps(self):
        """지식 격차 식별"""
        # 실제로는 더 정교한 분석 필요
        potential_gaps = [
            "최신 기술 동향",
            "업무 효율성 향상",
            "창작 및 글쓰기",
            "데이터 분석 방법"
        ]
        
        return potential_gaps[:2]
    
    async def weekly_knowledge_synthesis(self):
        """주간 지식 종합"""
        print("📊 주간 지식 종합 작업...")
        
        # 수집된 지식 통합 및 정리
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT domain, COUNT(*) as knowledge_count
            FROM web_knowledge 
            WHERE last_updated > datetime('now', '-7 days')
            GROUP BY domain
        """)
        
        weekly_stats = cursor.fetchall()
        
        print("주간 학습 통계:")
        for domain, count in weekly_stats:
            print(f"  - {domain}: {count}개 지식 항목 수집")
    
    async def call_llm_async(self, prompt, timeout=15):
        """비동기 LLM 호출"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        # aiohttp 사용 시 비동기 처리
        async with aiohttp.ClientSession() as session:
            async with session.post(self.llm_endpoint, json=payload, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "")
                else:
                    raise Exception(f"LLM API 오류: {response.status}")
    
    def parse_llm_json(self, text):
        """LLM 응답 JSON 파싱"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start != -1 and end > start:
                json_text = text[start:end]
                return json.loads(json_text)
            return {}
        except:
            return {}
    
    async def run_interactive_session(self):
        """대화형 세션 실행"""
        print("\n🤖 범용 AI 비서와의 대화를 시작합니다!")
        print("💡 '웹서치 [주제]'로 자동 리서치 요청 가능")
        print("💡 'exit'로 종료")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'exit':
                    print("👋 대화를 종료합니다. 즐거웠어요!")
                    break
                
                if user_input.startswith('웹서치 '):
                    # 수동 웹 리서치 트리거
                    topic = user_input[4:].strip()
                    research_result = await self.autonomous_web_research(topic)
                    
                    summary = f"'{topic}' 리서치 완료!\n"
                    summary += f"📚 {len(research_result['sources'])}개 소스 수집\n"
                    summary += f"💡 핵심 인사이트: {len(research_result['key_insights'])}개\n"
                    summary += f"🎯 신뢰도: {research_result['confidence_score']}\n"
                    
                    if research_result['key_insights']:
                        summary += "\n주요 발견사항:\n"
                        for i, insight in enumerate(research_result['key_insights'][:3], 1):
                            summary += f"{i}. {insight}\n"
                    
                    print(f"\n🤖 AI: {summary}")
                else:
                    # 일반 대화
                    response = await self.intelligent_conversation(user_input, conversation_history)
                    conversation_history.append({
                        'user': user_input,
                        'ai': response,
                        'timestamp': datetime.now()
                    })
                    
                    # 대화 히스토리 제한 (메모리 관리)
                    if len(conversation_history) > 10:
                        conversation_history = conversation_history[-10:]
                
            except KeyboardInterrupt:
                print("\n\n👋 Ctrl+C로 종료되었습니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")


async def main():
    """메인 실행 함수"""
    
    print("🚀 범용 자율학습 AI 비서")
    print("🧠 게임 AI → 웹 리서치 → 개인 비서로 진화")
    print("🔥 GPU 가속 벡터 검색 + 자율 학습")
    print()
    
    # AI 초기화
    assistant = UniversalLearningAssistant()
    
    print("\n메뉴:")
    print("1. 대화형 비서 모드")
    print("2. 자율 웹 리서치 테스트")
    print("3. 지식 현황 확인")
    
    choice = input("\n선택하세요 (1-3): ").strip()
    
    if choice == "1":
        await assistant.run_interactive_session()
        
    elif choice == "2":
        topic = input("리서치 주제를 입력하세요: ").strip()
        if topic:
            result = await assistant.autonomous_web_research(topic)
            print(f"\n📊 리서치 결과:")
            print(f"   소스: {len(result['sources'])}개")
            print(f"   인사이트: {len(result['key_insights'])}개")
            print(f"   신뢰도: {result['confidence_score']}")
    
    elif choice == "3":
        cursor = assistant.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM web_knowledge")
        knowledge_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM conversation_history")
        conversation_count = cursor.fetchone()[0]
        
        print(f"\n📚 현재 지식 현황:")
        print(f"   수집된 웹 지식: {knowledge_count}개")
        print(f"   대화 기록: {conversation_count}개")
    
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    asyncio.run(main())