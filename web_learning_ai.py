"""
🌐 자율 정보 수집 및 학습 영웅전설4 AI
스스로 인터넷에서 게임 정보를 찾아 학습하는 시스템
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin, quote
from datetime import datetime
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class AutoLearningWebCrawler:
    """자율 웹 크롤링 및 정보 학습"""
    
    def __init__(self):
        self.session = None
        self.knowledge_db = sqlite3.connect("auto_learned_knowledge.db")
        self.embedding_model = None
        self.search_queries = []
        
        # 영웅전설4 관련 검색 키워드
        self.base_keywords = [
            "영웅전설4", "Legend of Heroes 4", "가가브 트릴로지",
            "DOSBox", "방향키", "키보드 조작", "게임 조작법",
            "전투 시스템", "캐릭터 이동", "RPG 조작"
        ]
        
        # 자동 발견할 키워드들
        self.discovered_keywords = set()
        
        self.init_knowledge_db()
        self.load_embedding_model()
        
        print("🌐 자율 학습 웹 크롤러 초기화 완료")
    
    def init_knowledge_db(self):
        """지식 데이터베이스 초기화"""
        cursor = self.knowledge_db.cursor()
        
        # 웹 지식 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_url TEXT,
                title TEXT,
                content TEXT,
                keywords TEXT,
                relevance_score REAL,
                discovered_at TEXT,
                embedding_vector TEXT,
                knowledge_type TEXT,
                verified BOOLEAN DEFAULT FALSE
            )
        """)
        
        # 발견된 패턴 테이블  
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT UNIQUE,
                pattern_description TEXT,
                source_evidence TEXT,
                confidence_score REAL,
                usage_success_rate REAL DEFAULT 0.0,
                times_tested INTEGER DEFAULT 0,
                discovered_at TEXT
            )
        """)
        
        self.knowledge_db.commit()
        print("📚 자율 학습 데이터베이스 준비 완료")
    
    def load_embedding_model(self):
        """임베딩 모델 로드"""
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print("🔤 다국어 임베딩 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ 임베딩 모델 로드 실패: {e}")
    
    async def autonomous_search(self, current_game_context=""):
        """현재 게임 상황에 맞는 자율 검색"""
        
        # 현재 상황 분석해서 검색 키워드 생성
        search_queries = self.generate_contextual_queries(current_game_context)
        
        print(f"🔍 자율 검색 시작: {len(search_queries)}개 쿼리")
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            for query in search_queries:
                print(f"   검색: {query}")
                
                try:
                    # 다양한 소스에서 검색
                    await self.search_multiple_sources(query)
                    await asyncio.sleep(1)  # 서버 부하 방지
                    
                except Exception as e:
                    print(f"   ⚠️ 검색 실패: {e}")
        
        # 검색 결과 분석 및 학습
        self.analyze_and_learn()
    
    def generate_contextual_queries(self, game_context):
        """게임 상황에 맞는 검색어 생성"""
        
        base_queries = [
            "영웅전설4 조작법",
            "영웅전설4 키보드 사용법", 
            "Legend of Heroes 4 controls",
            "DOSBox RPG 게임 조작",
            "영웅전설4 공략"
        ]
        
        # 현재 상황에 따른 동적 쿼리 생성
        if "전투" in game_context or "battle" in game_context.lower():
            base_queries.extend([
                "영웅전설4 전투 시스템",
                "영웅전설4 공격 방법",
                "턴제 RPG 전투 조작법"
            ])
        
        if "이동" in game_context or "movement" in game_context.lower():
            base_queries.extend([
                "영웅전설4 캐릭터 이동",
                "방향키 사용법",
                "RPG 게임 탐험 방법"
            ])
        
        if "메뉴" in game_context or "UI" in game_context:
            base_queries.extend([
                "영웅전설4 메뉴 사용법",
                "게임 인벤토리 조작",
                "RPG 상태창 보는법"
            ])
        
        return base_queries[:10]  # 최대 10개로 제한
    
    async def search_multiple_sources(self, query):
        """다양한 소스에서 검색"""
        
        sources = [
            self.search_namu_wiki,
            self.search_google_snippets,
            self.search_game_community,
        ]
        
        tasks = []
        for source_func in sources:
            task = asyncio.create_task(source_func(query))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 저장
        for result in results:
            if isinstance(result, Exception):
                continue
            if result:
                self.store_web_knowledge(result, query)
    
    async def search_namu_wiki(self, query):
        """나무위키에서 검색"""
        try:
            search_url = f"https://namu.wiki/w/{quote(query)}"
            
            async with self.session.get(search_url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 본문 내용 추출
                    content_div = soup.find('div', class_='wiki-content')
                    if content_div:
                        text = content_div.get_text(strip=True)
                        
                        return {
                            'source': '나무위키',
                            'url': search_url,
                            'title': query,
                            'content': text[:2000],  # 처음 2000자만
                            'type': 'wiki'
                        }
        except Exception as e:
            print(f"나무위키 검색 실패: {e}")
        
        return None
    
    async def search_google_snippets(self, query):
        """구글 검색 스니펫 수집"""
        try:
            # DuckDuckGo API 사용 (구글 대신)
            search_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
            
            async with self.session.get(search_url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Abstract나 Answer에서 정보 추출
                    content = ""
                    if data.get('Abstract'):
                        content += data['Abstract']
                    if data.get('Answer'):
                        content += " " + data['Answer']
                    
                    if content:
                        return {
                            'source': 'DuckDuckGo',
                            'url': search_url,
                            'title': query,
                            'content': content,
                            'type': 'search_snippet'
                        }
        except Exception as e:
            print(f"검색 스니펫 수집 실패: {e}")
        
        return None
    
    async def search_game_community(self, query):
        """게임 커뮤니티 검색 (시뮬레이션)"""
        # 실제로는 루리웹, 디시인사이드 등에서 검색하지만
        # 여기서는 시뮬레이션된 결과 반환
        
        game_tips = {
            "방향키": "상하좌우 방향키로 캐릭터 이동. Enter키로 확인, ESC키로 취소",
            "전투": "전투 시 숫자키로 공격 선택, 방향키로 대상 선택",
            "메뉴": "Alt키로 메뉴 호출, Tab키로 상태창 확인",
            "이동": "필드에서 방향키로 8방향 이동 가능",
            "조작": "기본적으로 키보드만 사용하며, 마우스는 지원하지 않음"
        }
        
        # 키워드 매칭으로 관련 팁 찾기
        relevant_tips = []
        for keyword, tip in game_tips.items():
            if keyword in query or any(k in query for k in keyword):
                relevant_tips.append(tip)
        
        if relevant_tips:
            return {
                'source': '게임 커뮤니티',
                'url': 'simulated',
                'title': f'{query} 관련 팁',
                'content': " ".join(relevant_tips),
                'type': 'community_tip'
            }
        
        return None
    
    def store_web_knowledge(self, knowledge_data, original_query):
        """웹에서 수집한 지식 저장"""
        if not knowledge_data:
            return
        
        # 관련성 점수 계산
        relevance_score = self.calculate_relevance(knowledge_data['content'], original_query)
        
        # 임베딩 벡터 생성
        embedding_vector = None
        if self.embedding_model:
            embedding = self.embedding_model.encode(knowledge_data['content'])
            embedding_vector = json.dumps(embedding.tolist())
        
        # 키워드 추출
        keywords = self.extract_keywords(knowledge_data['content'])
        
        # 데이터베이스에 저장
        cursor = self.knowledge_db.cursor()
        cursor.execute("""
            INSERT INTO web_knowledge 
            (source_url, title, content, keywords, relevance_score, 
             discovered_at, embedding_vector, knowledge_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            knowledge_data['url'],
            knowledge_data['title'], 
            knowledge_data['content'],
            json.dumps(keywords),
            relevance_score,
            datetime.now().isoformat(),
            embedding_vector,
            knowledge_data['type']
        ))
        
        self.knowledge_db.commit()
        print(f"💾 지식 저장: {knowledge_data['title']} (관련성: {relevance_score:.2f})")
    
    def calculate_relevance(self, content, query):
        """내용과 쿼리의 관련성 점수 계산"""
        
        # 키워드 매칭 기반 간단한 관련성 계산
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # 공통 단어 비율
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        # 게임 관련 키워드 보너스
        game_keywords = ['영웅전설', '방향키', '키보드', '조작', 'rpg', '게임']
        bonus = sum(1 for keyword in game_keywords if keyword in content.lower()) * 0.1
        
        return min(1.0, jaccard_similarity + bonus)
    
    def extract_keywords(self, content):
        """내용에서 키워드 추출"""
        
        # 게임 조작 관련 키워드 패턴
        control_patterns = [
            r'방향키',
            r'Enter|엔터',
            r'ESC|Escape|취소',
            r'Alt|알트',
            r'Tab|탭',
            r'Shift|시프트',
            r'Ctrl|컨트롤',
            r'스페이스|Space',
            r'숫자키',
            r'키보드',
            r'마우스'
        ]
        
        # 게임 용어 패턴
        game_patterns = [
            r'전투|배틀',
            r'이동|움직임',
            r'공격|어택',
            r'방어|디펜스', 
            r'메뉴|인벤토리',
            r'상태창|스테이터스',
            r'캐릭터|주인공',
            r'적|몬스터|enemy',
            r'레벨|경험치',
            r'아이템|장비'
        ]
        
        found_keywords = []
        
        # 패턴 매칭
        for patterns in [control_patterns, game_patterns]:
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                found_keywords.extend(matches)
        
        # 중복 제거 및 정리
        unique_keywords = list(set(found_keywords))
        
        return unique_keywords[:20]  # 최대 20개
    
    def analyze_and_learn(self):
        """수집된 정보 분석 및 패턴 학습"""
        
        print("🧠 수집된 정보 분석 중...")
        
        cursor = self.knowledge_db.cursor()
        cursor.execute("""
            SELECT content, keywords, relevance_score 
            FROM web_knowledge 
            WHERE relevance_score > 0.3 
            ORDER BY discovered_at DESC 
            LIMIT 50
        """)
        
        recent_knowledge = cursor.fetchall()
        
        if not recent_knowledge:
            print("❌ 분석할 지식이 부족합니다")
            return
        
        # 키보드 조작 패턴 추출
        control_patterns = self.extract_control_patterns(recent_knowledge)
        
        # 게임 플레이 팁 추출  
        gameplay_tips = self.extract_gameplay_tips(recent_knowledge)
        
        # 학습된 패턴 저장
        for pattern_name, pattern_data in control_patterns.items():
            self.store_discovered_pattern(pattern_name, pattern_data, "keyboard_control")
        
        for tip_name, tip_data in gameplay_tips.items():
            self.store_discovered_pattern(tip_name, tip_data, "gameplay_strategy")
        
        print(f"💡 발견된 패턴: {len(control_patterns) + len(gameplay_tips)}개")
    
    def extract_control_patterns(self, knowledge_list):
        """키보드 조작 패턴 추출"""
        
        patterns = {}
        
        for content, keywords_json, relevance in knowledge_list:
            try:
                keywords = json.loads(keywords_json)
            except:
                keywords = []
            
            # 방향키 패턴
            if any('방향키' in str(k) for k in keywords):
                patterns['direction_keys'] = {
                    'description': '방향키로 캐릭터 이동 제어',
                    'keys': ['UP', 'DOWN', 'LEFT', 'RIGHT'],
                    'confidence': min(1.0, relevance * 1.2)
                }
            
            # 확인/취소 패턴
            if any(k in str(keywords).lower() for k in ['enter', '엔터', 'esc', 'escape']):
                patterns['confirm_cancel'] = {
                    'description': 'Enter로 확인, ESC로 취소',
                    'keys': ['ENTER', 'ESCAPE'], 
                    'confidence': min(1.0, relevance * 1.1)
                }
            
            # 메뉴 패턴
            if any(k in str(keywords).lower() for k in ['alt', '알트', 'tab', '탭']):
                patterns['menu_access'] = {
                    'description': 'Alt로 메뉴, Tab으로 상태창',
                    'keys': ['ALT', 'TAB'],
                    'confidence': min(1.0, relevance)
                }
        
        return patterns
    
    def extract_gameplay_tips(self, knowledge_list):
        """게임플레이 팁 추출"""
        
        tips = {}
        
        for content, keywords_json, relevance in knowledge_list:
            content_lower = content.lower()
            
            # 전투 관련 팁
            if '전투' in content_lower or 'battle' in content_lower:
                tips['battle_strategy'] = {
                    'description': '전투 시 전략적 행동 필요',
                    'evidence': content[:200],
                    'confidence': relevance
                }
            
            # 탐험 관련 팁  
            if '탐험' in content_lower or '이동' in content_lower:
                tips['exploration_strategy'] = {
                    'description': '체계적 탐험으로 효율성 증대',
                    'evidence': content[:200], 
                    'confidence': relevance
                }
        
        return tips
    
    def store_discovered_pattern(self, pattern_name, pattern_data, pattern_type):
        """발견된 패턴 저장"""
        
        cursor = self.knowledge_db.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO discovered_patterns
            (pattern_name, pattern_description, source_evidence, 
             confidence_score, discovered_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            f"{pattern_type}_{pattern_name}",
            pattern_data['description'],
            json.dumps(pattern_data),
            pattern_data['confidence'],
            datetime.now().isoformat()
        ))
        
        self.knowledge_db.commit()
    
    def get_relevant_knowledge(self, current_situation):
        """현재 상황에 맞는 학습된 지식 검색"""
        
        cursor = self.knowledge_db.cursor()
        cursor.execute("""
            SELECT pattern_name, pattern_description, source_evidence, confidence_score
            FROM discovered_patterns
            WHERE confidence_score > 0.5
            ORDER BY confidence_score DESC
        """)
        
        patterns = cursor.fetchall()
        
        # 현재 상황과 관련성 있는 패턴 필터링
        relevant_patterns = []
        situation_lower = current_situation.lower()
        
        for name, description, evidence, confidence in patterns:
            # 키워드 매칭으로 관련성 판단
            if any(keyword in situation_lower for keyword in 
                   ['이동', '전투', '메뉴', 'move', 'battle', 'menu']):
                
                if any(keyword in name.lower() for keyword in
                       ['direction', 'confirm', 'menu', 'battle', 'exploration']):
                    
                    relevant_patterns.append({
                        'name': name,
                        'description': description,
                        'confidence': confidence,
                        'evidence': evidence
                    })
        
        return relevant_patterns[:5]  # 상위 5개만
    
    def get_learning_summary(self):
        """학습 현황 요약"""
        
        cursor = self.knowledge_db.cursor()
        
        # 수집된 지식 통계
        cursor.execute("SELECT COUNT(*) FROM web_knowledge")
        total_knowledge = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM discovered_patterns") 
        total_patterns = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence_score) FROM discovered_patterns")
        avg_confidence = cursor.fetchone()[0] or 0
        
        return {
            'total_web_knowledge': total_knowledge,
            'discovered_patterns': total_patterns,
            'average_confidence': avg_confidence,
            'last_learning': datetime.now().isoformat()
        }


class WebEnhancedGameAI:
    """웹 학습 강화 게임 AI"""
    
    def __init__(self):
        from isolated_seeker import IsolatedDOSBoxSeeker
        
        self.base_seeker = IsolatedDOSBoxSeeker()
        self.web_crawler = AutoLearningWebCrawler()
        
        # LLM 설정
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.model = "qwen2.5-coder:7b"
        
        # 학습 상태
        self.learning_phase = "initial"  # initial -> informed -> expert
        self.web_learning_completed = False
        
        print("🌐 웹 학습 강화 게임 AI 초기화 완료")
    
    async def initial_web_learning(self):
        """초기 웹 학습 단계"""
        
        print("🔍 게임에 대한 초기 웹 학습 시작...")
        
        # 기본 게임 정보 수집
        await self.web_crawler.autonomous_search("영웅전설4 기본 조작법")
        
        # 수집된 지식 요약
        summary = self.web_crawler.get_learning_summary()
        print(f"📚 초기 학습 완료: {summary['total_web_knowledge']}개 지식, {summary['discovered_patterns']}개 패턴")
        
        self.web_learning_completed = True
        self.learning_phase = "informed"
    
    def get_web_informed_decision(self, current_situation):
        """웹에서 학습한 지식 기반 결정"""
        
        # 관련 패턴 검색
        relevant_knowledge = self.web_crawler.get_relevant_knowledge(current_situation)
        
        if not relevant_knowledge:
            return self.fallback_decision()
        
        # LLM에게 웹 지식과 함께 분석 요청
        web_informed_prompt = f"""
당신은 웹에서 학습한 지식을 바탕으로 게임을 플레이하는 AI입니다.

현재 상황: {current_situation}

웹에서 학습한 관련 지식:
{self.format_web_knowledge(relevant_knowledge)}

이 지식을 바탕으로 다음 행동을 결정하세요 (0-6):
0: 왼쪽 이동
1: 오른쪽 이동  
2: 위쪽 이동
3: 아래쪽 이동
4: 확인/공격 (Enter)
5: 취소/메뉴 (ESC)
6: 대기

JSON 형태로:
{{
    "action": 1,
    "reasoning": "웹에서 학습한 지식 기반 판단",
    "web_knowledge_used": "사용된 웹 지식 설명",
    "confidence": 0.8
}}
"""
        
        try:
            response = self.call_llm(web_informed_prompt)
            decision = self.parse_llm_json(response)
            
            if decision and 'action' in decision:
                print(f"🌐 웹 지식 기반 결정: {decision['reasoning']}")
                return decision
            
        except Exception as e:
            print(f"⚠️ 웹 지식 기반 분석 실패: {e}")
        
        return self.fallback_decision()
    
    def format_web_knowledge(self, knowledge_list):
        """웹 지식을 LLM이 이해하기 쉽게 포맷팅"""
        
        if not knowledge_list:
            return "관련된 웹 지식이 없습니다."
        
        formatted = []
        for i, knowledge in enumerate(knowledge_list, 1):
            formatted.append(f"""
지식 {i} (신뢰도: {knowledge['confidence']:.2f}):
- 패턴: {knowledge['name']}
- 설명: {knowledge['description']}
""")
        
        return "\n".join(formatted)
    
    async def adaptive_web_learning(self, current_context):
        """상황에 맞는 적응적 웹 학습"""
        
        # 현재 상황에서 부족한 지식이 있으면 추가 학습
        relevant_knowledge = self.web_crawler.get_relevant_knowledge(current_context)
        
        if len(relevant_knowledge) < 2:  # 관련 지식이 부족하면
            print(f"🔍 추가 웹 학습 필요: {current_context}")
            await self.web_crawler.autonomous_search(current_context)
    
    def call_llm(self, prompt, timeout=10):
        """LLM 호출"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.9}
        }
        
        response = requests.post(self.llm_endpoint, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"LLM API 오류: {response.status_code}")
    
    def parse_llm_json(self, text):
        """LLM 응답 JSON 파싱"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                json_text = text[start:end]
                return json.loads(json_text)
        except:
            pass
        return {}
    
    def fallback_decision(self):
        """폴백 결정"""
        return {
            "action": np.random.randint(0, 7),
            "reasoning": "웹 지식 부족으로 랜덤 탐험",
            "confidence": 0.3
        }
    
    def execute_action(self, action_id):
        """행동 실행"""
        vk_keys = {
            0: self.base_seeker.VK_LEFT,
            1: self.base_seeker.VK_RIGHT,
            2: self.base_seeker.VK_UP,
            3: self.base_seeker.VK_DOWN,
            4: self.base_seeker.VK_RETURN,
            5: self.base_seeker.VK_ESCAPE,
            6: None
        }
        
        vk_code = vk_keys.get(action_id)
        if vk_code is not None:
            return self.base_seeker.send_key_message(vk_code)
        return True
    
    async def web_enhanced_play(self, max_iterations=100):
        """웹 학습 강화 자율 플레이"""
        
        print("🌐 웹 학습 강화 자율 플레이 시작!")
        
        if not self.base_seeker.find_dosbox_window():
            print("❌ DOSBox 창을 찾을 수 없습니다!")
            return
        
        # 초기 웹 학습
        if not self.web_learning_completed:
            await self.initial_web_learning()
        
        iteration = 0
        
        while iteration < max_iterations:
            try:
                print(f"\n--- 웹 강화 플레이 #{iteration + 1} ---")
                
                # 현재 화면 캡처 및 분석
                current_screen = self.base_seeker.capture_dosbox_window()
                if current_screen is None:
                    continue
                
                # 게임 상황 분석
                is_battle = self.base_seeker.is_battle_screen(current_screen)
                current_situation = "전투 상황" if is_battle else "필드 탐험"
                
                print(f"🎮 현재 상황: {current_situation}")
                
                # 상황에 맞는 추가 웹 학습 (필요시)
                await self.adaptive_web_learning(current_situation)
                
                # 웹 지식 기반 결정
                decision = self.get_web_informed_decision(current_situation)
                
                action = decision["action"]
                reasoning = decision.get("reasoning", "")
                web_knowledge = decision.get("web_knowledge_used", "")
                
                print(f"🧠 선택한 행동: {action} - {reasoning}")
                if web_knowledge:
                    print(f"📚 활용된 웹 지식: {web_knowledge}")
                
                # 행동 실행
                self.execute_action(action)
                
                # 결과 관찰
                time.sleep(1.5)
                
                iteration += 1
                
                # 주기적 학습 요약
                if iteration % 10 == 0:
                    summary = self.web_crawler.get_learning_summary()
                    print(f"\n📊 학습 현황 ({iteration}회 플레이)")
                    print(f"   웹 지식: {summary['total_web_knowledge']}개")
                    print(f"   발견된 패턴: {summary['discovered_patterns']}개")
                    print(f"   평균 신뢰도: {summary['average_confidence']:.2f}")
                
            except KeyboardInterrupt:
                print("\n⏹️ 플레이 중단")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")
                iteration += 1
        
        print("\n🎉 웹 학습 강화 플레이 완료!")
        self.print_final_web_report()
    
    def print_final_web_report(self):
        """최종 웹 학습 보고서"""
        summary = self.web_crawler.get_learning_summary()
        
        print("\n" + "="*60)
        print("🌐 웹 학습 강화 AI 최종 보고서")
        print("="*60)
        
        print(f"📚 학습 성과:")
        print(f"   수집된 웹 지식: {summary['total_web_knowledge']}개")
        print(f"   발견한 패턴: {summary['discovered_patterns']}개")
        print(f"   패턴 신뢰도: {summary['average_confidence']:.2f}")
        
        print(f"\n🎯 자율 학습 특징:")
        print(f"   - 인터넷에서 자동으로 게임 정보 수집")
        print(f"   - 상황에 맞는 적응적 추가 학습")
        print(f"   - 웹 지식을 게임 플레이에 실시간 적용")
        print(f"   - 지속적인 패턴 발견 및 개선")


async def main():
    """메인 실행 함수"""
    
    print("🌐 웹 학습 강화 영웅전설4 AI")
    print("스스로 인터넷에서 게임 정보를 찾아 학습하는 시스템")
    print()
    
    print("🚀 특징:")
    print("- 자율적 웹 크롤링으로 게임 정보 수집")
    print("- 키보드 조작법, 게임 팁 자동 학습")
    print("- 상황별 적응적 추가 학습")
    print("- 웹 지식을 실시간 게임 플레이에 활용")
    print()
    
    try:
        iterations = input("학습 플레이 횟수 (기본값 50): ").strip()
        max_iterations = int(iterations) if iterations else 50
        
        print(f"\n🌐 {max_iterations}회 웹 학습 강화 플레이를 시작합니다!")
        print("💡 처음에는 웹에서 게임 정보를 수집합니다 (1-2분 소요)")
        print("\n시작하려면 Enter를 누르세요...")
        input()
        
        # 웹 강화 AI 실행
        ai = WebEnhancedGameAI()
        await ai.web_enhanced_play(max_iterations=max_iterations)
        
    except KeyboardInterrupt:
        print("\n👋 웹 학습 AI 개발 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main())