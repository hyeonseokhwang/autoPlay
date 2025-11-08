"""
🧠 RAG 기반 자율 학습 영웅전설4 AI
경험을 벡터화하여 지속적으로 지식이 축적되는 시스템
"""

import numpy as np
import json
import time
import cv2
import pickle
from datetime import datetime
from collections import defaultdict, deque
import requests
import hashlib
import os
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

class GameKnowledgeRAG:
    """게임 지식 RAG 시스템"""
    
    def __init__(self, db_path="game_knowledge.db"):
        self.db_path = db_path
        self.embedding_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # 벡터 데이터베이스 초기화
        self.init_vector_database()
        self.load_embedding_model()
        
        # 지식 카테고리
        self.knowledge_categories = {
            "screen_states": "화면 상태별 최적 행동",
            "battle_patterns": "전투 패턴 및 전략", 
            "exploration_routes": "탐험 경로 및 발견",
            "success_sequences": "성공적인 행동 시퀀스",
            "failure_analysis": "실패 원인 및 교훈",
            "game_mechanics": "게임 메커니즘 이해",
            "contextual_hints": "상황별 힌트 및 팁"
        }
    
    def init_vector_database(self):
        """벡터 데이터베이스 초기화"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # 경험 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                category TEXT,
                screen_hash TEXT,
                action_taken TEXT,
                result TEXT,
                success_score REAL,
                context_description TEXT,
                learned_concept TEXT,
                embedding_vector TEXT,
                relevance_count INTEGER DEFAULT 0
            )
        """)
        
        # 패턴 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT UNIQUE,
                pattern_description TEXT,
                confidence_score REAL,
                usage_count INTEGER DEFAULT 0,
                last_updated TEXT,
                embedding_vector TEXT
            )
        """)
        
        # 성공 시퀀스 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS success_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_name TEXT,
                actions_sequence TEXT,
                context_conditions TEXT,
                success_rate REAL,
                total_attempts INTEGER,
                embedding_vector TEXT
            )
        """)
        
        self.conn.commit()
        print("📦 RAG 벡터 데이터베이스 초기화 완료")
    
    def load_embedding_model(self):
        """임베딩 모델 로드"""
        try:
            # 한국어 지원 임베딩 모델
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print("🔤 다국어 임베딩 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ 임베딩 모델 로드 실패: {e}")
            print("💡 대안: TF-IDF 벡터라이저 사용")
            self.embedding_model = None
    
    def vectorize_experience(self, experience_text):
        """경험을 벡터로 변환"""
        if self.embedding_model:
            # Sentence Transformers 사용
            embedding = self.embedding_model.encode(experience_text)
            return embedding.tolist()
        else:
            # TF-IDF 대안 사용
            try:
                tfidf_matrix = self.vectorizer.fit_transform([experience_text])
                return tfidf_matrix.toarray()[0].tolist()
            except:
                return [0.0] * 100  # 기본 벡터
    
    def store_experience(self, category, screen_hash, action, result, success_score, context, learned_concept):
        """경험을 RAG 데이터베이스에 저장"""
        
        # 경험을 텍스트로 구성
        experience_text = f"""
        상황: {context}
        화면: {screen_hash}
        행동: {action}
        결과: {result}
        성공도: {success_score}
        학습내용: {learned_concept}
        """
        
        # 벡터화
        embedding_vector = self.vectorize_experience(experience_text)
        embedding_json = json.dumps(embedding_vector)
        
        # 데이터베이스에 저장
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO experiences 
            (timestamp, category, screen_hash, action_taken, result, 
             success_score, context_description, learned_concept, embedding_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            category,
            screen_hash, 
            str(action),
            result,
            success_score,
            context,
            learned_concept,
            embedding_json
        ))
        
        self.conn.commit()
        experience_id = cursor.lastrowid
        
        print(f"💾 경험 저장됨 (ID: {experience_id}): {learned_concept}")
        return experience_id
    
    def retrieve_relevant_experiences(self, current_context, top_k=5):
        """현재 상황과 유사한 과거 경험 검색"""
        
        # 현재 상황을 벡터화
        query_vector = self.vectorize_experience(current_context)
        
        # 데이터베이스에서 모든 경험 가져오기
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM experiences ORDER BY timestamp DESC LIMIT 100")
        experiences = cursor.fetchall()
        
        if not experiences:
            return []
        
        # 유사도 계산
        similarities = []
        for exp in experiences:
            exp_id = exp[0]
            embedding_json = exp[9]  # embedding_vector 컬럼
            
            try:
                exp_vector = json.loads(embedding_json)
                
                # 벡터 길이 맞추기
                if len(exp_vector) != len(query_vector):
                    continue
                
                # 코사인 유사도 계산
                similarity = cosine_similarity([query_vector], [exp_vector])[0][0]
                similarities.append((similarity, exp))
                
            except Exception as e:
                continue
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 k개 반환
        relevant_experiences = []
        for sim_score, exp in similarities[:top_k]:
            relevant_experiences.append({
                'id': exp[0],
                'similarity': sim_score,
                'category': exp[2],
                'action': exp[4],
                'result': exp[5],
                'success_score': exp[6],
                'context': exp[7],
                'learned_concept': exp[8],
                'timestamp': exp[1]
            })
        
        return relevant_experiences
    
    def extract_pattern(self, experiences_batch):
        """경험 배치에서 패턴 추출"""
        if len(experiences_batch) < 3:
            return None
        
        # 성공적인 경험들 분석
        successful_experiences = [exp for exp in experiences_batch if exp['success_score'] > 0.6]
        
        if len(successful_experiences) < 2:
            return None
        
        # 공통 패턴 찾기
        common_actions = defaultdict(int)
        common_contexts = defaultdict(int)
        
        for exp in successful_experiences:
            common_actions[exp['action']] += 1
            # 컨텍스트에서 키워드 추출
            context_words = exp['context'].split()
            for word in context_words:
                if len(word) > 2:  # 의미있는 단어만
                    common_contexts[word] += 1
        
        # 패턴 생성
        if common_actions and common_contexts:
            most_common_action = max(common_actions.items(), key=lambda x: x[1])
            most_common_context = max(common_contexts.items(), key=lambda x: x[1])
            
            pattern_name = f"{most_common_context[0]}_{most_common_action[0]}"
            pattern_description = f"{most_common_context[0]} 상황에서 {most_common_action[0]} 행동이 효과적"
            
            confidence_score = (most_common_action[1] + most_common_context[1]) / len(successful_experiences)
            
            # 패턴을 데이터베이스에 저장
            self.store_pattern(pattern_name, pattern_description, confidence_score)
            
            return {
                'name': pattern_name,
                'description': pattern_description,
                'confidence': confidence_score
            }
        
        return None
    
    def store_pattern(self, pattern_name, description, confidence):
        """패턴을 데이터베이스에 저장"""
        
        # 패턴 설명을 벡터화
        pattern_vector = self.vectorize_experience(description)
        embedding_json = json.dumps(pattern_vector)
        
        cursor = self.conn.cursor()
        
        # 기존 패턴 업데이트 또는 새로 삽입
        cursor.execute("""
            INSERT OR REPLACE INTO patterns 
            (pattern_name, pattern_description, confidence_score, usage_count, last_updated, embedding_vector)
            VALUES (?, ?, ?, 
                    COALESCE((SELECT usage_count FROM patterns WHERE pattern_name = ?) + 1, 1),
                    ?, ?)
        """, (pattern_name, description, confidence, pattern_name, 
              datetime.now().isoformat(), embedding_json))
        
        self.conn.commit()
        print(f"🧩 패턴 저장: {pattern_name} (신뢰도: {confidence:.2f})")
    
    def get_contextual_advice(self, current_situation):
        """현재 상황에 맞는 조언 생성"""
        
        # 유사한 과거 경험 검색
        relevant_experiences = self.retrieve_relevant_experiences(current_situation, top_k=3)
        
        if not relevant_experiences:
            return "이전 경험이 부족합니다. 탐험을 통해 학습하겠습니다."
        
        # 조언 생성
        advice_parts = []
        
        for exp in relevant_experiences:
            if exp['similarity'] > 0.7:  # 높은 유사도만
                success_indicator = "성공적" if exp['success_score'] > 0.6 else "실패한"
                advice_parts.append(
                    f"유사한 상황에서 '{exp['action']}' 행동이 {success_indicator}이었습니다 "
                    f"(유사도: {exp['similarity']:.2f})"
                )
        
        if advice_parts:
            return " | ".join(advice_parts)
        else:
            return "유사한 경험이 있지만 확실하지 않습니다. 신중히 탐험하겠습니다."
    
    def get_knowledge_summary(self):
        """축적된 지식 요약"""
        cursor = self.conn.cursor()
        
        # 통계 수집
        cursor.execute("SELECT COUNT(*) FROM experiences")
        total_experiences = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM patterns") 
        total_patterns = cursor.fetchone()[0]
        
        cursor.execute("SELECT category, COUNT(*) FROM experiences GROUP BY category")
        category_stats = cursor.fetchall()
        
        cursor.execute("SELECT AVG(success_score) FROM experiences")
        avg_success = cursor.fetchone()[0] or 0
        
        summary = {
            'total_experiences': total_experiences,
            'total_patterns': total_patterns,
            'avg_success_rate': avg_success,
            'category_distribution': dict(category_stats)
        }
        
        return summary


class RAGEnhancedSelfLearningAI:
    """RAG 강화 자율 학습 AI"""
    
    def __init__(self):
        from isolated_seeker import IsolatedDOSBoxSeeker
        
        self.base_seeker = IsolatedDOSBoxSeeker()
        
        # LLM 설정
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.model = "qwen2.5-coder:7b"
        
        # RAG 시스템
        self.rag_system = GameKnowledgeRAG()
        
        # 학습 세션 관리
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exploration_count = 0
        self.learning_batch_size = 5
        self.pending_experiences = []
        
        print("🧠 RAG 강화 자율 학습 AI 초기화 완료")
        print(f"📊 세션 ID: {self.session_id}")
        
        # 기존 지식 요약 출력
        self.print_knowledge_status()
    
    def print_knowledge_status(self):
        """현재 지식 상태 출력"""
        summary = self.rag_system.get_knowledge_summary()
        
        print(f"\n📚 현재 지식 상태:")
        print(f"   총 경험: {summary['total_experiences']}개")
        print(f"   발견한 패턴: {summary['total_patterns']}개")
        print(f"   평균 성공률: {summary['avg_success_rate']:.2f}")
        
        if summary['category_distribution']:
            print(f"   카테고리별 경험:")
            for category, count in summary['category_distribution'].items():
                print(f"     - {category}: {count}개")
    
    def analyze_screen_with_rag(self, screen):
        """RAG를 활용한 화면 분석"""
        
        # 기본 화면 분석
        screen_hash = self.hash_screen(screen)
        basic_analysis = self.analyze_screen_basic(screen)
        
        # RAG에서 관련 경험 검색
        current_context = f"화면유형: {basic_analysis.get('screen_type', '알수없음')}, 특징: {basic_analysis.get('prominent_elements', [])}"
        
        relevant_experiences = self.rag_system.retrieve_relevant_experiences(current_context, top_k=3)
        
        # 상황별 조언 가져오기
        contextual_advice = self.rag_system.get_contextual_advice(current_context)
        
        # LLM에게 RAG 정보와 함께 분석 요청
        rag_enhanced_prompt = f"""
당신은 이 게임을 학습 중인 AI입니다.

현재 화면 분석:
{json.dumps(basic_analysis, ensure_ascii=False, indent=2)}

과거 유사한 경험들:
{self.format_experiences_for_llm(relevant_experiences)}

AI 조언:
{contextual_advice}

이 정보들을 종합하여 다음을 결정해주세요:

1. 현재 상황에 대한 이해
2. 추천하는 행동 (0-6)
3. 그 이유
4. 예상되는 결과
5. 이번 행동으로 학습할 수 있는 것

JSON 형태로:
{{
    "situation_understanding": "상황 이해",
    "recommended_action": 2,
    "reasoning": "선택 이유",
    "expected_outcome": "예상 결과",
    "learning_opportunity": "학습 기회"
}}
"""
        
        try:
            llm_response = self.call_llm(rag_enhanced_prompt)
            decision = self.parse_llm_json(llm_response)
            
            # 기본값 설정
            if not decision or 'recommended_action' not in decision:
                decision = {
                    "situation_understanding": basic_analysis.get('screen_type', '알수없음'),
                    "recommended_action": np.random.randint(0, 7),
                    "reasoning": "RAG 정보 부족으로 탐험적 행동",
                    "expected_outcome": "새로운 정보 발견 기대",
                    "learning_opportunity": "이 상황에서의 행동 결과 학습"
                }
            
            return decision, current_context, relevant_experiences
            
        except Exception as e:
            print(f"⚠️ RAG 분석 실패: {e}")
            return self.fallback_decision(), current_context, []
    
    def format_experiences_for_llm(self, experiences):
        """경험들을 LLM이 이해하기 쉽게 포맷팅"""
        if not experiences:
            return "관련된 과거 경험이 없습니다."
        
        formatted = []
        for i, exp in enumerate(experiences, 1):
            success_desc = "성공적" if exp['success_score'] > 0.6 else "실패한"
            formatted.append(f"""
경험 {i} (유사도: {exp['similarity']:.2f}):
- 행동: {exp['action']}
- 결과: {exp['result']} ({success_desc})
- 학습내용: {exp['learned_concept']}
""")
        
        return "\n".join(formatted)
    
    def learn_and_store_experience(self, before_context, action_decision, after_screen, after_context):
        """경험을 분석하고 RAG에 저장"""
        
        # 성공도 평가
        success_score = self.evaluate_action_success(before_context, action_decision, after_context)
        
        # 학습된 개념 추출
        learning_prompt = f"""
다음 행동의 결과를 분석하여 학습할 수 있는 개념을 추출해주세요:

이전 상황: {before_context}
선택한 행동: {action_decision.get('recommended_action')} - {action_decision.get('reasoning')}
예상 결과: {action_decision.get('expected_outcome')}
실제 결과: {after_context}
성공도: {success_score}

학습할 수 있는 개념이나 규칙을 간단히 설명해주세요.
"""
        
        try:
            learning_response = self.call_llm(learning_prompt)
            learned_concept = learning_response.strip()
        except:
            learned_concept = f"행동 {action_decision.get('recommended_action')} 결과 관찰"
        
        # 카테고리 결정
        category = self.determine_experience_category(before_context, action_decision)
        
        # RAG에 저장
        screen_hash = self.hash_screen(after_screen) if after_screen is not None else "unknown"
        
        experience_id = self.rag_system.store_experience(
            category=category,
            screen_hash=screen_hash,
            action=action_decision.get('recommended_action'),
            result=after_context,
            success_score=success_score,
            context=before_context,
            learned_concept=learned_concept
        )
        
        # 배치 처리를 위해 대기열에 추가
        self.pending_experiences.append({
            'id': experience_id,
            'category': category,
            'action': action_decision.get('recommended_action'),
            'result': after_context,
            'success_score': success_score,
            'context': before_context,
            'learned_concept': learned_concept,
            'timestamp': datetime.now().isoformat()
        })
        
        # 배치가 찼으면 패턴 추출
        if len(self.pending_experiences) >= self.learning_batch_size:
            self.process_learning_batch()
        
        return learned_concept
    
    def process_learning_batch(self):
        """학습 배치 처리 및 패턴 추출"""
        print(f"\n🔍 {len(self.pending_experiences)}개 경험으로부터 패턴 추출 중...")
        
        # 패턴 추출
        extracted_pattern = self.rag_system.extract_pattern(self.pending_experiences)
        
        if extracted_pattern:
            print(f"💡 새로운 패턴 발견: {extracted_pattern['name']}")
            print(f"   설명: {extracted_pattern['description']}")
            print(f"   신뢰도: {extracted_pattern['confidence']:.2f}")
        
        # 배치 초기화
        self.pending_experiences = []
        
        # 주기적 지식 요약 출력
        if self.exploration_count % 20 == 0:
            self.print_knowledge_status()
    
    def evaluate_action_success(self, before_context, action_decision, after_context):
        """행동의 성공도 평가"""
        
        # 기본 성공도 (화면 변화 있으면 0.5)
        base_score = 0.5 if "화면" in after_context else 0.3
        
        # 예상과 실제 결과 비교
        expected = action_decision.get('expected_outcome', '').lower()
        actual = after_context.lower()
        
        # 키워드 매칭으로 예상 정확도 측정
        expected_words = set(expected.split())
        actual_words = set(actual.split())
        
        if expected_words and actual_words:
            overlap = len(expected_words.intersection(actual_words))
            prediction_accuracy = overlap / len(expected_words.union(actual_words))
            base_score += prediction_accuracy * 0.3
        
        # 탐험 보너스 (새로운 것 발견)
        if "새로운" in after_context or "발견" in after_context:
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def determine_experience_category(self, context, action_decision):
        """경험의 카테고리 결정"""
        
        context_lower = context.lower()
        reasoning_lower = action_decision.get('reasoning', '').lower()
        
        if any(word in context_lower for word in ['전투', 'battle', 'hp', 'mp']):
            return 'battle_patterns'
        elif any(word in reasoning_lower for word in ['탐험', '이동', '새로운']):
            return 'exploration_routes'
        elif '성공' in reasoning_lower or '효과' in reasoning_lower:
            return 'success_sequences'
        elif '실패' in reasoning_lower or '잘못' in reasoning_lower:
            return 'failure_analysis'
        elif any(word in context_lower for word in ['화면', '메뉴', '상태']):
            return 'screen_states'
        else:
            return 'game_mechanics'
    
    def hash_screen(self, screen):
        """화면 해시 생성"""
        if screen is None:
            return "no_screen"
        
        small_screen = cv2.resize(screen, (64, 48))
        gray = cv2.cvtColor(small_screen, cv2.COLOR_BGR2GRAY)
        
        features = [
            np.mean(gray),
            np.std(gray), 
            len(np.unique(gray)),
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ]
        
        feature_str = "_".join(f"{f:.2f}" for f in features)
        return hashlib.md5(feature_str.encode()).hexdigest()[:8]
    
    def analyze_screen_basic(self, screen):
        """기본 화면 분석"""
        if screen is None:
            return {"screen_type": "invalid", "prominent_elements": []}
        
        # 간단한 시각적 특징 추출
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        return {
            "screen_type": "game_screen",
            "prominent_elements": ["UI요소", "게임화면"],
            "brightness": float(np.mean(gray)),
            "contrast": float(np.std(gray))
        }
    
    def call_llm(self, prompt, timeout=15):
        """LLM 호출"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
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
            return {}
        except:
            return {}
    
    def fallback_decision(self):
        """폴백 결정"""
        return {
            "situation_understanding": "분석 실패",
            "recommended_action": np.random.randint(0, 7),
            "reasoning": "랜덤 탐험",
            "expected_outcome": "새로운 정보",
            "learning_opportunity": "기본 행동 반응 학습"
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
    
    def autonomous_rag_learning(self, max_iterations=100):
        """RAG 기반 자율 학습 실행"""
        
        print("🚀 RAG 기반 자율 학습 시작!")
        print("📊 경험이 축적될수록 더 똑똑해집니다")
        print()
        
        if not self.base_seeker.find_dosbox_window():
            print("❌ DOSBox 창을 찾을 수 없습니다!")
            return
        
        while self.exploration_count < max_iterations:
            try:
                print(f"\n--- RAG 학습 #{self.exploration_count + 1} ---")
                
                # 현재 화면 캡처
                current_screen = self.base_seeker.capture_dosbox_window()
                if current_screen is None:
                    continue
                
                # RAG 강화 분석
                decision, context_before, relevant_exp = self.analyze_screen_with_rag(current_screen)
                
                print(f"🧠 상황 이해: {decision['situation_understanding']}")
                print(f"🎯 선택한 행동: {decision['recommended_action']} - {decision['reasoning']}")
                
                if relevant_exp:
                    print(f"📚 활용된 과거 경험: {len(relevant_exp)}개")
                
                # 행동 실행
                self.execute_action(decision['recommended_action'])
                
                # 결과 관찰
                time.sleep(1.5)
                result_screen = self.base_seeker.capture_dosbox_window()
                
                if result_screen is not None:
                    # 결과 분석
                    context_after = f"행동 후 화면 변화 관찰됨"
                    
                    # 경험 저장 및 학습
                    learned_concept = self.learn_and_store_experience(
                        context_before, decision, result_screen, context_after
                    )
                    
                    print(f"💡 학습: {learned_concept}")
                
                self.exploration_count += 1
                
                # 주기적 진행 상황 출력
                if self.exploration_count % 10 == 0:
                    summary = self.rag_system.get_knowledge_summary()
                    print(f"\n📈 진행 상황 ({self.exploration_count}회 탐험)")
                    print(f"   축적된 경험: {summary['total_experiences']}개")
                    print(f"   발견한 패턴: {summary['total_patterns']}개")
                    print(f"   현재 성공률: {summary['avg_success_rate']:.2f}")
                
            except KeyboardInterrupt:
                print("\n⏹️ 학습 중단")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")
                self.exploration_count += 1
        
        # 최종 처리
        if self.pending_experiences:
            self.process_learning_batch()
        
        print("\n🎉 RAG 기반 자율 학습 완료!")
        self.print_final_rag_report()
    
    def print_final_rag_report(self):
        """최종 RAG 학습 보고서"""
        summary = self.rag_system.get_knowledge_summary()
        
        print("\n" + "="*60)
        print("🧠 RAG 기반 자율 학습 최종 보고서")
        print("="*60)
        
        print(f"📊 학습 통계:")
        print(f"   총 탐험 횟수: {self.exploration_count}")
        print(f"   축적된 경험: {summary['total_experiences']}개")
        print(f"   발견한 패턴: {summary['total_patterns']}개") 
        print(f"   최종 성공률: {summary['avg_success_rate']:.2f}")
        
        print(f"\n📚 카테고리별 학습 현황:")
        for category, count in summary['category_distribution'].items():
            print(f"   {category}: {count}개 경험")
        
        print(f"\n🎯 RAG 시스템 효과:")
        print(f"   - 경험 벡터화로 유사 상황 빠른 검색")
        print(f"   - 패턴 자동 추출로 전략 개발")
        print(f"   - 지속적 지식 축적으로 성능 향상")
        
        print(f"\n💾 지식베이스: {self.rag_system.db_path}")


def main():
    """메인 실행 함수"""
    
    print("🧠 RAG 기반 자율 학습 영웅전설4 AI")
    print("📚 경험을 벡터화하여 지속적으로 학습하는 시스템")
    print()
    
    print("🚀 특징:")
    print("- 모든 경험을 벡터 데이터베이스에 저장")
    print("- 유사한 상황에서 과거 경험 활용")
    print("- 자동 패턴 추출 및 전략 개발")
    print("- 세션 간 지식 누적 (영구 학습)")
    print()
    
    try:
        # 의존성 확인
        try:
            import sentence_transformers
            print("✅ Sentence Transformers 사용 가능")
        except ImportError:
            print("⚠️ Sentence Transformers 없음. pip install sentence-transformers")
            print("   TF-IDF 백업 시스템 사용")
        
        iterations = input("학습 횟수 (기본값 50): ").strip()
        max_iterations = int(iterations) if iterations else 50
        
        print(f"\n🚀 {max_iterations}회 RAG 기반 자율 학습을 시작합니다!")
        print("💡 Ctrl+C로 중단 시 학습한 내용은 자동 저장됩니다")
        print("\n시작하려면 Enter를 누르세요...")
        input()
        
        # RAG 강화 AI 실행
        ai = RAGEnhancedSelfLearningAI()
        ai.autonomous_rag_learning(max_iterations=max_iterations)
        
    except KeyboardInterrupt:
        print("\n👋 RAG 학습 시스템 개발 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()