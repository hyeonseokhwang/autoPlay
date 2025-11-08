"""
🎮 실제 게임 플레이 중심 RAG AI
GPU를 효율적으로 사용하면서 진짜 게임을 플레이하는 AI
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
import torch
from sentence_transformers import SentenceTransformer

class GameplayRAG:
    """실제 게임플레이 중심 RAG 시스템"""
    
    def __init__(self, use_gpu=True):
        # GPU 설정
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"🚀 GPU 사용: {self.device}")
        
        # 임베딩 모델 (GPU로 로드)
        print("🔤 임베딩 모델 GPU 로딩 중...")
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            device=self.device
        )
        print(f"✅ 임베딩 모델 로드 완료 ({self.device})")
        
        # SQLite DB
        self.db_path = "gameplay_knowledge.db"
        self.init_database()
        
        # 실제 게임 행동 매핑
        self.actions = {
            0: {"name": "왼쪽 이동", "key": "VK_LEFT", "category": "movement"},
            1: {"name": "오른쪽 이동", "key": "VK_RIGHT", "category": "movement"},
            2: {"name": "위로 이동", "key": "VK_UP", "category": "movement"},
            3: {"name": "아래로 이동", "key": "VK_DOWN", "category": "movement"},
            4: {"name": "확인/공격", "key": "VK_RETURN", "category": "action"},
            5: {"name": "취소/메뉴", "key": "VK_ESCAPE", "category": "action"},
            6: {"name": "대기", "key": None, "category": "wait"}
        }
        
        print("🎮 실제 게임플레이 RAG 시스템 준비 완료!")
    
    def init_database(self):
        """게임플레이 DB 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gameplay_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                screen_before TEXT,
                action_taken INTEGER,
                screen_after TEXT,
                screen_changed BOOLEAN,
                battle_detected BOOLEAN,
                hp_changed BOOLEAN,
                success_score REAL,
                situation_description TEXT,
                learned_insight TEXT,
                embedding_vector TEXT,
                game_progress INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS successful_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                situation_context TEXT,
                action_sequence TEXT,
                success_rate REAL,
                total_uses INTEGER DEFAULT 1,
                embedding_vector TEXT,
                discovered_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        print("📦 게임플레이 DB 초기화 완료")
    
    def encode_experience(self, situation_text):
        """경험을 GPU로 벡터화"""
        with torch.no_grad():
            # GPU에서 빠르게 인코딩
            embedding = self.embedding_model.encode(
                situation_text,
                convert_to_tensor=True,
                device=self.device
            )
            return embedding.cpu().numpy().tolist()
    
    def store_gameplay_experience(self, screen_before, action, screen_after, 
                                battle_before, battle_after, success_score, situation, insight):
        """실제 게임플레이 경험 저장"""
        
        # 상황 텍스트 생성
        situation_text = f"""
        상황: {situation}
        행동: {self.actions[action]['name']}
        전투상태변화: {battle_before} → {battle_after}
        성공도: {success_score}
        깨달은점: {insight}
        """
        
        # GPU로 벡터화
        embedding = self.encode_experience(situation_text)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO gameplay_experiences 
            (timestamp, screen_before, action_taken, screen_after, 
             screen_changed, battle_detected, success_score, 
             situation_description, learned_insight, embedding_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            screen_before,
            action,
            screen_after,
            screen_before != screen_after,
            battle_after,
            success_score,
            situation,
            insight,
            json.dumps(embedding)
        ))
        
        conn.commit()
        conn.close()
        
        return cursor.lastrowid
    
    def find_similar_situations(self, current_situation, top_k=3):
        """현재 상황과 유사한 과거 경험 검색"""
        
        # 현재 상황을 벡터화
        query_embedding = self.encode_experience(current_situation)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, action_taken, success_score, situation_description, 
                   learned_insight, embedding_vector, screen_changed, battle_detected
            FROM gameplay_experiences 
            ORDER BY id DESC LIMIT 50
        """)
        
        experiences = cursor.fetchall()
        conn.close()
        
        if not experiences:
            return []
        
        # 유사도 계산
        similarities = []
        for exp in experiences:
            try:
                exp_embedding = json.loads(exp[5])
                
                # 코사인 유사도
                sim_score = np.dot(query_embedding, exp_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(exp_embedding)
                )
                
                similarities.append({
                    'similarity': float(sim_score),
                    'action': exp[1],
                    'success_score': exp[2],
                    'situation': exp[3],
                    'insight': exp[4],
                    'screen_changed': exp[6],
                    'battle_detected': exp[7]
                })
                
            except:
                continue
        
        # 유사도 순 정렬
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]


class RealGameplayAI:
    """실제 게임을 플레이하는 AI"""
    
    def __init__(self):
        from isolated_seeker import IsolatedDOSBoxSeeker
        
        self.seeker = IsolatedDOSBoxSeeker()
        self.rag = GameplayRAG(use_gpu=True)
        
        # LLM 설정
        self.llm_url = "http://localhost:11434/api/generate"
        self.model_name = "qwen2.5-coder:7b"
        
        # 게임 상태 추적
        self.current_hp = 100
        self.current_mp = 100
        self.battle_count = 0
        self.exploration_count = 0
        
        # 학습 통계
        self.successful_moves = 0
        self.total_moves = 0
        
        print("🎮 실제 게임플레이 AI 준비 완료!")
        print(f"🔥 GPU 활용 RAG 시스템 활성화")
    
    def analyze_game_screen(self, screen):
        """실제 게임 화면 분석"""
        if screen is None:
            return {
                "type": "invalid",
                "battle_active": False,
                "characters_visible": False,
                "hp_visible": False,
                "description": "화면 캡처 실패"
            }
        
        # 실제 게임 요소 감지
        is_battle = self.seeker.is_battle_screen(screen)
        
        # 화면 특성 분석
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        # 텍스트 영역 감지 (HP/MP 등)
        text_areas = self.detect_text_regions(gray)
        
        # 캐릭터/적 감지
        entities = self.detect_game_entities(screen)
        
        analysis = {
            "type": "battle" if is_battle else "field",
            "battle_active": is_battle,
            "text_regions": len(text_areas),
            "entities_count": len(entities),
            "brightness": float(np.mean(gray)),
            "activity_level": float(np.std(gray)),
            "description": self.generate_scene_description(is_battle, len(text_areas), len(entities))
        }
        
        return analysis
    
    def detect_text_regions(self, gray_image):
        """텍스트 영역 감지 (HP/MP 바 등)"""
        # 간단한 텍스트 영역 감지
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 텍스트 같은 비율 필터링
            if 10 < w < 200 and 5 < h < 50 and w > h:
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    def detect_game_entities(self, screen):
        """게임 내 엔티티 감지 (캐릭터, 적 등)"""
        # HSV로 변환하여 특정 색상 감지
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        
        # 캐릭터 색상 범위 (대략적)
        lower_char = np.array([0, 50, 50])
        upper_char = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_char, upper_char)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        entities = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 최소 크기 필터
                entities.append(contour)
        
        return entities
    
    def generate_scene_description(self, is_battle, text_count, entity_count):
        """장면 설명 생성"""
        if is_battle:
            return f"전투 화면 - 텍스트 {text_count}개, 객체 {entity_count}개"
        else:
            return f"필드 화면 - 텍스트 {text_count}개, 객체 {entity_count}개"
    
    def make_intelligent_decision(self, screen_analysis, similar_experiences):
        """RAG 정보 기반 지능적 결정"""
        
        # 상황 컨텍스트 생성
        context = f"""
현재 상황:
- 화면 유형: {screen_analysis['type']}
- 전투 상태: {screen_analysis['battle_active']}
- 텍스트 영역: {screen_analysis['text_regions']}개
- 게임 객체: {screen_analysis['entities_count']}개
- 화면 활동성: {screen_analysis['activity_level']:.2f}

과거 유사한 경험:
"""
        
        for i, exp in enumerate(similar_experiences, 1):
            context += f"""
경험 {i} (유사도: {exp['similarity']:.2f}):
- 행동: {self.rag.actions[exp['action']]['name']}
- 성공도: {exp['success_score']:.2f}
- 결과: {exp['insight']}
"""
        
        # LLM에게 결정 요청
        prompt = f"""
영웅전설4를 플레이 중입니다. 다음 상황에서 최적의 행동을 선택하세요.

{context}

사용 가능한 행동:
0: 왼쪽 이동 (새 지역 탐험)
1: 오른쪽 이동 (새 지역 탐험)
2: 위로 이동 (새 지역 탐험)
3: 아래로 이동 (새 지역 탐험)
4: 확인/공격 (전투 시 공격, 평상시 조사)
5: 취소/메뉴 (메뉴 열기, 전투 중 방어)
6: 대기 (상황 관찰)

JSON 형태로 답하세요:
{{
    "action": 1,
    "reasoning": "오른쪽으로 이동해서 새로운 적을 찾아 전투 경험을 쌓겠습니다",
    "confidence": 0.8,
    "expected_outcome": "새로운 지역 발견 또는 적 조우"
}}
"""
        
        try:
            response = self.call_llm(prompt, timeout=10)
            decision = self.parse_json_response(response)
            
            if decision and 'action' in decision:
                return decision
            else:
                # LLM 실패 시 RAG 기반 폴백
                return self.rag_based_decision(screen_analysis, similar_experiences)
                
        except Exception as e:
            print(f"⚠️ LLM 결정 실패: {e}")
            return self.rag_based_decision(screen_analysis, similar_experiences)
    
    def rag_based_decision(self, screen_analysis, similar_experiences):
        """RAG만으로 결정 (LLM 실패 시)"""
        
        # 가장 성공적이었던 경험의 행동 선택
        if similar_experiences:
            best_exp = max(similar_experiences, key=lambda x: x['success_score'])
            
            return {
                "action": best_exp['action'],
                "reasoning": f"과거 성공 경험 활용: {best_exp['insight']}",
                "confidence": best_exp['similarity'],
                "expected_outcome": "과거와 유사한 긍정적 결과"
            }
        
        # 경험이 없으면 상황에 맞는 기본 행동
        if screen_analysis['battle_active']:
            return {
                "action": 4,
                "reasoning": "전투 상황에서 기본 공격",
                "confidence": 0.6,
                "expected_outcome": "적에게 피해"
            }
        else:
            # 탐험 행동 (랜덤하게 이동)
            action = np.random.choice([0, 1, 2, 3])
            return {
                "action": action,
                "reasoning": "새로운 지역 탐험",
                "confidence": 0.5,
                "expected_outcome": "새로운 발견"
            }
    
    def execute_game_action(self, action_id):
        """실제 게임 행동 실행"""
        
        action_info = self.rag.actions[action_id]
        print(f"🎮 실행: {action_info['name']}")
        
        if action_info['key'] is None:
            # 대기 행동
            time.sleep(1)
            return True
        
        # 실제 키 입력
        vk_mapping = {
            "VK_LEFT": self.seeker.VK_LEFT,
            "VK_RIGHT": self.seeker.VK_RIGHT,
            "VK_UP": self.seeker.VK_UP,
            "VK_DOWN": self.seeker.VK_DOWN,
            "VK_RETURN": self.seeker.VK_RETURN,
            "VK_ESCAPE": self.seeker.VK_ESCAPE
        }
        
        vk_code = vk_mapping.get(action_info['key'])
        if vk_code:
            return self.seeker.send_key_message(vk_code)
        
        return False
    
    def evaluate_action_result(self, before_analysis, after_analysis, action, decision):
        """행동 결과 평가"""
        
        success_score = 0.5  # 기본 점수
        
        # 화면 변화 보너스
        if before_analysis['type'] != after_analysis['type']:
            success_score += 0.3
            print("✅ 화면 상태 변화 감지!")
        
        # 전투 진입 보너스
        if not before_analysis['battle_active'] and after_analysis['battle_active']:
            success_score += 0.4
            self.battle_count += 1
            print(f"⚔️ 전투 #{self.battle_count} 시작!")
        
        # 활동성 변화
        activity_change = abs(after_analysis['activity_level'] - before_analysis['activity_level'])
        if activity_change > 10:
            success_score += 0.2
            print("📈 화면 활동성 증가!")
        
        # 탐험 보너스
        if self.rag.actions[action]['category'] == 'movement' and before_analysis['type'] == after_analysis['type']:
            success_score += 0.1  # 이동했다는 것 자체가 탐험
        
        return min(1.0, success_score)
    
    def generate_learning_insight(self, before_analysis, after_analysis, action, success_score):
        """학습 통찰 생성"""
        
        action_name = self.rag.actions[action]['name']
        
        insights = []
        
        if success_score > 0.7:
            insights.append(f"{action_name} 행동이 효과적이었음")
        
        if before_analysis['type'] != after_analysis['type']:
            insights.append(f"화면 전환 성공: {before_analysis['type']} → {after_analysis['type']}")
        
        if after_analysis['battle_active']:
            insights.append("전투 상황 진입 성공")
        
        if not insights:
            insights.append(f"{action_name} 실행으로 상황 관찰")
        
        return " | ".join(insights)
    
    def call_llm(self, prompt, timeout=10):
        """LLM 호출"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.8,
                "num_predict": 100
            }
        }
        
        response = requests.post(self.llm_url, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"LLM 오류: {response.status_code}")
    
    def parse_json_response(self, text):
        """JSON 응답 파싱"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start != -1 and end > start:
                json_text = text[start:end]
                return json.loads(json_text)
            
            return None
        except:
            return None
    
    def real_gameplay_loop(self, max_actions=30):
        """실제 게임플레이 루프"""
        
        print("🚀 실제 게임플레이 AI 시작!")
        print("🎮 영웅전설4가 열려있는지 확인하세요!")
        print()
        
        if not self.seeker.find_dosbox_window():
            print("❌ DOSBox 창을 찾을 수 없습니다!")
            return
        
        print("✅ DOSBox 연결 성공!")
        print(f"🎯 목표: {max_actions}번의 실제 게임 행동")
        print()
        
        for action_num in range(max_actions):
            try:
                print(f"\n--- 🎮 게임 행동 #{action_num + 1} ---")
                
                # 1. 현재 화면 분석
                screen_before = self.seeker.capture_dosbox_window()
                if screen_before is None:
                    print("⚠️ 화면 캡처 실패")
                    continue
                
                analysis_before = self.analyze_game_screen(screen_before)
                print(f"📊 현재 상황: {analysis_before['description']}")
                
                # 2. RAG에서 유사한 경험 검색
                situation_text = f"화면: {analysis_before['type']}, 전투: {analysis_before['battle_active']}, 활동성: {analysis_before['activity_level']:.1f}"
                similar_exp = self.rag.find_similar_situations(situation_text, top_k=3)
                
                if similar_exp:
                    print(f"🧠 유사한 과거 경험: {len(similar_exp)}개")
                
                # 3. 지능적 결정
                decision = self.make_intelligent_decision(analysis_before, similar_exp)
                action = decision['action']
                
                print(f"🎯 결정: {self.rag.actions[action]['name']}")
                print(f"💭 이유: {decision['reasoning']}")
                
                # 4. 행동 실행
                execution_success = self.execute_game_action(action)
                
                if not execution_success:
                    print("❌ 행동 실행 실패")
                    continue
                
                # 5. 결과 관찰
                time.sleep(1.5)  # 게임 반응 대기
                
                screen_after = self.seeker.capture_dosbox_window()
                if screen_after is None:
                    continue
                
                analysis_after = self.analyze_game_screen(screen_after)
                
                # 6. 결과 평가 및 학습
                success_score = self.evaluate_action_result(
                    analysis_before, analysis_after, action, decision
                )
                
                insight = self.generate_learning_insight(
                    analysis_before, analysis_after, action, success_score
                )
                
                print(f"📈 성공도: {success_score:.2f}")
                print(f"💡 학습: {insight}")
                
                # 7. RAG에 경험 저장
                self.rag.store_gameplay_experience(
                    screen_before=str(hash(screen_before.tobytes())),
                    action=action,
                    screen_after=str(hash(screen_after.tobytes())),
                    battle_before=analysis_before['battle_active'],
                    battle_after=analysis_after['battle_active'],
                    success_score=success_score,
                    situation=situation_text,
                    insight=insight
                )
                
                # 8. 통계 업데이트
                self.total_moves += 1
                if success_score > 0.6:
                    self.successful_moves += 1
                
                # 9. 주기적 진행 상황 출력
                if (action_num + 1) % 5 == 0:
                    success_rate = self.successful_moves / self.total_moves if self.total_moves > 0 else 0
                    print(f"\n📊 진행 상황 ({action_num + 1}/{max_actions})")
                    print(f"   성공률: {success_rate:.2f} ({self.successful_moves}/{self.total_moves})")
                    print(f"   전투 횟수: {self.battle_count}")
                
            except KeyboardInterrupt:
                print("\n⏹️ 게임플레이 중단")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")
                continue
        
        # 최종 결과
        print("\n🎉 실제 게임플레이 완료!")
        self.print_final_stats()
    
    def print_final_stats(self):
        """최종 통계 출력"""
        success_rate = self.successful_moves / self.total_moves if self.total_moves > 0 else 0
        
        print("\n" + "="*50)
        print("🏆 실제 게임플레이 AI 최종 결과")
        print("="*50)
        print(f"총 행동 수: {self.total_moves}")
        print(f"성공한 행동: {self.successful_moves}")
        print(f"성공률: {success_rate:.2%}")
        print(f"발견한 전투: {self.battle_count}회")
        print(f"GPU 활용 임베딩: ✅")
        print(f"RAG 지식 베이스: {self.rag.db_path}")


def main():
    """메인 실행"""
    
    print("🎮 실제 게임플레이 RAG AI")
    print("🔥 GPU 가속 임베딩 + 진짜 게임 플레이")
    print()
    
    # GPU 확인
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU 감지: {gpu_name}")
        print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory // 1024**2}MB")
    else:
        print("⚠️ GPU 없음 - CPU로 실행")
    
    print()
    
    try:
        actions = input("게임 행동 횟수 (기본값 20): ").strip()
        max_actions = int(actions) if actions else 20
        
        print(f"\n🎮 {max_actions}번의 실제 게임 행동을 시작합니다!")
        print("🎯 영웅전설4가 DOSBox에서 실행 중인지 확인하세요!")
        print("\n시작하려면 Enter를 누르세요...")
        input()
        
        # 실제 게임플레이 AI 실행
        ai = RealGameplayAI()
        ai.real_gameplay_loop(max_actions=max_actions)
        
    except KeyboardInterrupt:
        print("\n👋 실제 게임플레이 AI 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()