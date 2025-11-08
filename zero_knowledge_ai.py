"""
🧠 완전 자율 학습 AI (알파고 제로 스타일)
아무것도 가르치지 않고 스스로 게임을 터득하게 하는 시스템
"""

import time
import json
import cv2
import numpy as np
from collections import deque, defaultdict
import requests
import hashlib
from datetime import datetime
import pickle
import os

class SelfLearningHeroAI:
    """완전 자율 학습 영웅전설4 AI - 제로 지식에서 시작"""
    
    def __init__(self):
        from isolated_seeker import IsolatedDOSBoxSeeker
        
        self.base_seeker = IsolatedDOSBoxSeeker()
        
        # LLM 설정
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.model = "qwen2.5-coder:7b"  # 추론 능력이 좋은 모델
        
        # 완전 자율 학습을 위한 지식 베이스
        self.knowledge = {
            "screen_states": {},           # 화면 상태별 경험
            "action_consequences": {},     # 행동 → 결과 매핑
            "successful_sequences": [],    # 성공한 행동 시퀀스
            "curiosity_targets": set(),    # 탐험해볼 만한 것들
            "learned_concepts": {},        # 스스로 발견한 개념들
            "meta_strategies": []          # 고수준 전략들
        }
        
        # 자율 학습 파라미터
        self.exploration_rate = 0.8    # 탐험 vs 활용
        self.curiosity_threshold = 0.3 # 새로운 것에 대한 관심도
        self.memory_size = 1000        # 기억할 경험 수
        
        # 경험 메모리 (자동으로 패턴을 찾아냄)
        self.experiences = deque(maxlen=self.memory_size)
        self.screen_history = deque(maxlen=50)
        
        # 자체 생성 보상 시스템
        self.intrinsic_motivation = {
            "novelty_bonus": 10,        # 새로운 것 발견 시 보상
            "progress_bonus": 5,        # 진전이 있을 때 보상
            "consistency_bonus": 3,     # 일관된 패턴 발견 시 보상
            "exploration_bonus": 2      # 탐험 자체에 대한 보상
        }
        
        print("🧠 완전 자율 학습 AI 초기화...")
        print("📚 기존 지식: 없음 (제로 지식 시작)")
        print("🎯 목표: 스스로 게임 규칙과 전략 발견")
        
        self.load_previous_knowledge()
    
    def load_previous_knowledge(self):
        """이전 학습 세션의 지식 로드 (선택적)"""
        knowledge_file = "self_learned_knowledge.pkl"
        
        if os.path.exists(knowledge_file):
            try:
                with open(knowledge_file, 'rb') as f:
                    saved_knowledge = pickle.load(f)
                
                # 기존 지식과 병합 (완전 재시작 vs 계속 학습 선택 가능)
                print("🔍 이전 학습 데이터 발견!")
                print(f"   - 알려진 화면 상태: {len(saved_knowledge.get('screen_states', {}))}")
                print(f"   - 학습된 개념: {len(saved_knowledge.get('learned_concepts', {}))}")
                print(f"   - 성공 시퀀스: {len(saved_knowledge.get('successful_sequences', []))}")
                
                choice = input("이전 지식 사용? (y/n, 기본값 n): ").strip().lower()
                if choice == 'y':
                    self.knowledge = saved_knowledge
                    print("✅ 이전 지식으로 계속 학습")
                else:
                    print("🆕 완전 새로운 학습 시작")
                    
            except Exception as e:
                print(f"⚠️ 이전 지식 로드 실패: {e}")
    
    def analyze_screen_with_zero_knowledge(self, screen):
        """제로 지식 상태에서 화면 분석"""
        
        if screen is None:
            return {"type": "invalid", "features": {}}
        
        # 화면을 해시로 변환하여 고유 식별자 생성
        screen_hash = self.hash_screen(screen)
        
        # LLM에게 "처음 보는" 관점에서 분석하게 함
        analysis_prompt = f"""
당신은 처음으로 이 게임 화면을 보는 AI입니다.
어떤 게임인지, 어떤 규칙인지 전혀 모릅니다.

화면을 보고 다음을 추론해주세요:
1. 이 화면에서 가장 눈에 띄는 요소들
2. 움직일 수 있는 것들 (캐릭터, 커서 등)
3. 숫자나 바 형태의 정보들
4. 반복되는 패턴이나 구조들
5. 이전에 본 화면과의 차이점 (있다면)

순수하게 시각적 관찰만 하고, 게임 용어는 사용하지 마세요.
JSON 형태로 답해주세요:
{{
    "prominent_elements": ["요소1", "요소2"],
    "interactive_objects": ["객체1", "객체2"], 
    "numerical_info": ["정보1", "정보2"],
    "patterns": ["패턴1", "패턴2"],
    "screen_type": "추정_화면_유형",
    "novelty_score": 0.8
}}
"""
        
        try:
            llm_response = self.call_llm(analysis_prompt)
            analysis = self.parse_llm_json(llm_response)
            
            # 화면 상태 기록
            if screen_hash not in self.knowledge["screen_states"]:
                self.knowledge["screen_states"][screen_hash] = {
                    "first_seen": datetime.now().isoformat(),
                    "visit_count": 0,
                    "llm_analysis": analysis,
                    "discovered_actions": [],
                    "success_rate": 0.0
                }
                
                # 새로운 화면 발견 보상
                novelty_reward = self.intrinsic_motivation["novelty_bonus"]
                print(f"🆕 새로운 화면 유형 발견! (+{novelty_reward} 보상)")
            
            self.knowledge["screen_states"][screen_hash]["visit_count"] += 1
            return analysis
            
        except Exception as e:
            print(f"⚠️ 화면 분석 실패: {e}")
            return {"type": "unknown", "features": {}}
    
    def hash_screen(self, screen):
        """화면을 해시값으로 변환 (유사한 화면끼리 그룹핑)"""
        
        # 화면을 작은 크기로 리사이즈하여 해시 생성
        small_screen = cv2.resize(screen, (64, 48))
        gray = cv2.cvtColor(small_screen, cv2.COLOR_BGR2GRAY)
        
        # 간단한 특징 기반 해시
        features = [
            np.mean(gray),              # 평균 밝기
            np.std(gray),               # 밝기 편차
            len(np.unique(gray)),       # 색상 다양성
            cv2.Laplacian(gray, cv2.CV_64F).var()  # 텍스처
        ]
        
        feature_str = "_".join(f"{f:.2f}" for f in features)
        return hashlib.md5(feature_str.encode()).hexdigest()[:8]
    
    def generate_curious_action(self, current_analysis):
        """호기심 기반 행동 생성"""
        
        curiosity_prompt = f"""
당신은 이 게임을 처음 플레이하는 AI입니다.

현재 화면 분석:
{json.dumps(current_analysis, ensure_ascii=False, indent=2)}

다음 중 어떤 행동을 시도해보고 싶나요? 호기심과 탐험 정신으로 선택하세요:
0: 왼쪽 방향키
1: 오른쪽 방향키  
2: 위쪽 방향키
3: 아래쪽 방향키
4: Enter/확인키
5: Escape/취소키
6: 잠깐 기다리기

선택한 이유도 함께 설명해주세요.

JSON 형태로:
{{
    "action": 2,
    "reasoning": "위쪽에 뭔가 있어 보여서 탐험해보고 싶음",
    "curiosity_level": 0.8,
    "expected_outcome": "새로운 영역이나 정보 발견"
}}
"""
        
        try:
            response = self.call_llm(curiosity_prompt)
            decision = self.parse_llm_json(response)
            
            # 호기심 수준에 따른 보상
            curiosity = decision.get("curiosity_level", 0.5)
            if curiosity > self.curiosity_threshold:
                bonus = self.intrinsic_motivation["exploration_bonus"]
                print(f"🔍 높은 호기심 행동! (+{bonus} 보상)")
            
            return decision
            
        except Exception as e:
            print(f"⚠️ 호기심 행동 생성 실패: {e}")
            # 폴백: 랜덤 탐험
            return {
                "action": np.random.randint(0, 7),
                "reasoning": "랜덤 탐험 (LLM 실패)",
                "curiosity_level": 0.3
            }
    
    def learn_from_consequence(self, before_screen, action, after_screen, meta_info):
        """행동의 결과로부터 학습"""
        
        before_hash = self.hash_screen(before_screen)
        after_hash = self.hash_screen(after_screen)
        
        # 경험 기록
        experience = {
            "timestamp": time.time(),
            "before_state": before_hash,
            "action": action,
            "after_state": after_hash,
            "screen_changed": before_hash != after_hash,
            "meta_info": meta_info
        }
        
        self.experiences.append(experience)
        
        # 행동 결과 분석을 LLM에게 맡김
        learning_prompt = f"""
행동의 결과를 분석해주세요:

행동: {action} ({self.get_action_name(action)})
화면 변화: {"있음" if before_hash != after_hash else "없음"}
이전 화면 ID: {before_hash}
이후 화면 ID: {after_hash}

이 행동으로부터 무엇을 배울 수 있나요?

JSON 형태로:
{{
    "learned_rule": "배운 규칙이나 패턴",
    "effectiveness": 0.7,
    "new_concept": "새로 발견한 개념 (있다면)",
    "strategy_update": "전략 업데이트 사항"
}}
"""
        
        try:
            learning_response = self.call_llm(learning_prompt)
            learning_result = self.parse_llm_json(learning_response)
            
            # 새로운 개념 발견 시 지식 베이스 업데이트
            if learning_result.get("new_concept"):
                concept = learning_result["new_concept"]
                if concept not in self.knowledge["learned_concepts"]:
                    self.knowledge["learned_concepts"][concept] = {
                        "discovered_at": datetime.now().isoformat(),
                        "confidence": learning_result.get("effectiveness", 0.5),
                        "examples": []
                    }
                    
                    bonus = self.intrinsic_motivation["progress_bonus"]
                    print(f"💡 새로운 개념 발견: {concept} (+{bonus} 보상)")
            
            # 행동-결과 매핑 업데이트
            state_action = f"{before_hash}_{action}"
            if state_action not in self.knowledge["action_consequences"]:
                self.knowledge["action_consequences"][state_action] = []
            
            self.knowledge["action_consequences"][state_action].append({
                "result_state": after_hash,
                "effectiveness": learning_result.get("effectiveness", 0.5),
                "timestamp": time.time()
            })
            
            return learning_result
            
        except Exception as e:
            print(f"⚠️ 학습 실패: {e}")
            return None
    
    def detect_progress_patterns(self):
        """진전 패턴 자동 감지"""
        
        if len(self.experiences) < 10:
            return
        
        recent_experiences = list(self.experiences)[-10:]
        
        # 화면 변화 패턴 분석
        screen_changes = [exp["screen_changed"] for exp in recent_experiences]
        change_rate = sum(screen_changes) / len(screen_changes)
        
        # 새로운 화면 발견률
        unique_states = len(set(exp["after_state"] for exp in recent_experiences))
        novelty_rate = unique_states / len(recent_experiences)
        
        pattern_analysis_prompt = f"""
최근 10번의 행동 패턴을 분석해주세요:

화면 변화율: {change_rate:.2f} (1.0이 모든 행동에서 화면이 바뀜)
새로운 화면 비율: {novelty_rate:.2f}

이 패턴에서 어떤 진전이나 학습 신호를 발견할 수 있나요?

{{
    "progress_detected": true/false,
    "pattern_type": "탐험중/정체중/학습중",
    "recommendation": "다음 행동 권장사항"
}}
"""
        
        try:
            pattern_response = self.call_llm(pattern_analysis_prompt)
            pattern_result = self.parse_llm_json(pattern_response)
            
            if pattern_result.get("progress_detected"):
                bonus = self.intrinsic_motivation["consistency_bonus"]
                print(f"📈 학습 진전 감지! ({pattern_result.get('pattern_type')}) (+{bonus} 보상)")
                
                # 메타 전략 업데이트
                self.knowledge["meta_strategies"].append({
                    "timestamp": time.time(),
                    "pattern": pattern_result.get("pattern_type"),
                    "recommendation": pattern_result.get("recommendation")
                })
            
        except Exception as e:
            print(f"⚠️ 패턴 분석 실패: {e}")
    
    def call_llm(self, prompt, timeout=10):
        """LLM 호출"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,  # 창의적 사고를 위해 약간 높게
                "top_p": 0.9
            }
        }
        
        response = requests.post(self.llm_endpoint, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"LLM API 오류: {response.status_code}")
    
    def parse_llm_json(self, text):
        """LLM 응답에서 JSON 파싱"""
        try:
            # JSON 부분 추출
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start != -1 and end > start:
                json_text = text[start:end]
                return json.loads(json_text)
            else:
                return {}
        except:
            return {}
    
    def get_action_name(self, action_id):
        """행동 ID를 이름으로 변환"""
        actions = {
            0: "왼쪽", 1: "오른쪽", 2: "위쪽", 3: "아래쪽",
            4: "확인", 5: "취소", 6: "대기"
        }
        return actions.get(action_id, "알수없음")
    
    def execute_action(self, action_id):
        """행동 실행"""
        vk_keys = {
            0: self.base_seeker.VK_LEFT,
            1: self.base_seeker.VK_RIGHT,
            2: self.base_seeker.VK_UP,
            3: self.base_seeker.VK_DOWN,
            4: self.base_seeker.VK_RETURN,
            5: self.base_seeker.VK_ESCAPE,
            6: None  # 대기
        }
        
        vk_code = vk_keys.get(action_id)
        if vk_code is not None:
            return self.base_seeker.send_key_message(vk_code)
        
        return True
    
    def save_knowledge(self):
        """학습된 지식 저장"""
        knowledge_file = "self_learned_knowledge.pkl"
        
        try:
            with open(knowledge_file, 'wb') as f:
                pickle.dump(self.knowledge, f)
            
            print(f"🧠 지식 저장 완료: {knowledge_file}")
            print(f"   - 알려진 화면: {len(self.knowledge['screen_states'])}")
            print(f"   - 학습된 개념: {len(self.knowledge['learned_concepts'])}")
            print(f"   - 경험 수: {len(self.experiences)}")
            
        except Exception as e:
            print(f"❌ 지식 저장 실패: {e}")
    
    def autonomous_exploration(self, max_iterations=100):
        """완전 자율 탐험 및 학습"""
        
        print("🚀 완전 자율 학습 시작!")
        print("📋 규칙: AI가 스스로 게임을 탐험하고 학습합니다")
        print("🎯 목표: 아무것도 가르치지 않고 스스로 터득하게 하기")
        print()
        
        if not self.base_seeker.find_dosbox_window():
            print("❌ DOSBox 창을 찾을 수 없습니다!")
            return
        
        iteration = 0
        
        while iteration < max_iterations:
            try:
                print(f"\n--- 탐험 #{iteration + 1} ---")
                
                # 현재 화면 캡처
                current_screen = self.base_seeker.capture_dosbox_window()
                if current_screen is None:
                    print("⚠️ 화면 캡처 실패")
                    continue
                
                # 제로 지식 화면 분석
                screen_analysis = self.analyze_screen_with_zero_knowledge(current_screen)
                print(f"🔍 화면 분석: {screen_analysis.get('screen_type', '알수없음')}")
                
                # 호기심 기반 행동 결정
                action_decision = self.generate_curious_action(screen_analysis)
                action = action_decision["action"]
                reasoning = action_decision.get("reasoning", "")
                
                print(f"🤔 선택한 행동: {self.get_action_name(action)} - {reasoning}")
                
                # 행동 실행
                self.execute_action(action)
                
                # 잠시 대기 후 결과 관찰
                time.sleep(1)
                
                # 결과 화면 캡처
                result_screen = self.base_seeker.capture_dosbox_window()
                if result_screen is not None:
                    # 결과로부터 학습
                    meta_info = {
                        "iteration": iteration,
                        "reasoning": reasoning,
                        "curiosity_level": action_decision.get("curiosity_level", 0.5)
                    }
                    
                    learning_result = self.learn_from_consequence(
                        current_screen, action, result_screen, meta_info
                    )
                    
                    if learning_result:
                        learned_rule = learning_result.get("learned_rule", "")
                        if learned_rule:
                            print(f"💡 학습: {learned_rule}")
                
                # 주기적 패턴 분석
                if iteration % 10 == 0 and iteration > 0:
                    print(f"\n📊 {iteration}번 탐험 후 패턴 분석...")
                    self.detect_progress_patterns()
                    self.print_learning_summary()
                
                # 주기적 저장
                if iteration % 25 == 0 and iteration > 0:
                    self.save_knowledge()
                
                iteration += 1
                
                # 탐험률 조정 (시간이 지날수록 활용 증가)
                self.exploration_rate = max(0.1, self.exploration_rate * 0.995)
                
            except KeyboardInterrupt:
                print("\n⏹️ 사용자 중단")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")
                iteration += 1
        
        # 최종 저장
        self.save_knowledge()
        print("\n🎉 자율 학습 완료!")
        self.print_final_report()
    
    def print_learning_summary(self):
        """중간 학습 요약"""
        print("\n📚 현재 학습 상황:")
        print(f"   알려진 화면 유형: {len(self.knowledge['screen_states'])}")
        print(f"   발견한 개념: {len(self.knowledge['learned_concepts'])}")
        print(f"   축적된 경험: {len(self.experiences)}")
        
        if self.knowledge['learned_concepts']:
            print("   최근 발견한 개념들:")
            for concept, info in list(self.knowledge['learned_concepts'].items())[-3:]:
                confidence = info.get('confidence', 0)
                print(f"     - {concept} (신뢰도: {confidence:.2f})")
    
    def print_final_report(self):
        """최종 학습 보고서"""
        print("\n" + "="*50)
        print("🧠 완전 자율 학습 최종 보고서")
        print("="*50)
        
        print(f"📊 통계:")
        print(f"   - 발견한 화면 유형: {len(self.knowledge['screen_states'])}")
        print(f"   - 학습한 개념: {len(self.knowledge['learned_concepts'])}")
        print(f"   - 누적 경험: {len(self.experiences)}")
        print(f"   - 개발한 전략: {len(self.knowledge['meta_strategies'])}")
        
        print(f"\n💡 발견한 주요 개념들:")
        for concept, info in self.knowledge['learned_concepts'].items():
            confidence = info.get('confidence', 0)
            discovered_time = info.get('discovered_at', '알수없음')
            print(f"   - {concept} (신뢰도: {confidence:.2f}) - {discovered_time[:10]}")
        
        print(f"\n🎯 개발한 메타 전략:")
        for strategy in self.knowledge['meta_strategies'][-5:]:  # 최근 5개
            pattern = strategy.get('pattern', '알수없음')
            recommendation = strategy.get('recommendation', '')
            print(f"   - {pattern}: {recommendation}")


def main():
    """메인 실행 함수"""
    
    print("🧠 알파고 제로 스타일 완전 자율 학습 AI")
    print("🎯 목표: 아무것도 가르치지 않고 스스로 게임 터득하기")
    print()
    
    print("📋 준비사항:")
    print("1. DOSBox에서 영웅전설4 실행")
    print("2. LLM 모델 실행 (qwen2.5-coder:7b 권장)")
    print("3. 시간 여유 (학습에는 시간이 필요)")
    print()
    
    print("⚠️ 주의사항:")
    print("- 처음에는 무작위로 행동할 수 있습니다")
    print("- 점진적으로 패턴을 발견하고 학습합니다")
    print("- 학습된 지식은 자동 저장됩니다")
    print()
    
    try:
        iterations = input("탐험 횟수 (기본값 100): ").strip()
        max_iterations = int(iterations) if iterations else 100
        
        print(f"\n🚀 {max_iterations}번의 자율 탐험을 시작합니다!")
        print("시작하려면 Enter를 누르세요...")
        input()
        
        # 자율 학습 AI 실행
        ai = SelfLearningHeroAI()
        ai.autonomous_exploration(max_iterations=max_iterations)
        
    except KeyboardInterrupt:
        print("\n👋 자율 학습 AI 개발 화이팅!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()