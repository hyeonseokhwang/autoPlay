"""
LLM 기반 적응형 AI 플레이어 설계안
로컬 LLM (DeepSeek, Ollama 등)을 활용한 자율 학습 게임 AI
"""

import json
import time
import requests
import base64
from datetime import datetime
from collections import deque

class LLMGameAI:
    def __init__(self, llm_endpoint="http://localhost:11434/api/generate"):
        """
        LLM 기반 게임 AI 초기화
        
        Args:
            llm_endpoint: 로컬 LLM API 엔드포인트 (Ollama 기본)
        """
        self.llm_endpoint = llm_endpoint
        self.model_name = "deepseek-coder"  # 또는 "llama3.2", "qwen2" 등
        
        # 경험 메모리
        self.experience_memory = deque(maxlen=1000)
        self.success_patterns = []
        self.failure_patterns = []
        
        # 게임 상태 히스토리
        self.game_history = []
        self.current_strategy = "exploration"
        
        # 학습된 지식
        self.learned_knowledge = {
            "battle_indicators": [],
            "effective_moves": {},
            "danger_signs": [],
            "success_sequences": []
        }
    
    def analyze_screen_with_llm(self, image, game_state):
        """
        LLM에게 현재 화면 상황을 분석하게 함
        """
        # 이미지를 base64로 인코딩
        image_b64 = self.encode_image_to_base64(image)
        
        prompt = f"""
        당신은 영웅전설4 게임을 플레이하는 전문 AI입니다.
        
        현재 게임 상태:
        - 씬 타입: {game_state.get('scene_type', 'unknown')}
        - 전투 상황: {game_state.get('is_battle', False)}
        - 캐릭터 발견: {game_state.get('character', {}).get('found', False)}
        - 적 수: {len(game_state.get('enemies', []))}
        
        과거 경험:
        {self.get_relevant_experience()}
        
        현재 화면을 분석하고 다음 중 최적의 행동을 선택하세요:
        1. move_left - 왼쪽으로 이동
        2. move_right - 오른쪽으로 이동  
        3. move_up - 위로 이동
        4. move_down - 아래로 이동
        5. attack - 공격/확인
        6. defend - 방어/취소
        7. wait - 대기
        8. retreat - 후퇴
        
        응답 형식 (JSON):
        {{
            "action": "선택한 행동",
            "reasoning": "판단 근거",
            "confidence": 0.8,
            "expected_outcome": "예상 결과",
            "learning_note": "이 상황에서 배운 점"
        }}
        """
        
        try:
            response = self.call_llm(prompt, image_b64)
            return self.parse_llm_response(response)
        except Exception as e:
            print(f"LLM 분석 실패: {e}")
            return self.fallback_decision(game_state)
    
    def call_llm(self, prompt, image_b64=None):
        """로컬 LLM API 호출"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # 일관된 결정을 위해 낮게
                "top_p": 0.9
            }
        }
        
        if image_b64:
            payload["images"] = [image_b64]
        
        response = requests.post(self.llm_endpoint, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"LLM API 오류: {response.status_code}")
    
    def learn_from_outcome(self, action, game_state_before, game_state_after, success):
        """
        행동 결과를 바탕으로 학습
        """
        experience = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "state_before": game_state_before,
            "state_after": game_state_after,
            "success": success,
            "context": self.get_game_context(game_state_before)
        }
        
        self.experience_memory.append(experience)
        
        # 성공/실패 패턴 업데이트
        if success:
            self.success_patterns.append({
                "context": experience["context"],
                "action": action,
                "outcome": "positive"
            })
        else:
            self.failure_patterns.append({
                "context": experience["context"], 
                "action": action,
                "outcome": "negative"
            })
        
        # 지식 베이스 업데이트
        self.update_knowledge_base(experience)
    
    def update_knowledge_base(self, experience):
        """경험을 바탕으로 지식 베이스 업데이트"""
        
        # 전투 지표 학습
        if experience["state_after"].get("is_battle") and not experience["state_before"].get("is_battle"):
            battle_trigger = {
                "action": experience["action"],
                "context": experience["context"]
            }
            self.learned_knowledge["battle_indicators"].append(battle_trigger)
        
        # 효과적인 이동 패턴 학습
        if experience["success"]:
            move_key = f"{experience['context']}_{experience['action']}"
            if move_key not in self.learned_knowledge["effective_moves"]:
                self.learned_knowledge["effective_moves"][move_key] = 0
            self.learned_knowledge["effective_moves"][move_key] += 1
    
    def get_relevant_experience(self):
        """현재 상황과 관련된 과거 경험 조회"""
        recent_experiences = list(self.experience_memory)[-10:]  # 최근 10개
        
        summary = "최근 경험:\n"
        for exp in recent_experiences:
            summary += f"- {exp['action']} → {'성공' if exp['success'] else '실패'}\n"
        
        return summary
    
    def get_game_context(self, game_state):
        """게임 상태를 컨텍스트로 변환"""
        context_parts = []
        
        if game_state.get("is_battle"):
            context_parts.append("battle")
        if game_state.get("is_field"):
            context_parts.append("field")
        
        enemy_count = len(game_state.get("enemies", []))
        if enemy_count > 0:
            context_parts.append(f"enemies_{enemy_count}")
        
        return "_".join(context_parts) if context_parts else "neutral"
    
    def encode_image_to_base64(self, image):
        """이미지를 base64로 인코딩"""
        import cv2
        
        # 이미지 크기 조정 (LLM 처리 속도 향상)
        resized = cv2.resize(image, (320, 240))
        
        # PNG 포맷으로 인코딩
        _, buffer = cv2.imencode('.png', resized)
        
        # base64 인코딩
        return base64.b64encode(buffer).decode('utf-8')
    
    def parse_llm_response(self, response):
        """LLM 응답 파싱"""
        try:
            # JSON 응답 파싱 시도
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        # JSON 파싱 실패 시 텍스트에서 액션 추출
        return self.extract_action_from_text(response)
    
    def extract_action_from_text(self, text):
        """텍스트에서 행동 추출"""
        actions = ["move_left", "move_right", "move_up", "move_down", 
                  "attack", "defend", "wait", "retreat"]
        
        for action in actions:
            if action in text.lower():
                return {
                    "action": action,
                    "reasoning": "텍스트에서 추출됨",
                    "confidence": 0.5
                }
        
        return self.fallback_decision({})
    
    def fallback_decision(self, game_state):
        """LLM 실패 시 폴백 결정"""
        if game_state.get("is_battle"):
            return {"action": "attack", "reasoning": "전투 중 기본 행동"}
        else:
            return {"action": "move_right", "reasoning": "탐험 중 기본 이동"}


# 사용 예시
def integrate_llm_with_seeker():
    """기존 탐색기에 LLM 통합"""
    
    class LLMEnhancedSeeker:
        def __init__(self):
            from isolated_seeker import IsolatedDOSBoxSeeker
            
            self.base_seeker = IsolatedDOSBoxSeeker()
            self.llm_ai = LLMGameAI()
            
        def enhanced_decision_making(self):
            """LLM 강화된 의사결정"""
            
            # 화면 캡처
            screen = self.base_seeker.capture_dosbox_window()
            game_state = {"screen": screen}  # 실제로는 더 자세한 분석 필요
            
            # LLM에게 분석 요청
            decision = self.llm_ai.analyze_screen_with_llm(screen, game_state)
            
            # 결정 실행
            action_result = self.execute_decision(decision)
            
            # 결과 학습
            new_game_state = {"screen": self.base_seeker.capture_dosbox_window()}
            self.llm_ai.learn_from_outcome(
                decision["action"], 
                game_state, 
                new_game_state, 
                action_result
            )
        
        def execute_decision(self, decision):
            """LLM 결정 실행"""
            action = decision["action"]
            
            if action.startswith("move_"):
                direction = action.split("_")[1]
                return self.base_seeker.move_in_direction(direction)
            elif action == "attack":
                return self.base_seeker.send_key_message(self.base_seeker.VK_RETURN)
            # 기타 액션들...
            
            return True

if __name__ == "__main__":
    print("LLM 기반 적응형 게임 AI 설계 완료!")
    print("\n필요한 구성요소:")
    print("1. 로컬 LLM 서버 (Ollama + DeepSeek)")
    print("2. 이미지 분석 기능")
    print("3. 경험 학습 시스템") 
    print("4. 지식 베이스 관리")