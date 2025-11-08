"""
영웅전설4 적응형 AI 플레이어
로컬 LLM + 강화학습을 결합한 자율 학습 시스템
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from isolated_seeker import IsolatedDOSBoxSeeker
import time
import json
import cv2
import numpy as np
from collections import deque
import requests

class AdaptiveHeroAI:
    """적응형 영웅전설4 AI"""
    
    def __init__(self):
        # 기본 게임 제어
        self.base_seeker = IsolatedDOSBoxSeeker()
        
        # 학습 시스템
        self.experience_db = []
        self.success_patterns = {}
        self.failure_patterns = {}
        
        # LLM 설정 (Ollama 기본)
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.llm_model = "llama3.2"  # 또는 "deepseek-coder"
        
        # 게임 상태 추적
        self.game_memory = deque(maxlen=100)
        self.battle_count = 0
        self.learning_enabled = True
        
        # 행동 매핑
        self.actions = {
            0: ("move_left", "왼쪽 이동"),
            1: ("move_right", "오른쪽 이동"),
            2: ("move_up", "위쪽 이동"), 
            3: ("move_down", "아래쪽 이동"),
            4: ("attack", "공격/확인"),
            5: ("defend", "방어/취소"),
            6: ("wait", "대기"),
        }
        
        print("🤖 적응형 AI 초기화 완료!")
        self.check_llm_connection()
    
    def check_llm_connection(self):
        """LLM 연결 확인"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [m["name"] for m in models]
                print(f"✅ LLM 서버 연결됨. 사용 가능한 모델: {available_models}")
                
                # 모델 자동 선택
                if "deepseek-coder" in available_models:
                    self.llm_model = "deepseek-coder"
                elif "llama3.2" in available_models:
                    self.llm_model = "llama3.2"
                else:
                    self.llm_model = available_models[0] if available_models else "llama3.2"
                
                print(f"🧠 선택된 모델: {self.llm_model}")
                return True
        except:
            print("❌ LLM 서버에 연결할 수 없습니다. 기본 AI로 실행합니다.")
            print("   Ollama 설치 후 'ollama pull llama3.2' 실행하세요.")
        
        return False
    
    def analyze_with_llm(self, screen_description, game_context):
        """LLM으로 상황 분석"""
        try:
            prompt = f"""
당신은 영웅전설4를 플레이하는 전문 AI입니다.

현재 상황:
{screen_description}

게임 컨텍스트:
- 전투 횟수: {self.battle_count}
- 최근 행동: {self.get_recent_actions()}
- 학습된 패턴: {len(self.success_patterns)}개

다음 중 최적의 행동을 하나만 선택하세요:
0: 왼쪽 이동
1: 오른쪽 이동  
2: 위쪽 이동
3: 아래쪽 이동
4: 공격/확인
5: 방어/취소
6: 대기

숫자만 답하세요 (0-6):
"""
            
            response = requests.post(self.llm_endpoint, json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            }, timeout=5)
            
            if response.status_code == 200:
                answer = response.json()["response"].strip()
                
                # 숫자 추출
                for char in answer:
                    if char.isdigit() and int(char) < len(self.actions):
                        action_id = int(char)
                        print(f"🧠 LLM 결정: {self.actions[action_id][1]} ({action_id})")
                        return action_id
        
        except Exception as e:
            print(f"⚠ LLM 분석 실패: {e}")
        
        # 폴백: 패턴 기반 결정
        return self.pattern_based_decision(game_context)
    
    def pattern_based_decision(self, game_context):
        """학습된 패턴 기반 결정"""
        
        # 성공 패턴 활용
        context_key = self.get_context_key(game_context)
        
        if context_key in self.success_patterns:
            best_action = max(self.success_patterns[context_key].items(), 
                            key=lambda x: x[1])
            print(f"📚 패턴 기반 결정: {self.actions[best_action[0]][1]}")
            return best_action[0]
        
        # 기본 탐험 행동
        if game_context.get("is_battle", False):
            return 4  # 전투 시 공격
        else:
            return np.random.choice([0, 1])  # 필드에서 좌우 이동
    
    def get_context_key(self, game_context):
        """게임 상황을 키로 변환"""
        keys = []
        
        if game_context.get("is_battle", False):
            keys.append("battle")
        else:
            keys.append("field")
        
        if game_context.get("enemy_count", 0) > 0:
            keys.append(f"enemies_{game_context['enemy_count']}")
        
        return "_".join(keys)
    
    def get_recent_actions(self):
        """최근 행동 요약"""
        if len(self.game_memory) < 3:
            return "시작"
        
        recent = list(self.game_memory)[-3:]
        action_names = [self.actions.get(action, ["unknown"])[0] 
                       for action in recent if isinstance(action, int)]
        return " → ".join(action_names)
    
    def learn_from_result(self, action, game_state_before, game_state_after):
        """결과로부터 학습"""
        if not self.learning_enabled:
            return
        
        # 성공/실패 판단
        success = self.evaluate_success(game_state_before, game_state_after, action)
        
        # 컨텍스트 생성
        context_key = self.get_context_key(game_state_before)
        
        # 패턴 업데이트
        if success:
            if context_key not in self.success_patterns:
                self.success_patterns[context_key] = {}
            
            if action not in self.success_patterns[context_key]:
                self.success_patterns[context_key][action] = 0
            
            self.success_patterns[context_key][action] += 1
            print(f"✅ 학습: {context_key} -> {self.actions[action][1]} (성공)")
        
        else:
            if context_key not in self.failure_patterns:
                self.failure_patterns[context_key] = {}
            
            if action not in self.failure_patterns[context_key]:
                self.failure_patterns[context_key][action] = 0
            
            self.failure_patterns[context_key][action] += 1
            print(f"❌ 학습: {context_key} -> {self.actions[action][1]} (실패)")
        
        # 경험 저장
        experience = {
            "timestamp": time.time(),
            "action": action,
            "state_before": game_state_before,
            "state_after": game_state_after,
            "success": success
        }
        self.experience_db.append(experience)
    
    def evaluate_success(self, state_before, state_after, action):
        """행동의 성공/실패 평가"""
        
        # 전투 발견은 성공
        if not state_before.get("is_battle") and state_after.get("is_battle"):
            return True
        
        # HP 감소는 실패 (전투 중)
        hp_before = state_before.get("hp", 100)
        hp_after = state_after.get("hp", 100)
        if hp_before > hp_after:
            return False
        
        # 새로운 화면 탐험은 성공
        if self.is_screen_changed(state_before, state_after):
            return True
        
        # 기본적으로는 중립
        return True
    
    def is_screen_changed(self, state_before, state_after):
        """화면 변화 감지"""
        # 간단한 구현 - 실제로는 더 정교한 이미지 비교 필요
        return np.random.random() > 0.7
    
    def get_screen_description(self, screen):
        """화면을 텍스트로 설명"""
        if screen is None:
            return "화면을 캡처할 수 없음"
        
        # 화면 분석
        is_battle = self.base_seeker.is_battle_screen(screen)
        
        description = []
        if is_battle:
            description.append("전투 화면 감지")
            description.append("HP/MP 바가 보임")
        else:
            description.append("필드 화면")
            description.append("캐릭터가 이동 가능한 상태")
        
        # 화면 밝기 분석
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        description.append(f"화면 밝기: {brightness:.1f}")
        
        return ". ".join(description)
    
    def save_learning_progress(self):
        """학습 진행상황 저장"""
        data = {
            "battle_count": self.battle_count,
            "success_patterns": self.success_patterns,
            "failure_patterns": self.failure_patterns,
            "experience_count": len(self.experience_db)
        }
        
        with open("ai_learning_progress.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 학습 데이터 저장 완료 (경험: {len(self.experience_db)}개)")
    
    def load_learning_progress(self):
        """이전 학습 데이터 로드"""
        try:
            with open("ai_learning_progress.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.battle_count = data.get("battle_count", 0)
            self.success_patterns = data.get("success_patterns", {})
            self.failure_patterns = data.get("failure_patterns", {})
            
            print(f"📁 이전 학습 데이터 로드 (전투: {self.battle_count}회, 패턴: {len(self.success_patterns)}개)")
        
        except FileNotFoundError:
            print("📝 새로운 학습 시작")
    
    def adaptive_play(self, max_battles=5):
        """적응형 자동 플레이"""
        
        print(f"🎮 적응형 AI 플레이 시작! (목표: {max_battles}회 전투)")
        
        # 이전 학습 데이터 로드
        self.load_learning_progress()
        
        if not self.base_seeker.find_dosbox_window():
            print("❌ DOSBox 창을 찾을 수 없습니다!")
            return
        
        consecutive_failures = 0
        
        while self.battle_count < max_battles:
            try:
                # 현재 화면 분석
                screen = self.base_seeker.capture_dosbox_window()
                if screen is None:
                    print("⚠ 화면 캡처 실패")
                    consecutive_failures += 1
                    if consecutive_failures > 5:
                        break
                    continue
                
                consecutive_failures = 0
                
                # 게임 상태 분석
                game_state = {
                    "is_battle": self.base_seeker.is_battle_screen(screen),
                    "screen": screen,
                    "timestamp": time.time()
                }
                
                # 전투 카운트 업데이트
                if game_state["is_battle"] and not getattr(self, "_last_battle_state", False):
                    self.battle_count += 1
                    print(f"⚔ 전투 #{self.battle_count} 감지!")
                
                self._last_battle_state = game_state["is_battle"]
                
                # 화면 설명 생성
                screen_description = self.get_screen_description(screen)
                
                # AI 결정 (LLM 우선, 폴백은 패턴 기반)
                action = self.analyze_with_llm(screen_description, game_state)
                
                # 행동 실행
                self.execute_action(action)
                
                # 결과 관찰 및 학습
                time.sleep(0.5)  # 반응 시간
                
                new_screen = self.base_seeker.capture_dosbox_window()
                new_game_state = {
                    "is_battle": self.base_seeker.is_battle_screen(new_screen) if new_screen is not None else False,
                    "screen": new_screen,
                    "timestamp": time.time()
                }
                
                # 학습
                self.learn_from_result(action, game_state, new_game_state)
                
                # 메모리에 추가
                self.game_memory.append(action)
                
                # 주기적 저장 (5번마다)
                if len(self.experience_db) % 5 == 0:
                    self.save_learning_progress()
                
                time.sleep(1)  # 다음 행동까지 대기
                
            except KeyboardInterrupt:
                print("\n⏹ 사용자 중단")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                consecutive_failures += 1
                if consecutive_failures > 3:
                    break
        
        # 최종 저장
        self.save_learning_progress()
        print(f"🏁 AI 플레이 완료! 총 {self.battle_count}회 전투")
        self.print_learning_summary()
    
    def execute_action(self, action_id):
        """행동 실행"""
        action_name = self.actions[action_id][0]
        
        if action_name.startswith("move_"):
            direction = action_name.split("_")[1]
            if direction == "left":
                self.base_seeker.send_key_message(self.base_seeker.VK_LEFT)
            elif direction == "right":
                self.base_seeker.send_key_message(self.base_seeker.VK_RIGHT)
            elif direction == "up":
                self.base_seeker.send_key_message(self.base_seeker.VK_UP)
            elif direction == "down":
                self.base_seeker.send_key_message(self.base_seeker.VK_DOWN)
        
        elif action_name == "attack":
            self.base_seeker.send_key_message(self.base_seeker.VK_RETURN)
        
        elif action_name == "defend":
            self.base_seeker.send_key_message(self.base_seeker.VK_ESCAPE)
        
        # wait은 아무것도 하지 않음
        
        print(f"🎯 실행: {self.actions[action_id][1]}")
    
    def print_learning_summary(self):
        """학습 요약 출력"""
        print("\n📊 학습 요약:")
        print(f"   총 경험: {len(self.experience_db)}개")
        print(f"   성공 패턴: {len(self.success_patterns)}개")
        print(f"   실패 패턴: {len(self.failure_patterns)}개")
        
        if self.success_patterns:
            print("\n🏆 효과적인 패턴:")
            for context, actions in list(self.success_patterns.items())[:3]:
                best_action = max(actions.items(), key=lambda x: x[1])
                action_name = self.actions[best_action[0]][1]
                print(f"   {context}: {action_name} ({best_action[1]}회 성공)")


def main():
    """메인 실행 함수"""
    
    print("🚀 영웅전설4 적응형 AI 시작!")
    print("\n필요사항:")
    print("1. DOSBox에서 영웅전설4 실행")
    print("2. (선택) Ollama LLM 서버 실행")
    print("   - 설치: https://ollama.ai/")
    print("   - 모델: ollama pull llama3.2")
    print("\n시작하려면 Enter를 누르세요...")
    input()
    
    ai = AdaptiveHeroAI()
    ai.adaptive_play(max_battles=5)

if __name__ == "__main__":
    main()