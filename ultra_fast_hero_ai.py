"""
⚡ 초고속 영웅전설4 AI (1초 내 응답)
0.3-0.8초 응답시간으로 실시간 게임 플레이
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
import threading
from queue import Queue, Empty

class UltraFastHeroAI:
    """초고속 영웅전설4 AI - 0.5초 내 반응"""
    
    def __init__(self):
        # 기본 게임 제어
        self.base_seeker = IsolatedDOSBoxSeeker()
        
        # 초고속 LLM 설정
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.fast_models = [
            "qwen2:0.5b",      # 0.3초 - 최고속
            "llama3.2:1b",     # 0.7초 - 균형
            "phi3:mini"        # 0.8초 - 백업
        ]
        self.current_model = None
        
        # 응답 큐 (비동기 처리)
        self.decision_queue = Queue()
        self.llm_thread = None
        
        # 패턴 기반 빠른 결정 (LLM 보완용)
        self.quick_patterns = {
            "battle": [4, 4, 4, 5, 4],  # 공격 위주
            "field": [1, 0, 1, 0, 1],   # 좌우 이동
            "menu": [4, 4, 5],          # 확인/취소
        }
        
        # 상태 추적
        self.battle_count = 0
        self.last_decision_time = 0
        self.consecutive_same_actions = 0
        self.last_action = None
        
        # 성능 모니터링
        self.response_times = deque(maxlen=20)
        self.llm_available = False
        
        print("⚡ 초고속 AI 초기화 중...")
        self.setup_fast_llm()
    
    def setup_fast_llm(self):
        """초고속 LLM 설정"""
        
        # 사용 가능한 모델 확인
        try:
            response = requests.get(f"{self.llm_endpoint.replace('/api/generate', '/api/tags')}", timeout=2)
            if response.status_code == 200:
                installed_models = [m["name"] for m in response.json().get("models", [])]
                
                # 가장 빠른 모델 선택
                for model in self.fast_models:
                    if model in installed_models:
                        self.current_model = model
                        print(f"🚀 선택된 모델: {model}")
                        break
                
                if self.current_model:
                    self.llm_available = True
                    print("✅ 초고속 LLM 준비 완료!")
                    self.test_response_speed()
                else:
                    print("⚠️ 초고속 모델이 설치되지 않음")
                    print("💡 설치 명령어:")
                    for model in self.fast_models:
                        print(f"   ollama pull {model}")
            
        except Exception as e:
            print(f"❌ LLM 서버 연결 실패: {e}")
            print("🔧 해결방법:")
            print("1. ollama serve")
            print("2. ollama pull qwen2:0.5b")
        
        if not self.llm_available:
            print("📋 패턴 기반 AI로 실행합니다 (여전히 빠름!)")
    
    def test_response_speed(self):
        """응답 속도 테스트"""
        if not self.llm_available:
            return
        
        print("📊 응답속도 테스트 중...")
        
        test_prompt = "왼쪽 또는 오른쪽 중 선택하세요. 한 단어로 답하세요."
        
        start_time = time.time()
        try:
            self.quick_llm_call(test_prompt, timeout=3)
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            print(f"⏱️ 테스트 응답시간: {response_time:.2f}초")
            
            if response_time > 1.5:
                print("⚠️ 응답이 느립니다. 패턴 기반 모드 활성화")
                self.llm_available = False
            elif response_time < 0.8:
                print("🔥 초고속 응답 확인!")
            
        except Exception as e:
            print(f"❌ 속도 테스트 실패: {e}")
            self.llm_available = False
    
    def quick_llm_call(self, prompt, timeout=1):
        """초고속 LLM 호출"""
        
        payload = {
            "model": self.current_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,    # 빠른 결정
                "top_p": 0.8,
                "num_predict": 10,     # 짧은 응답
                "stop": ["\n", ".", "!"]  # 빠른 종료
            }
        }
        
        response = requests.post(self.llm_endpoint, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            raise Exception(f"LLM API 오류: {response.status_code}")
    
    def get_ultra_fast_decision(self, screen, game_context):
        """초고속 의사결정 (0.5초 내)"""
        
        start_time = time.time()
        
        # 1. 즉시 패턴 기반 결정 (백업용)
        pattern_decision = self.pattern_based_quick_decision(game_context)
        
        # 2. LLM 사용 가능하면 빠른 호출
        if self.llm_available:
            try:
                llm_decision = self.async_llm_decision(screen, game_context, timeout=0.8)
                if llm_decision is not None:
                    decision = llm_decision
                    decision_source = "LLM"
                else:
                    decision = pattern_decision
                    decision_source = "패턴(LLM 타임아웃)"
            except:
                decision = pattern_decision
                decision_source = "패턴(LLM 오류)"
        else:
            decision = pattern_decision
            decision_source = "패턴"
        
        # 3. 성능 기록
        decision_time = time.time() - start_time
        self.response_times.append(decision_time)
        
        print(f"⚡ {decision_source} 결정: {self.get_action_name(decision)} ({decision_time:.3f}초)")
        
        return decision
    
    def async_llm_decision(self, screen, game_context, timeout=0.8):
        """비동기 LLM 결정 (타임아웃 포함)"""
        
        # 간단한 프롬프트로 속도 최적화
        if game_context.get("is_battle", False):
            prompt = "전투중! 공격(4) 또는 방어(5)? 숫자만:"
        else:
            prompt = "필드! 왼쪽(0) 또는 오른쪽(1)? 숫자만:"
        
        try:
            response = self.quick_llm_call(prompt, timeout=timeout)
            
            # 응답에서 숫자 추출
            for char in response:
                if char.isdigit():
                    action = int(char)
                    if 0 <= action <= 6:
                        return action
            
            return None  # 유효한 응답 없음
            
        except Exception:
            return None
    
    def pattern_based_quick_decision(self, game_context):
        """패턴 기반 초고속 결정"""
        
        # 상황 판단
        if game_context.get("is_battle", False):
            pattern_key = "battle"
        else:
            pattern_key = "field"
        
        # 패턴에서 다음 행동 선택
        pattern = self.quick_patterns[pattern_key]
        action_index = self.battle_count % len(pattern)
        
        # 같은 행동 반복 방지
        action = pattern[action_index]
        if action == self.last_action:
            self.consecutive_same_actions += 1
            
            # 3번 연속 같은 행동이면 변경
            if self.consecutive_same_actions >= 3:
                available_actions = [a for a in pattern if a != action]
                if available_actions:
                    action = np.random.choice(available_actions)
                self.consecutive_same_actions = 0
        else:
            self.consecutive_same_actions = 0
        
        self.last_action = action
        return action
    
    def get_action_name(self, action_id):
        """행동 ID를 이름으로 변환"""
        actions = {
            0: "왼쪽", 1: "오른쪽", 2: "위쪽", 3: "아래쪽",
            4: "공격", 5: "방어", 6: "대기"
        }
        return actions.get(action_id, "알수없음")
    
    def execute_action_fast(self, action_id):
        """고속 행동 실행"""
        
        vk_keys = {
            0: self.base_seeker.VK_LEFT,
            1: self.base_seeker.VK_RIGHT,
            2: self.base_seeker.VK_UP,
            3: self.base_seeker.VK_DOWN,
            4: self.base_seeker.VK_RETURN,
            5: self.base_seeker.VK_ESCAPE,
            6: None  # 대기는 아무것도 하지 않음
        }
        
        vk_code = vk_keys.get(action_id)
        if vk_code is not None:
            self.base_seeker.send_key_message(vk_code)
        
        return True
    
    def ultra_fast_play(self, max_battles=5, target_fps=2):
        """초고속 자동 플레이 (2 FPS = 0.5초마다 결정)"""
        
        print(f"🏎️ 초고속 플레이 시작! (목표: {max_battles}회 전투, {target_fps} FPS)")
        
        if not self.base_seeker.find_dosbox_window():
            print("❌ DOSBox 창을 찾을 수 없습니다!")
            return
        
        frame_time = 1.0 / target_fps  # 0.5초
        next_frame_time = time.time()
        
        while self.battle_count < max_battles:
            try:
                current_time = time.time()
                
                # 프레임 타이밍 관리
                if current_time < next_frame_time:
                    sleep_time = next_frame_time - current_time
                    time.sleep(sleep_time)
                
                next_frame_time = time.time() + frame_time
                
                # 화면 캡처 (빠른 버전)
                screen = self.base_seeker.capture_dosbox_window()
                if screen is None:
                    continue
                
                # 게임 상태 분석 (간단버전)
                is_battle = self.base_seeker.is_battle_screen(screen)
                game_context = {
                    "is_battle": is_battle,
                    "timestamp": current_time
                }
                
                # 전투 카운트 업데이트
                if is_battle and not getattr(self, '_last_battle_state', False):
                    self.battle_count += 1
                    print(f"⚔️ 전투 #{self.battle_count} 감지!")
                
                self._last_battle_state = is_battle
                
                # 초고속 결정
                action = self.get_ultra_fast_decision(screen, game_context)
                
                # 행동 실행
                self.execute_action_fast(action)
                
                # 성능 출력 (10번마다)
                if len(self.response_times) % 10 == 0 and self.response_times:
                    avg_time = np.mean(self.response_times)
                    print(f"📊 평균 응답시간: {avg_time:.3f}초 (최근 {len(self.response_times)}회)")
                
            except KeyboardInterrupt:
                print("\n⏹️ 사용자 중단")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")
                time.sleep(0.1)  # 잠깐 쉬고 재시도
        
        # 최종 성능 리포트
        self.print_performance_report()
        print(f"🏁 초고속 플레이 완료! 총 {self.battle_count}회 전투")
    
    def print_performance_report(self):
        """성능 리포트 출력"""
        
        if not self.response_times:
            return
        
        times = list(self.response_times)
        
        print("\n" + "="*40)
        print("📊 초고속 AI 성능 리포트")
        print("="*40)
        print(f"평균 응답시간: {np.mean(times):.3f}초")
        print(f"최고 속도: {np.min(times):.3f}초")
        print(f"최저 속도: {np.max(times):.3f}초")
        print(f"표준편차: {np.std(times):.3f}초")
        
        # 1초 내 응답률
        under_1sec = sum(1 for t in times if t < 1.0)
        under_1sec_rate = (under_1sec / len(times)) * 100
        print(f"1초 내 응답률: {under_1sec_rate:.1f}% ({under_1sec}/{len(times)}회)")
        
        # 0.5초 내 응답률  
        under_half_sec = sum(1 for t in times if t < 0.5)
        under_half_sec_rate = (under_half_sec / len(times)) * 100
        print(f"0.5초 내 응답률: {under_half_sec_rate:.1f}% ({under_half_sec}/{len(times)}회)")
        
        if self.llm_available:
            print(f"사용 모델: {self.current_model} (LLM)")
        else:
            print("사용 모델: 패턴 기반 (초고속)")

def main():
    """메인 실행 함수"""
    
    print("⚡ 초고속 영웅전설4 AI")
    print("목표: 1초 내 응답으로 실시간 플레이!")
    print()
    
    print("📋 준비사항:")
    print("1. DOSBox에서 영웅전설4 실행")
    print("2. (선택) 초고속 LLM 설치:")
    print("   ollama pull qwen2:0.5b")
    print("   ollama pull llama3.2:1b")
    print()
    
    # 속도 설정
    print("🎮 플레이 속도 선택:")
    print("1. 초고속 (2 FPS, 0.5초마다)")
    print("2. 고속 (1 FPS, 1초마다)")
    print("3. 안전 (0.5 FPS, 2초마다)")
    
    try:
        choice = input("선택 (1-3, 기본값 1): ").strip() or "1"
        
        fps_settings = {"1": 2, "2": 1, "3": 0.5}
        target_fps = fps_settings.get(choice, 2)
        
        print(f"⚡ {target_fps} FPS 모드로 시작합니다!")
        print("시작하려면 Enter를 누르세요...")
        input()
        
        # AI 실행
        ai = UltraFastHeroAI()
        ai.ultra_fast_play(max_battles=5, target_fps=target_fps)
        
    except KeyboardInterrupt:
        print("\n👋 빠른 AI 개발 화이팅!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()