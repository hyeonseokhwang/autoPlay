"""
전투 탐색 AI 플레이어
오픈월드에서 탐험하며 전투를 찾는 특화 버전
"""

import sys
import os
import time
import random

# 경로 추가
sys.path.append('G:/LucasAI')

from src.screen_capture import ScreenCapture
from src.game_vision import GameVision
from src.input_controller import InputController


class BattleSeeker:
    def __init__(self):
        """전투 탐색 AI 초기화"""
        self.screen_capture = ScreenCapture()
        self.game_vision = GameVision()
        self.input_controller = InputController()
        
        # 상태 변수
        self.battles_found = 0
        self.target_battles = 5
        self.current_direction = "right"
        self.direction_time = 0
        self.last_direction_change = time.time()
        self.move_duration = 3  # 3초씩 한 방향으로 이동
        
        # 탐험 통계
        self.total_exploration_time = 0
        self.field_movements = 0
        self.battle_encounters = 0
        
        print("영웅전설4 전투 탐색 AI 초기화 완료!")
    
    def setup(self):
        """초기 설정"""
        if not self.screen_capture.find_window():
            print("❌ DOSBox 윈도우를 찾을 수 없습니다.")
            return False
        
        print("✓ DOSBox 윈도우 발견")
        self.input_controller.set_target_window(self.screen_capture.hwnd)
        return True
    
    def analyze_current_screen(self):
        """현재 화면 분석"""
        image = self.screen_capture.capture_screen()
        if image is None:
            return None
        
        game_state = self.game_vision.analyze_game_state(image)
        return game_state
    
    def explore_field(self):
        """필드에서 탐험하기"""
        current_time = time.time()
        
        # 일정 시간마다 방향 전환
        if current_time - self.last_direction_change > self.move_duration:
            self.change_direction()
            self.last_direction_change = current_time
        
        # 현재 방향으로 이동
        if self.current_direction == "left":
            print("← 왼쪽으로 탐험 중...")
            self.input_controller.move_left(0.2)
        elif self.current_direction == "right":
            print("→ 오른쪽으로 탐험 중...")
            self.input_controller.move_right(0.2)
        elif self.current_direction == "up":
            print("↑ 위쪽으로 탐험 중...")
            self.input_controller.move_up(0.2)
        elif self.current_direction == "down":
            print("↓ 아래쪽으로 탐험 중...")
            self.input_controller.move_down(0.2)
        
        self.field_movements += 1
        time.sleep(0.1)
    
    def change_direction(self):
        """방향 변경"""
        directions = ["left", "right", "up", "down"]
        # 현재 방향과 반대 방향을 우선적으로 선택
        if self.current_direction == "left":
            preferred = ["right", "up", "down"]
        elif self.current_direction == "right":
            preferred = ["left", "up", "down"]
        elif self.current_direction == "up":
            preferred = ["down", "left", "right"]
        elif self.current_direction == "down":
            preferred = ["up", "left", "right"]
        else:
            preferred = directions
        
        # 70% 확률로 우선 방향, 30% 확률로 랜덤
        if random.random() < 0.7:
            self.current_direction = random.choice(preferred)
        else:
            self.current_direction = random.choice(directions)
        
        print(f"🔄 방향 전환: {self.current_direction}")
        
        # 이동 시간도 랜덤으로 조정 (2-5초)
        self.move_duration = random.uniform(2, 5)
    
    def handle_battle(self):
        """전투 상황 처리"""
        print(f"⚔️ 전투 발견! ({self.battles_found + 1}/{self.target_battles})")
        self.battle_encounters += 1
        
        # 전투 중 대기 (실제로는 전투 로직 구현 가능)
        battle_start = time.time()
        
        while True:
            game_state = self.analyze_current_screen()
            if game_state is None:
                time.sleep(0.5)
                continue
            
            # 전투가 끝났는지 확인
            if not game_state['is_battle']:
                battle_duration = time.time() - battle_start
                print(f"✅ 전투 종료! (지속시간: {battle_duration:.1f}초)")
                self.battles_found += 1
                break
            
            # 전투 중 임시 행동 (Enter 키로 진행)
            print("🗡️ 전투 진행 중...")
            self.input_controller.confirm()
            time.sleep(2)
            
            # 너무 오래 걸리면 ESC로 탈출 시도
            if time.time() - battle_start > 30:
                print("⏰ 전투가 너무 길어서 탈출 시도...")
                self.input_controller.cancel()
                time.sleep(1)
                break
    
    def run(self):
        """메인 실행 루프"""
        if not self.setup():
            return False
        
        print(f"\n🎯 목표: {self.target_battles}번의 전투 경험하기")
        print("탐험을 시작합니다...\n")
        
        start_time = time.time()
        
        while self.battles_found < self.target_battles:
            try:
                # 현재 화면 분석
                game_state = self.analyze_current_screen()
                
                if game_state is None:
                    print("화면 분석 실패, 재시도...")
                    time.sleep(1)
                    continue
                
                # 전투 화면인지 확인
                if game_state['is_battle']:
                    self.handle_battle()
                
                # 필드 화면에서 탐험
                elif game_state['is_field']:
                    self.explore_field()
                
                # 기타 상황
                else:
                    print("🤔 알 수 없는 화면, 랜덤 이동...")
                    direction = random.choice(["left", "right", "up", "down"])
                    self.input_controller.move_direction(direction, 0.5)
                    time.sleep(0.5)
                
                # 진행 상황 출력 (10초마다)
                if int(time.time() - start_time) % 10 == 0 and time.time() - start_time > 0:
                    elapsed = time.time() - start_time
                    print(f"📊 진행상황 - 전투: {self.battles_found}/{self.target_battles}, "
                          f"탐험시간: {elapsed:.0f}초, 이동횟수: {self.field_movements}")
                
            except KeyboardInterrupt:
                print("\n사용자가 중단했습니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")
                time.sleep(1)
        
        # 결과 출력
        total_time = time.time() - start_time
        print(f"\n🎉 탐험 완료!")
        print(f"총 {self.battles_found}번의 전투를 경험했습니다.")
        print(f"총 탐험 시간: {total_time:.1f}초")
        print(f"총 이동 횟수: {self.field_movements}")
        print(f"전투 조우율: {self.battle_encounters}회")
        
        return True


# 메인 실행
if __name__ == "__main__":
    print("=" * 50)
    print("   영웅전설4 전투 탐색 AI")
    print("=" * 50)
    
    battle_seeker = BattleSeeker()
    
    print("5초 후 탐험을 시작합니다...")
    print("중지하려면 Ctrl+C를 누르세요.")
    
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    success = battle_seeker.run()
    
    if success:
        print("\n프로그램을 종료합니다.")
    else:
        print("\n설정 실패로 프로그램을 종료합니다.")