"""
AI 플레이어 모듈
게임 상황을 분석하고 최적의 행동을 결정합니다.
"""

import time
import random
import json
from enum import Enum


class GameState(Enum):
    """게임 상태 열거형"""
    EXPLORING = "exploring"
    FIGHTING = "fighting"
    IN_MENU = "in_menu"
    IN_DIALOG = "in_dialog"
    MOVING_TO_TARGET = "moving_to_target"
    IDLE = "idle"


class AIPlayer:
    def __init__(self, screen_capture, game_vision, input_controller, config_path="config/settings.json"):
        """
        AI 플레이어 초기화
        
        Args:
            screen_capture: 화면 캡처 객체
            game_vision: 게임 비전 분석 객체
            input_controller: 입력 제어 객체
            config_path: 설정 파일 경로
        """
        self.screen_capture = screen_capture
        self.game_vision = game_vision
        self.input_controller = input_controller
        
        # 설정 로드
        self.config = self._load_config(config_path)
        self.ai_settings = self.config.get("ai_settings", {})
        
        # AI 설정값들
        self.decision_delay = self.ai_settings.get("decision_delay", 0.5)
        self.exploration_priority = self.ai_settings.get("exploration_priority", 0.7)
        self.combat_priority = self.ai_settings.get("combat_priority", 0.9)
        self.safe_mode = self.ai_settings.get("safe_mode", True)
        
        # 상태 변수들
        self.current_state = GameState.IDLE
        self.last_position = None
        self.target_position = None
        self.last_action_time = 0
        self.stuck_counter = 0
        self.exploration_map = {}
        
        # 행동 통계
        self.stats = {
            'total_actions': 0,
            'successful_moves': 0,
            'battles_won': 0,
            'items_collected': 0,
            'exploration_progress': 0
        }
        
        # 실행 플래그
        self.running = False
    
    def _load_config(self, config_path):
        """설정 파일을 로드합니다."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return {}
    
    def start(self):
        """AI 플레이어를 시작합니다."""
        print("AI 플레이어 시작...")
        self.running = True
        
        # 입력 컨트롤러에 윈도우 설정
        if hasattr(self.screen_capture, 'hwnd'):
            self.input_controller.set_target_window(self.screen_capture.hwnd)
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\nAI 플레이어 중지됨 (사용자 중단)")
        except Exception as e:
            print(f"\nAI 플레이어 오류: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """AI 플레이어를 중지합니다."""
        print("AI 플레이어 중지 중...")
        self.running = False
        self.input_controller.emergency_stop()
        self._print_stats()
    
    def _main_loop(self):
        """메인 게임 루프"""
        while self.running:
            try:
                # 화면 캡처
                image = self.screen_capture.capture_screen()
                if image is None:
                    print("화면 캡처 실패")
                    time.sleep(1)
                    continue
                
                # 게임 상태 분석
                game_state = self.game_vision.analyze_game_state(image)
                
                # 상태 업데이트
                self._update_state(game_state)
                
                # 행동 결정 및 실행
                action = self._decide_action(game_state)
                if action:
                    self._execute_action(action)
                    self.stats['total_actions'] += 1
                
                # 지연 시간
                time.sleep(self.decision_delay)
                
            except Exception as e:
                print(f"메인 루프 오류: {e}")
                time.sleep(1)
    
    def _update_state(self, game_state):
        """게임 상태를 바탕으로 AI 상태를 업데이트합니다."""
        scene_type = game_state.get('scene_type', 'unknown')
        ui_state = game_state.get('ui', {})
        
        # 메뉴 상태 확인
        if ui_state.get('menu_open') or ui_state.get('dialog_open'):
            self.current_state = GameState.IN_MENU if ui_state.get('menu_open') else GameState.IN_DIALOG
        # 전투 상태 확인
        elif len(game_state.get('enemies', [])) > 0:
            self.current_state = GameState.FIGHTING
        # 이동 중인지 확인
        elif self.target_position is not None:
            self.current_state = GameState.MOVING_TO_TARGET
        # 기본 탐험 상태
        else:
            self.current_state = GameState.EXPLORING
        
        # 위치 업데이트
        character_info = game_state.get('character', {})
        if character_info.get('found') and character_info.get('position'):
            current_pos = character_info['position']
            
            # 이동 성공 체크
            if self.last_position and current_pos != self.last_position:
                self.stats['successful_moves'] += 1
                self.stuck_counter = 0
            elif self.last_position == current_pos:
                self.stuck_counter += 1
            
            self.last_position = current_pos
    
    def _decide_action(self, game_state):
        """현재 상태를 바탕으로 다음 행동을 결정합니다."""
        
        if self.current_state == GameState.IN_MENU:
            return self._decide_menu_action(game_state)
        elif self.current_state == GameState.IN_DIALOG:
            return self._decide_dialog_action(game_state)
        elif self.current_state == GameState.FIGHTING:
            return self._decide_combat_action(game_state)
        elif self.current_state == GameState.MOVING_TO_TARGET:
            return self._decide_movement_action(game_state)
        elif self.current_state == GameState.EXPLORING:
            return self._decide_exploration_action(game_state)
        else:
            return self._decide_idle_action(game_state)
    
    def _decide_menu_action(self, game_state):
        """메뉴 상태에서의 행동 결정"""
        # 일단 메뉴를 닫습니다
        return {'type': 'key', 'key': 'cancel'}
    
    def _decide_dialog_action(self, game_state):
        """대화 상태에서의 행동 결정"""
        # 대화를 진행합니다
        return {'type': 'key', 'key': 'confirm'}
    
    def _decide_combat_action(self, game_state):
        """전투 상태에서의 행동 결정"""
        enemies = game_state.get('enemies', [])
        
        if not enemies:
            return None
        
        # 가장 가까운 적을 공격 대상으로 선택
        closest_enemy = min(enemies, key=lambda e: self._calculate_distance(
            self.last_position, e['position']
        ))
        
        # 적에게 접근하거나 공격
        if self._calculate_distance(self.last_position, closest_enemy['position']) > 50:
            # 적에게 접근
            direction = self._get_direction_to_target(self.last_position, closest_enemy['position'])
            return {'type': 'move', 'direction': direction}
        else:
            # 공격 (확인 키로 대체)
            return {'type': 'key', 'key': 'confirm'}
    
    def _decide_movement_action(self, game_state):
        """목표 지점으로 이동 중인 행동 결정"""
        if not self.target_position or not self.last_position:
            self.target_position = None
            return None
        
        # 목표에 도달했는지 확인
        distance = self._calculate_distance(self.last_position, self.target_position)
        if distance < 20:  # 목표에 충분히 가까워짐
            self.target_position = None
            return None
        
        # 목표 방향으로 이동
        direction = self._get_direction_to_target(self.last_position, self.target_position)
        return {'type': 'move', 'direction': direction}
    
    def _decide_exploration_action(self, game_state):
        """탐험 상태에서의 행동 결정"""
        # 아이템이 있으면 수집
        items = game_state.get('items', [])
        if items:
            closest_item = min(items, key=lambda i: self._calculate_distance(
                self.last_position, i['position']
            ))
            self.target_position = closest_item['position']
            return self._decide_movement_action(game_state)
        
        # 막혔으면 랜덤 방향 시도
        if self.stuck_counter > 5:
            self.stuck_counter = 0
            return {'type': 'move', 'direction': random.choice(['up', 'down', 'left', 'right'])}
        
        # 기본 탐험 로직
        return self._explore_randomly()
    
    def _decide_idle_action(self, game_state):
        """대기 상태에서의 행동 결정"""
        # 탐험 모드로 전환
        return self._explore_randomly()
    
    def _explore_randomly(self):
        """랜덤 탐험 행동"""
        directions = ['up', 'down', 'left', 'right']
        direction = random.choice(directions)
        return {'type': 'move', 'direction': direction}
    
    def _execute_action(self, action):
        """결정된 행동을 실행합니다."""
        action_type = action.get('type')
        
        if action_type == 'move':
            direction = action.get('direction')
            duration = action.get('duration', 0.2)
            success = self.input_controller.move_direction(direction, duration)
            if success:
                print(f"이동: {direction}")
        
        elif action_type == 'key':
            key = action.get('key')
            if key == 'confirm':
                success = self.input_controller.confirm()
            elif key == 'cancel':
                success = self.input_controller.cancel()
            elif key == 'menu':
                success = self.input_controller.menu()
            else:
                success = self.input_controller.press_key(key)
            
            if success:
                print(f"키 입력: {key}")
        
        self.last_action_time = time.time()
    
    def _calculate_distance(self, pos1, pos2):
        """두 점 사이의 거리를 계산합니다."""
        if not pos1 or not pos2:
            return float('inf')
        
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def _get_direction_to_target(self, current_pos, target_pos):
        """목표 위치로의 방향을 결정합니다."""
        if not current_pos or not target_pos:
            return 'up'
        
        x1, y1 = current_pos
        x2, y2 = target_pos
        
        dx = x2 - x1
        dy = y2 - y1
        
        # 가장 큰 변화량 방향으로 이동
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    def _print_stats(self):
        """통계 정보를 출력합니다."""
        print("\n=== AI 플레이어 통계 ===")
        print(f"총 행동 수: {self.stats['total_actions']}")
        print(f"성공한 이동: {self.stats['successful_moves']}")
        print(f"승리한 전투: {self.stats['battles_won']}")
        print(f"수집한 아이템: {self.stats['items_collected']}")
        print(f"탐험 진행도: {self.stats['exploration_progress']}%")


# 테스트 코드
if __name__ == "__main__":
    from screen_capture import ScreenCapture
    from game_vision import GameVision
    from input_controller import InputController
    
    # AI 플레이어 테스트
    print("영웅전설4 AI 플레이어 초기화 중...")
    
    # 모듈 초기화
    screen_capture = ScreenCapture()
    game_vision = GameVision()
    input_controller = InputController()
    
    # DOSBox 윈도우 찾기
    if not screen_capture.find_window():
        print("DOSBox 윈도우를 찾을 수 없습니다.")
        print("DOSBox에서 영웅전설4를 먼저 실행해주세요.")
        exit(1)
    
    # AI 플레이어 생성
    ai_player = AIPlayer(screen_capture, game_vision, input_controller)
    
    print("AI 플레이어가 10초 후에 시작됩니다...")
    print("중지하려면 Ctrl+C를 누르세요.")
    time.sleep(10)
    
    # AI 플레이어 시작
    ai_player.start()