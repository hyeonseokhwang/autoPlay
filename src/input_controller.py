"""
입력 제어 모듈
게임에 키보드 입력을 전송합니다.
"""

import pyautogui
import time
import win32gui
import win32con
import json


class InputController:
    def __init__(self, config_path="config/settings.json"):
        """
        입력 제어 클래스 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.input_settings = self.config.get("input_settings", {})
        self.key_delay = self.input_settings.get("key_delay", 0.1)
        self.movement_keys = self.input_settings.get("movement_keys", {})
        self.action_keys = self.input_settings.get("action_keys", {})
        
        # 윈도우 핸들
        self.target_hwnd = None
        
        # pyautogui 설정
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        
    def _load_config(self, config_path):
        """설정 파일을 로드합니다."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return {}
    
    def set_target_window(self, hwnd):
        """
        입력을 전송할 대상 윈도우를 설정합니다.
        
        Args:
            hwnd: 윈도우 핸들
        """
        self.target_hwnd = hwnd
    
    def activate_window(self):
        """대상 윈도우를 활성화합니다."""
        if self.target_hwnd:
            try:
                win32gui.SetForegroundWindow(self.target_hwnd)
                time.sleep(0.1)
                return True
            except Exception as e:
                print(f"윈도우 활성화 실패: {e}")
                return False
        return False
    
    def press_key(self, key, duration=None):
        """
        키를 누릅니다.
        
        Args:
            key (str): 누를 키
            duration (float): 키를 누르고 있을 시간 (None이면 기본값)
        """
        if not self.activate_window():
            return False
            
        try:
            if duration:
                pyautogui.keyDown(key)
                time.sleep(duration)
                pyautogui.keyUp(key)
            else:
                pyautogui.press(key)
            
            time.sleep(self.key_delay)
            return True
        except Exception as e:
            print(f"키 입력 실패 ({key}): {e}")
            return False
    
    def press_keys_combination(self, keys):
        """
        키 조합을 누릅니다.
        
        Args:
            keys (list): 동시에 누를 키들의 리스트
        """
        if not self.activate_window():
            return False
            
        try:
            pyautogui.hotkey(*keys)
            time.sleep(self.key_delay)
            return True
        except Exception as e:
            print(f"키 조합 입력 실패 ({keys}): {e}")
            return False
    
    def move_up(self, duration=None):
        """위쪽으로 이동합니다."""
        key = self.movement_keys.get("up", "up")
        return self.press_key(key, duration)
    
    def move_down(self, duration=None):
        """아래쪽으로 이동합니다."""
        key = self.movement_keys.get("down", "down")
        return self.press_key(key, duration)
    
    def move_left(self, duration=None):
        """왼쪽으로 이동합니다."""
        key = self.movement_keys.get("left", "left")
        return self.press_key(key, duration)
    
    def move_right(self, duration=None):
        """오른쪽으로 이동합니다."""
        key = self.movement_keys.get("right", "right")
        return self.press_key(key, duration)
    
    def move_direction(self, direction, duration=None):
        """
        지정된 방향으로 이동합니다.
        
        Args:
            direction (str): 'up', 'down', 'left', 'right'
            duration (float): 이동 시간
        """
        move_methods = {
            'up': self.move_up,
            'down': self.move_down,
            'left': self.move_left,
            'right': self.move_right
        }
        
        method = move_methods.get(direction.lower())
        if method:
            return method(duration)
        else:
            print(f"잘못된 방향: {direction}")
            return False
    
    def confirm(self):
        """확인 키를 누릅니다."""
        key = self.action_keys.get("confirm", "enter")
        return self.press_key(key)
    
    def cancel(self):
        """취소 키를 누릅니다."""
        key = self.action_keys.get("cancel", "esc")
        return self.press_key(key)
    
    def menu(self):
        """메뉴 키를 누릅니다."""
        key = self.action_keys.get("menu", "space")
        return self.press_key(key)
    
    def move_to_position(self, start_pos, target_pos, step_delay=0.2):
        """
        시작 위치에서 목표 위치까지 이동합니다.
        
        Args:
            start_pos (tuple): 시작 위치 (x, y)
            target_pos (tuple): 목표 위치 (x, y)
            step_delay (float): 각 이동 단계 사이의 지연시간
        """
        x1, y1 = start_pos
        x2, y2 = target_pos
        
        # X축 이동
        if x2 > x1:
            steps = x2 - x1
            for _ in range(steps):
                self.move_right()
                time.sleep(step_delay)
        elif x2 < x1:
            steps = x1 - x2
            for _ in range(steps):
                self.move_left()
                time.sleep(step_delay)
        
        # Y축 이동
        if y2 > y1:
            steps = y2 - y1
            for _ in range(steps):
                self.move_down()
                time.sleep(step_delay)
        elif y2 < y1:
            steps = y1 - y2
            for _ in range(steps):
                self.move_up()
                time.sleep(step_delay)
    
    def random_movement(self, duration=5.0):
        """
        랜덤하게 이동합니다.
        
        Args:
            duration (float): 랜덤 이동 총 시간
        """
        import random
        
        directions = ['up', 'down', 'left', 'right']
        start_time = time.time()
        
        while time.time() - start_time < duration:
            direction = random.choice(directions)
            move_time = random.uniform(0.1, 0.5)
            
            self.move_direction(direction, move_time)
            time.sleep(random.uniform(0.1, 0.3))
    
    def type_text(self, text, delay=0.05):
        """
        텍스트를 입력합니다.
        
        Args:
            text (str): 입력할 텍스트
            delay (float): 각 문자 사이의 지연시간
        """
        if not self.activate_window():
            return False
        
        try:
            for char in text:
                pyautogui.write(char)
                time.sleep(delay)
            return True
        except Exception as e:
            print(f"텍스트 입력 실패: {e}")
            return False
    
    def emergency_stop(self):
        """긴급 정지 - 모든 키 입력을 중단합니다."""
        try:
            # 모든 키를 떼기
            for key in ['up', 'down', 'left', 'right', 'enter', 'esc', 'space']:
                pyautogui.keyUp(key)
            print("긴급 정지 실행됨")
        except Exception as e:
            print(f"긴급 정지 실패: {e}")


# 테스트 코드
if __name__ == "__main__":
    from screen_capture import ScreenCapture
    
    # 입력 제어 테스트
    capture = ScreenCapture()
    controller = InputController()
    
    if capture.find_window():
        controller.set_target_window(capture.hwnd)
        
        print("5초 후 테스트 입력을 시작합니다...")
        time.sleep(5)
        
        print("방향키 테스트...")
        # 각 방향으로 0.5초씩 이동
        directions = ['up', 'down', 'left', 'right']
        for direction in directions:
            print(f"{direction} 이동")
            controller.move_direction(direction, 0.5)
            time.sleep(1)
        
        print("액션 키 테스트...")
        # 메뉴 열기/닫기 테스트
        controller.menu()
        time.sleep(1)
        controller.cancel()
        
        print("테스트 완료")
    else:
        print("게임 윈도우를 찾을 수 없습니다.")