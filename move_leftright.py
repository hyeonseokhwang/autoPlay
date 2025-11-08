import sys
import os
import time

# 경로 추가
sys.path.append('G:/LucasAI')

from src.screen_capture import ScreenCapture
from src.input_controller import InputController

print("영웅전설4 게임에서 1분간 좌우 움직임을 시작합니다...")

# 화면 캡처 및 입력 제어 초기화
capture = ScreenCapture()
controller = InputController()

# DOSBox 윈도우 찾기
if capture.find_window():
    print("DOSBox 윈도우 발견!")
    
    # 입력 컨트롤러에 윈도우 설정
    controller.set_target_window(capture.hwnd)
    
    print("3초 후 좌우 움직임 시작...")
    time.sleep(3)
    
    print("좌우 움직임 시작! (60초간)")
    start_time = time.time()
    
    # 60초간 좌우 반복 움직임 (3초씩 방향 전환)
    while time.time() - start_time < 60:
        elapsed = time.time() - start_time
        remaining = 60 - elapsed
        
        # 왼쪽으로 3초 이동
        print(f"[{elapsed:.0f}초] << 왼쪽으로 3초 이동 (남은시간: {remaining:.0f}초)")
        end_time = time.time() + 3
        while time.time() < end_time and (time.time() - start_time) < 60:
            controller.move_left(0.1)
            time.sleep(0.05)
        
        if (time.time() - start_time) >= 60:
            break
            
        elapsed = time.time() - start_time
        remaining = 60 - elapsed
        
        # 오른쪽으로 3초 이동  
        print(f"[{elapsed:.0f}초] >> 오른쪽으로 3초 이동 (남은시간: {remaining:.0f}초)")
        end_time = time.time() + 3
        while time.time() < end_time and (time.time() - start_time) < 60:
            controller.move_right(0.1)
            time.sleep(0.05)
    
    print("좌우 움직임 완료!")
    
else:
    print("DOSBox 윈도우를 찾을 수 없습니다.")