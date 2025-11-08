import pyautogui
import time

print("1분간 좌우 움직임 시작!")
print("DOSBox 창을 클릭해서 활성화한 후 3초 내에 게임으로 돌아가세요...")

# 3초 대기
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print("좌우 움직임 시작! (60초간)")

# pyautogui 안전 설정 해제
pyautogui.FAILSAFE = False

start_time = time.time()
direction = "left"

# 60초간 좌우 반복
while time.time() - start_time < 60:
    elapsed = time.time() - start_time
    remaining = 60 - elapsed
    
    if direction == "left":
        print(f"[{elapsed:.0f}초] << 왼쪽 이동 (남은시간: {remaining:.0f}초)")
        # 2초간 왼쪽 키 누르기
        pyautogui.keyDown('left')
        time.sleep(2)
        pyautogui.keyUp('left')
        direction = "right"
    else:
        print(f"[{elapsed:.0f}초] >> 오른쪽 이동 (남은시간: {remaining:.0f}초)")  
        # 2초간 오른쪽 키 누르기
        pyautogui.keyDown('right')
        time.sleep(2)
        pyautogui.keyUp('right')
        direction = "left"
    
    # 잠시 멈춤
    time.sleep(0.1)

print("좌우 움직임 완료!")