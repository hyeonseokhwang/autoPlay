"""
간단한 전투 탐색기 - pyautogui 직접 사용
"""

import pyautogui
import time
import random
import cv2
import numpy as np
from PIL import ImageGrab

# 안전 설정
pyautogui.FAILSAFE = False

class SimpleBattleSeeker:
    def __init__(self):
        self.battles_found = 0
        self.target_battles = 5
        self.current_direction = "right"
        self.last_direction_change = time.time()
        self.move_duration = 3
        
    def capture_screen(self):
        """간단한 화면 캡처"""
        try:
            screenshot = ImageGrab.grab()
            return np.array(screenshot)
        except:
            return None
    
    def is_battle_screen(self, image):
        """전투 화면인지 판별 - 몬스터/적 캐릭터가 있는지 확인"""
        if image is None:
            return False
        
        # 이미지를 BGR로 변환 (OpenCV 형식)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = image_bgr.shape[:2]
        
        # 1. 먼저 전투 UI가 있는지 확인 (하단 상태창)
        bottom_area = image_bgr[int(height * 0.7):, :]
        hsv_bottom = cv2.cvtColor(bottom_area, cv2.COLOR_BGR2HSV)
        
        # 갈색/오렌지 UI 감지
        brown_lower = np.array([10, 50, 50])
        brown_upper = np.array([25, 255, 255])
        brown_mask = cv2.inRange(hsv_bottom, brown_lower, brown_upper)
        
        brown_pixels = cv2.countNonZero(brown_mask)
        total_bottom_pixels = bottom_area.shape[0] * bottom_area.shape[1]
        ui_ratio = brown_pixels / total_bottom_pixels if total_bottom_pixels > 0 else 0
        
        # UI가 없으면 전투가 아님
        if ui_ratio < 0.15:
            return False
        
        # 2. 화면 중앙 영역에서 몬스터/적 캐릭터 감지
        # 전투 화면에서 캐릭터들이 나타나는 영역 (화면 중앙 상단)
        char_area = image_bgr[int(height * 0.2):int(height * 0.6), int(width * 0.2):int(width * 0.8)]
        hsv_char = cv2.cvtColor(char_area, cv2.COLOR_BGR2HSV)
        
        # 몬스터는 보통 어둡거나 특별한 색상을 가짐
        # 여러 색상 범위로 캐릭터 감지 시도
        character_found = False
        
        # 어두운 색상 (검은색, 회색 몬스터)
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, 80])
        dark_mask = cv2.inRange(hsv_char, dark_lower, dark_upper)
        
        # 빨간색 계열 (적 캐릭터)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv_char, red_lower1, red_upper1)
        
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv_char, red_lower2, red_upper2)
        
        # 파란색 계열 (일부 몬스터)
        blue_lower = np.array([100, 100, 100])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_char, blue_lower, blue_upper)
        
        # 보라색 계열 (마법 몬스터)
        purple_lower = np.array([130, 100, 100])
        purple_upper = np.array([160, 255, 255])
        purple_mask = cv2.inRange(hsv_char, purple_lower, purple_upper)
        
        # 각 마스크에서 윤곽선 검출
        masks = [dark_mask, red_mask1, red_mask2, blue_mask, purple_mask]
        
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                # 캐릭터 크기로 보이는 윤곽선이 있으면 전투 상황
                if 200 < area < 5000:  # 적절한 크기의 객체
                    x, y, w, h = cv2.boundingRect(contour)
                    # 가로세로 비율이 캐릭터 같으면
                    if 0.3 < w/h < 3.0 and w > 20 and h > 20:
                        character_found = True
                        break
            
            if character_found:
                break
        
        # UI가 있고 캐릭터도 감지되면 전투 화면
        return ui_ratio > 0.15 and character_found
    
    def move_in_direction(self, direction, duration=0.5):
        """지정된 방향으로 이동"""
        if direction == "left":
            pyautogui.keyDown('left')
            time.sleep(duration)
            pyautogui.keyUp('left')
        elif direction == "right":
            pyautogui.keyDown('right')
            time.sleep(duration)
            pyautogui.keyUp('right')
        elif direction == "up":
            pyautogui.keyDown('up')
            time.sleep(duration)
            pyautogui.keyUp('up')
        elif direction == "down":
            pyautogui.keyDown('down')
            time.sleep(duration)
            pyautogui.keyUp('down')
    
    def change_direction(self):
        """방향 변경"""
        directions = ["left", "right", "up", "down"]
        self.current_direction = random.choice(directions)
        self.move_duration = random.uniform(2, 4)
        print(f"🔄 방향 전환: {self.current_direction}")
    
    def handle_battle(self):
        """전투 처리"""
        print(f"⚔️ 전투 발견! ({self.battles_found + 1}/{self.target_battles})")
        battle_start = time.time()
        
        # 전투 중 처리
        while True:
            # 화면 캡처해서 전투가 끝났는지 확인
            screen = self.capture_screen()
            if screen is not None and not self.is_battle_screen(screen):
                battle_duration = time.time() - battle_start
                print(f"✅ 전투 종료! (지속시간: {battle_duration:.1f}초)")
                self.battles_found += 1
                break
            
            # 전투 진행 (Enter 키)
            print("🗡️ 전투 진행...")
            pyautogui.press('enter')
            time.sleep(2)
            
            # 너무 오래 걸리면 탈출
            if time.time() - battle_start > 30:
                print("⏰ 전투 시간 초과, 탈출 시도...")
                pyautogui.press('esc')
                time.sleep(1)
                self.battles_found += 1  # 강제로 카운트 증가
                break
    
    def explore(self):
        """탐험하기"""
        current_time = time.time()
        
        # 방향 전환 타이밍
        if current_time - self.last_direction_change > self.move_duration:
            self.change_direction()
            self.last_direction_change = current_time
        
        # 이동
        print(f"🚶 {self.current_direction} 방향으로 탐험 중...")
        self.move_in_direction(self.current_direction, 0.3)
        time.sleep(0.2)
    
    def analyze_screen_debug(self, image):
        """화면 분석 디버그 정보"""
        if image is None:
            return "화면 없음"
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = image_bgr.shape[:2]
        
        # UI 영역 분석
        bottom_area = image_bgr[int(height * 0.7):, :]
        hsv_bottom = cv2.cvtColor(bottom_area, cv2.COLOR_BGR2HSV)
        brown_lower = np.array([10, 50, 50])
        brown_upper = np.array([25, 255, 255])
        brown_mask = cv2.inRange(hsv_bottom, brown_lower, brown_upper)
        brown_pixels = cv2.countNonZero(brown_mask)
        total_pixels = bottom_area.shape[0] * bottom_area.shape[1]
        ui_ratio = brown_pixels / total_pixels if total_pixels > 0 else 0
        
        # 캐릭터 영역 분석
        char_area = image_bgr[int(height * 0.2):int(height * 0.6), int(width * 0.2):int(width * 0.8)]
        
        return f"UI비율: {ui_ratio:.3f}, 화면크기: {width}x{height}"
    
    def run(self):
        """메인 실행"""
        print("DOSBox 게임 창을 클릭해서 활성화해주세요!")
        print("5초 후 자동 탐험을 시작합니다...\n")
        
        for i in range(5, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print(f"🎯 목표: {self.target_battles}번의 전투 찾기")
        print("탐험 시작!\n")
        
        start_time = time.time()
        last_debug = time.time()
        
        while self.battles_found < self.target_battles:
            try:
                # 화면 분석
                screen = self.capture_screen()
                
                # 5초마다 화면 상태 디버그 출력
                if time.time() - last_debug > 5:
                    debug_info = self.analyze_screen_debug(screen)
                    is_battle = self.is_battle_screen(screen) if screen is not None else False
                    print(f"🔍 화면상태: {debug_info}, 전투판정: {is_battle}")
                    last_debug = time.time()
                
                if screen is not None and self.is_battle_screen(screen):
                    self.handle_battle()
                else:
                    self.explore()
                
                # 진행 상황 출력
                if int(time.time() - start_time) % 15 == 0:
                    elapsed = time.time() - start_time
                    print(f"📊 진행: 전투 {self.battles_found}/{self.target_battles}, 시간: {elapsed:.0f}초")
                
            except KeyboardInterrupt:
                print("\n사용자가 중단했습니다.")
                break
        
        total_time = time.time() - start_time
        print(f"\n🎉 탐험 완료!")
        print(f"총 {self.battles_found}번의 전투를 경험했습니다.")
        print(f"총 시간: {total_time:.1f}초")


if __name__ == "__main__":
    print("=" * 50)
    print("   영웅전설4 간단 전투 탐색기")
    print("=" * 50)
    
    seeker = SimpleBattleSeeker()
    seeker.run()