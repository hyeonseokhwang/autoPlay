"""
DOSBox 전용 전투 탐색기 - 특정 윈도우만 타겟팅
"""

import win32gui
import win32ui
import win32con
import win32api
import time
import random
import cv2
import numpy as np
from PIL import Image

class DOSBoxBattleSeeker:
    def __init__(self):
        self.battles_found = 0
        self.target_battles = 5
        self.current_direction = "right"  # 오른쪽부터 시작
        self.last_direction_change = time.time()
        self.move_duration = 4  # 좌우로 더 오래 이동
        self.dosbox_hwnd = None
        
        # 좌우 탐험 전용 설정
        self.field_changes = 0  # 필드 전환 횟수
        self.exploration_cycles = 0  # 탐험 사이클 수
        
    def find_dosbox_window(self):
        """DOSBox 윈도우 찾기"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if "DOSBox" in window_text:
                    windows.append((hwnd, window_text))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            self.dosbox_hwnd = windows[0][0]
            window_title = windows[0][1]
            print(f"✓ DOSBox 윈도우 발견: {window_title}")
            return True
        else:
            print("❌ DOSBox 윈도우를 찾을 수 없습니다.")
            return False
    
    def capture_dosbox_window(self):
        """DOSBox 윈도우만 캡처"""
        if not self.dosbox_hwnd or not win32gui.IsWindow(self.dosbox_hwnd):
            return None
        
        try:
            # 윈도우 위치와 크기 가져오기
            left, top, right, bottom = win32gui.GetWindowRect(self.dosbox_hwnd)
            width = right - left
            height = bottom - top
            
            # 윈도우 DC 가져오기
            hwndDC = win32gui.GetWindowDC(self.dosbox_hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # 비트맵 생성
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # 윈도우 내용을 비트맵에 복사
            result = saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
            
            # 비트맵을 이미지로 변환
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            img = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1)
            
            # 리소스 해제
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.dosbox_hwnd, hwndDC)
            
            return np.array(img)
            
        except Exception as e:
            print(f"화면 캡처 오류: {e}")
            return None
    
    def send_key_to_dosbox(self, key_code, press_time=0.1):
        """DOSBox 윈도우에만 키 입력 전송"""
        if not self.dosbox_hwnd or not win32gui.IsWindow(self.dosbox_hwnd):
            return False
        
        try:
            # 윈도우를 포그라운드로 가져오기
            win32gui.SetForegroundWindow(self.dosbox_hwnd)
            time.sleep(0.05)
            
            # 키 다운
            win32api.keybd_event(key_code, 0, 0, 0)
            time.sleep(press_time)
            # 키 업
            win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            
            return True
        except Exception as e:
            print(f"키 입력 오류: {e}")
            return False
    
    def is_battle_screen(self, image):
        """전투 화면인지 판별 - 몬스터/적 캐릭터가 있는지 확인"""
        if image is None:
            return False
        
        # 이미지가 RGB 형식인지 확인하고 BGR로 변환
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
        else:
            return False
        
        height, width = image_bgr.shape[:2]
        
        # DOSBox 내부의 게임 화면 영역만 추출 (타이틀바 제외)
        # 일반적으로 DOSBox는 상단에 타이틀바가 있음
        game_area = image_bgr[30:height-10, 10:width-10]  # 여백 제거
        
        if game_area.shape[0] < 100 or game_area.shape[1] < 100:
            return False
        
        game_height, game_width = game_area.shape[:2]
        
        # 1. 전투 UI 확인 (하단 상태창)
        bottom_area = game_area[int(game_height * 0.7):, :]
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
        char_area = game_area[int(game_height * 0.1):int(game_height * 0.6), 
                             int(game_width * 0.1):int(game_width * 0.9)]
        
        if char_area.shape[0] < 50 or char_area.shape[1] < 50:
            return False
            
        hsv_char = cv2.cvtColor(char_area, cv2.COLOR_BGR2HSV)
        
        # 몬스터 감지를 위한 여러 색상 마스크
        character_found = False
        
        # 어두운 색상 (검은색, 회색 몬스터)
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, 100])
        dark_mask = cv2.inRange(hsv_char, dark_lower, dark_upper)
        
        # 빨간색 계열
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv_char, red_lower1, red_upper1)
        
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv_char, red_lower2, red_upper2)
        
        # 파란색/보라색 계열
        blue_lower = np.array([100, 100, 100])
        blue_upper = np.array([160, 255, 255])
        blue_mask = cv2.inRange(hsv_char, blue_lower, blue_upper)
        
        masks = [dark_mask, red_mask1, red_mask2, blue_mask]
        
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                # 캐릭터 크기로 보이는 윤곽선 체크
                if 100 < area < 3000:  # DOSBox 해상도에 맞게 조정
                    x, y, w, h = cv2.boundingRect(contour)
                    if 0.2 < w/h < 5.0 and w > 10 and h > 10:
                        character_found = True
                        break
            
            if character_found:
                break
        
        return ui_ratio > 0.15 and character_found
    
    def move_in_direction(self, direction, duration=0.3):
        """DOSBox에 방향키 입력"""
        # 윈도우 키 코드
        VK_LEFT = 0x25
        VK_UP = 0x26
        VK_RIGHT = 0x27
        VK_DOWN = 0x28
        
        key_map = {
            "left": VK_LEFT,
            "right": VK_RIGHT,
            "up": VK_UP,
            "down": VK_DOWN
        }
        
        if direction in key_map:
            return self.send_key_to_dosbox(key_map[direction], duration)
        return False
    
    def handle_battle(self):
        """전투 처리"""
        print(f"⚔️ 전투 발견! ({self.battles_found + 1}/{self.target_battles})")
        battle_start = time.time()
        
        VK_RETURN = 0x0D  # Enter 키
        VK_ESCAPE = 0x1B  # ESC 키
        
        while True:
            # 화면 캡처해서 전투가 끝났는지 확인
            screen = self.capture_dosbox_window()
            if screen is not None and not self.is_battle_screen(screen):
                battle_duration = time.time() - battle_start
                print(f"✅ 전투 종료! (지속시간: {battle_duration:.1f}초)")
                self.battles_found += 1
                break
            
            # 전투 진행 (Enter 키)
            print("🗡️ 전투 진행...")
            self.send_key_to_dosbox(VK_RETURN, 0.1)
            time.sleep(2)
            
            # 너무 오래 걸리면 탈출
            if time.time() - battle_start > 30:
                print("⏰ 전투 시간 초과, 탈출 시도...")
                self.send_key_to_dosbox(VK_ESCAPE, 0.1)
                time.sleep(1)
                self.battles_found += 1
                break
    
    def change_direction(self):
        """방향 변경 - 좌우로만 이동"""
        directions = ["left", "right"]  # 좌우로만 제한
        # 현재 방향과 반대 방향 우선 선택
        if self.current_direction == "left":
            self.current_direction = "right"
        elif self.current_direction == "right":
            self.current_direction = "left"
        else:
            self.current_direction = random.choice(directions)
        
        self.move_duration = random.uniform(3, 6)  # 조금 더 오래 이동
        print(f"🔄 방향 전환: {self.current_direction}")
    
    def explore(self):
        """좌우 필드 탐험"""
        current_time = time.time()
        
        # 방향 전환 타이밍 (좌우 필드 전환을 위해 충분히 이동)
        if current_time - self.last_direction_change > self.move_duration:
            self.change_direction()
            self.last_direction_change = current_time
            self.field_changes += 1
            print(f"🌍 필드 전환 시도 #{self.field_changes}")
        
        # DOSBox에서 좌우로만 이동
        direction_symbol = "←" if self.current_direction == "left" else "→"
        print(f"🚶 {direction_symbol} DOSBox에서 {self.current_direction} 필드로 탐험 중...")
        
        # 연속 이동으로 필드 경계까지 가기
        success = self.move_in_direction(self.current_direction, 0.4)
        if not success:
            print("⚠️ 키 입력 실패, 재시도...")
        
        time.sleep(0.2)  # 조금 더 빠른 이동
    
    def analyze_screen_debug(self, image):
        """화면 분석 디버그"""
        if image is None:
            return "DOSBox 화면 없음"
        
        height, width = image.shape[:2] if len(image.shape) >= 2 else (0, 0)
        return f"DOSBox 크기: {width}x{height}"
    
    def run(self):
        """메인 실행"""
        print("DOSBox 윈도우를 찾는 중...")
        
        if not self.find_dosbox_window():
            return False
        
        print("3초 후 DOSBox에서 탐험을 시작합니다...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print(f"🎯 목표: {self.target_battles}번의 전투 찾기")
        print("DOSBox 탐험 시작!\n")
        
        start_time = time.time()
        last_debug = time.time()
        
        while self.battles_found < self.target_battles:
            try:
                # DOSBox 화면만 캡처
                screen = self.capture_dosbox_window()
                
                # 5초마다 디버그 정보
                if time.time() - last_debug > 5:
                    debug_info = self.analyze_screen_debug(screen)
                    is_battle = self.is_battle_screen(screen) if screen is not None else False
                    print(f"🔍 {debug_info}, 전투판정: {is_battle}")
                    last_debug = time.time()
                
                if screen is not None and self.is_battle_screen(screen):
                    self.handle_battle()
                else:
                    self.explore()
                
                # 진행 상황 출력
                if int(time.time() - start_time) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"📊 진행: 전투 {self.battles_found}/{self.target_battles}, "
                          f"시간: {elapsed:.0f}초, 필드전환: {self.field_changes}회")
                
            except KeyboardInterrupt:
                print("\n사용자가 중단했습니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")
                time.sleep(1)
        
        total_time = time.time() - start_time
        print(f"\n🎉 DOSBox 탐험 완료!")
        print(f"총 {self.battles_found}번의 전투를 경험했습니다.")
        print(f"총 시간: {total_time:.1f}초")
        
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("   영웅전설4 DOSBox 전용 전투 탐색기")
    print("=" * 60)
    
    seeker = DOSBoxBattleSeeker()
    seeker.run()