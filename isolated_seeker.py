"""
DOSBox 직접 메시지 전송 방식 전투 탐색기
시스템 키보드 입력을 점유하지 않고 DOSBox에만 직접 메시지 전송
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

class IsolatedDOSBoxSeeker:
    def __init__(self):
        self.battles_found = 0
        self.target_battles = 5
        self.current_direction = "right"
        self.last_direction_change = time.time()
        self.move_duration = 4
        self.dosbox_hwnd = None
        self.field_changes = 0
        
        # 윈도우 메시지 상수
        self.WM_KEYDOWN = 0x0100
        self.WM_KEYUP = 0x0101
        self.WM_CHAR = 0x0102
        
        # 가상 키 코드
        self.VK_LEFT = 0x25
        self.VK_RIGHT = 0x27
        self.VK_UP = 0x26
        self.VK_DOWN = 0x28
        self.VK_RETURN = 0x0D
        self.VK_ESCAPE = 0x1B
        
    def find_dosbox_window(self):
        """DOSBox 윈도우 찾기 - 개선된 감지"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                
                # 다양한 패턴으로 DOSBox 찾기
                dosbox_patterns = ["DOSBox", "dosbox", "DOSBOX", "ED4", "영웅전설"]
                
                for pattern in dosbox_patterns:
                    if pattern in window_text or pattern in class_name:
                        windows.append((hwnd, window_text, class_name))
                        break
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        print("🔍 발견된 윈도우들:")
        for hwnd, title, class_name in windows:
            print(f"  - 제목: '{title}', 클래스: '{class_name}', 핸들: {hwnd}")
        
        if windows:
            # 가장 적합한 윈도우 선택 (DOSBox가 제목에 있는 것 우선)
            best_window = None
            for hwnd, title, class_name in windows:
                if "DOSBox" in title:
                    best_window = (hwnd, title, class_name)
                    break
            
            if not best_window:
                best_window = windows[0]  # 첫 번째 윈도우 사용
            
            self.dosbox_hwnd = best_window[0]
            window_title = best_window[1]
            print(f"✓ DOSBox 윈도우 선택: {window_title}")
            print(f"  윈도우 핸들: {self.dosbox_hwnd}")
            return True
        else:
            print("❌ DOSBox 관련 윈도우를 찾을 수 없습니다.")
            print("   다음을 확인해주세요:")
            print("   1. DOSBox가 실행되어 있는지")
            print("   2. 게임이 로드되어 있는지") 
            print("   3. 윈도우가 최소화되지 않았는지")
            return False
    
    def send_key_message(self, vk_code, press_duration=0.1):
        """DOSBox에 직접 키 메시지 전송 (시스템 키보드 점유 안함)"""
        if not self.dosbox_hwnd or not win32gui.IsWindow(self.dosbox_hwnd):
            return False
        
        try:
            # DOSBox 내부의 실제 게임 윈도우 찾기
            # DOSBox는 때로 자식 윈도우가 실제 게임 화면임
            child_hwnd = win32gui.GetWindow(self.dosbox_hwnd, win32con.GW_CHILD)
            target_hwnd = child_hwnd if child_hwnd else self.dosbox_hwnd
            
            # 스캔 코드 계산
            scan_code = win32api.MapVirtualKey(vk_code, 0)
            lparam_down = (scan_code << 16) | 1
            lparam_up = (scan_code << 16) | 0xC0000001
            
            # 키 다운 메시지 전송
            win32gui.SendMessage(target_hwnd, self.WM_KEYDOWN, vk_code, lparam_down)
            time.sleep(press_duration)
            
            # 키 업 메시지 전송
            win32gui.SendMessage(target_hwnd, self.WM_KEYUP, vk_code, lparam_up)
            
            return True
            
        except Exception as e:
            print(f"메시지 전송 실패: {e}")
            return False
    
    def capture_dosbox_window(self):
        """DOSBox 윈도우 캡처"""
        if not self.dosbox_hwnd or not win32gui.IsWindow(self.dosbox_hwnd):
            return None
        
        try:
            left, top, right, bottom = win32gui.GetWindowRect(self.dosbox_hwnd)
            width = right - left
            height = bottom - top
            
            hwndDC = win32gui.GetWindowDC(self.dosbox_hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            result = saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
            
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
    
    def is_battle_screen(self, image):
        """전투 화면 감지 - HP/MP 표시 방식으로 구분"""
        if image is None:
            return False
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return False
        
        height, width = image_bgr.shape[:2]
        game_area = image_bgr[30:height-10, 10:width-10]
        
        if game_area.shape[0] < 100 or game_area.shape[1] < 100:
            return False
        
        game_height, game_width = game_area.shape[:2]
        
        # 1. 전투 화면 특징: 별도의 하단 전투 영역이 있음
        # 화면이 상하로 분할되어 있는지 확인
        middle_line = game_area[int(game_height * 0.6):int(game_height * 0.7), :]
        
        # 가로 구분선 감지 (전투 화면에서 상하 분할선)
        hsv_middle = cv2.cvtColor(middle_line, cv2.COLOR_BGR2HSV)
        
        # 구분선은 보통 갈색/오렌지색 UI 프레임
        separator_lower = np.array([10, 50, 100])
        separator_upper = np.array([25, 255, 255])
        separator_mask = cv2.inRange(hsv_middle, separator_lower, separator_upper)
        
        # 가로로 긴 구분선이 있는지 확인
        separator_pixels = cv2.countNonZero(separator_mask)
        middle_pixels = middle_line.shape[0] * middle_line.shape[1]
        separator_ratio = separator_pixels / middle_pixels if middle_pixels > 0 else 0
        
        # 2. 하단 전투 영역 확인 (전투 시에만 나타나는 영역)
        bottom_battle_area = game_area[int(game_height * 0.7):, :]
        
        # 하단 영역의 색상 분포 확인
        hsv_bottom = cv2.cvtColor(bottom_battle_area, cv2.COLOR_BGR2HSV)
        
        # 석조 바닥 패턴 (전투 영역의 특징)
        stone_lower = np.array([0, 0, 100])    # 회색/석조 색상
        stone_upper = np.array([180, 50, 200])
        stone_mask = cv2.inRange(hsv_bottom, stone_lower, stone_upper)
        
        stone_pixels = cv2.countNonZero(stone_mask)
        bottom_pixels = bottom_battle_area.shape[0] * bottom_battle_area.shape[1]
        stone_ratio = stone_pixels / bottom_pixels if bottom_pixels > 0 else 0
        
        # 3. 일반 화면의 HP/MP 텍스트 패턴 감지 (이것이 있으면 일반 화면)
        # 하단 상태창에서 "HP XXX MP XXX" 패턴 찾기
        status_area = game_area[int(game_height * 0.7):, :]
        
        # 텍스트 색상 감지 (흰색/노란색 텍스트)
        hsv_status = cv2.cvtColor(status_area, cv2.COLOR_BGR2HSV)
        
        # 흰색 텍스트
        white_text_lower = np.array([0, 0, 200])
        white_text_upper = np.array([180, 30, 255])
        white_text_mask = cv2.inRange(hsv_status, white_text_lower, white_text_upper)
        
        # 노란색 텍스트  
        yellow_text_lower = np.array([20, 100, 200])
        yellow_text_upper = np.array([30, 255, 255])
        yellow_text_mask = cv2.inRange(hsv_status, yellow_text_lower, yellow_text_upper)
        
        # 텍스트 비율 계산
        text_pixels = cv2.countNonZero(white_text_mask) + cv2.countNonZero(yellow_text_mask)
        status_pixels = status_area.shape[0] * status_area.shape[1]
        text_ratio = text_pixels / status_pixels if status_pixels > 0 else 0
        
        # 4. 전투 캐릭터 감지 (하단 전투 영역에서)
        battle_chars_found = False
        
        if separator_ratio > 0.1:  # 구분선이 있으면 하단에서 캐릭터 찾기
            battle_char_area = game_area[int(game_height * 0.65):, :]
            hsv_battle = cv2.cvtColor(battle_char_area, cv2.COLOR_BGR2HSV)
            
            # 드래곤이나 몬스터의 특징적 색상
            monster_colors = [
                ([35, 100, 100], [85, 255, 255]),    # 녹색 (드래곤)
                ([0, 100, 100], [15, 255, 255]),     # 빨간색
                ([160, 100, 100], [180, 255, 255]),  # 빨간색2
                ([100, 100, 100], [130, 255, 255])   # 파란색
            ]
            
            for lower, upper in monster_colors:
                mask = cv2.inRange(hsv_battle, np.array(lower), np.array(upper))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # 큰 객체 (몬스터)
                        battle_chars_found = True
                        break
                
                if battle_chars_found:
                    break
        
        # 5. 최종 판정
        # 전투 화면의 조건:
        # - 화면 분할선이 있고 (separator_ratio > 0.1)
        # - 하단에 석조 바닥이 있고 (stone_ratio > 0.1) 
        # - 전투 캐릭터가 있거나
        # - HP/MP 텍스트가 적음 (일반 화면과 구별)
        
        is_battle = (
            separator_ratio > 0.1 and           # 화면 분할선 존재
            (stone_ratio > 0.1 or               # 석조 바닥 존재 또는
             battle_chars_found or              # 전투 캐릭터 존재 또는  
             text_ratio < 0.05)                # HP/MP 텍스트가 적음 (전투 UI와 구별)
        )
        
        return is_battle
    
    def move_in_direction(self, direction, duration=0.3):
        """방향키 메시지 전송"""
        key_map = {
            "left": self.VK_LEFT,
            "right": self.VK_RIGHT,
            "up": self.VK_UP,
            "down": self.VK_DOWN
        }
        
        if direction in key_map:
            return self.send_key_message(key_map[direction], duration)
        return False
    
    def handle_battle(self):
        """전투 처리"""
        print(f"⚔️ 전투 발견! ({self.battles_found + 1}/{self.target_battles})")
        battle_start = time.time()
        
        while True:
            screen = self.capture_dosbox_window()
            if screen is not None and not self.is_battle_screen(screen):
                battle_duration = time.time() - battle_start
                print(f"✅ 전투 종료! (지속시간: {battle_duration:.1f}초)")
                self.battles_found += 1
                break
            
            print("🗡️ 전투 진행...")
            self.send_key_message(self.VK_RETURN, 0.1)
            time.sleep(2)
            
            if time.time() - battle_start > 30:
                print("⏰ 전투 시간 초과, 탈출 시도...")
                self.send_key_message(self.VK_ESCAPE, 0.1)
                time.sleep(1)
                self.battles_found += 1
                break
    
    def change_direction(self):
        """좌우 방향 전환"""
        if self.current_direction == "left":
            self.current_direction = "right"
        else:
            self.current_direction = "left"
        
        self.move_duration = random.uniform(3, 6)
        print(f"🔄 방향 전환: {self.current_direction}")
    
    def explore(self):
        """좌우 탐험"""
        current_time = time.time()
        
        if current_time - self.last_direction_change > self.move_duration:
            self.change_direction()
            self.last_direction_change = current_time
            self.field_changes += 1
            print(f"🌍 필드 전환 시도 #{self.field_changes}")
        
        direction_symbol = "←" if self.current_direction == "left" else "→"
        print(f"🚶 {direction_symbol} DOSBox 메시지로 {self.current_direction} 이동")
        
        success = self.move_in_direction(self.current_direction, 0.4)
        if not success:
            print("⚠️ 메시지 전송 실패, 재시도...")
        
        time.sleep(0.2)
    
    def run(self):
        """메인 실행"""
        print("🔍 DOSBox 윈도우를 찾는 중...")
        
        if not self.find_dosbox_window():
            return False
        
        print("\n✨ 독립 채널 모드로 실행됩니다!")
        print("   - 시스템 키보드 입력 점유 안함")
        print("   - DOSBox에만 직접 메시지 전송")
        print("   - 다른 작업 방해 안함")
        print()
        
        print("3초 후 탐험 시작...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print(f"🎯 목표: {self.target_battles}번의 전투")
        print("🚀 독립 채널 탐험 시작!\n")
        
        start_time = time.time()
        last_debug = time.time()
        
        while self.battles_found < self.target_battles:
            try:
                screen = self.capture_dosbox_window()
                
                if time.time() - last_debug > 5:
                    debug_info = f"DOSBox 크기: {screen.shape[1]}x{screen.shape[0]}" if screen is not None else "화면 없음"
                    is_battle = self.is_battle_screen(screen) if screen is not None else False
                    print(f"🔍 {debug_info}, 전투: {is_battle}")
                    last_debug = time.time()
                
                if screen is not None and self.is_battle_screen(screen):
                    self.handle_battle()
                else:
                    self.explore()
                
                if int(time.time() - start_time) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"📊 진행: 전투 {self.battles_found}/{self.target_battles}, "
                          f"시간: {elapsed:.0f}초, 필드전환: {self.field_changes}회")
                
            except KeyboardInterrupt:
                print("\n사용자가 중단했습니다.")
                break
            except Exception as e:
                print(f"오류: {e}")
                time.sleep(1)
        
        total_time = time.time() - start_time
        print(f"\n🎉 독립 채널 탐험 완료!")
        print(f"총 {self.battles_found}번의 전투 경험")
        print(f"총 시간: {total_time:.1f}초")
        print("시스템 키보드는 전혀 영향받지 않았습니다! ✨")
        
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("   영웅전설4 독립채널 전투 탐색기")
    print("   (시스템 키보드 점유 없음)")
    print("=" * 60)
    
    seeker = IsolatedDOSBoxSeeker()
    seeker.run()