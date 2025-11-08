"""
화면 캡처 모듈
도스박스 윈도우를 찾아서 실시간으로 화면을 캡처합니다.
"""

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
from PIL import Image
import mss
import time


class ScreenCapture:
    def __init__(self, window_title="DOSBox"):
        """
        화면 캡처 클래스 초기화
        
        Args:
            window_title (str): 캡처할 윈도우 제목
        """
        self.window_title = window_title
        self.hwnd = None
        self.sct = mss.mss()
        self.window_rect = None
        
    def find_window(self):
        """
        DOSBox 윈도우를 찾습니다.
        
        Returns:
            bool: 윈도우를 찾았으면 True, 아니면 False
        """
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if self.window_title.lower() in window_text.lower():
                    windows.append((hwnd, window_text))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            self.hwnd = windows[0][0]
            self._update_window_rect()
            print(f"윈도우 발견: {windows[0][1]}")
            return True
        else:
            print(f"'{self.window_title}' 윈도우를 찾을 수 없습니다.")
            return False
    
    def _update_window_rect(self):
        """윈도우 위치와 크기 정보를 업데이트합니다."""
        if self.hwnd:
            rect = win32gui.GetWindowRect(self.hwnd)
            self.window_rect = {
                "top": rect[1],
                "left": rect[0], 
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1]
            }
    
    def capture_screen(self):
        """
        현재 화면을 캡처합니다.
        
        Returns:
            numpy.ndarray: 캡처된 이미지 (BGR 포맷)
        """
        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            if not self.find_window():
                return None
        
        try:
            # 윈도우 위치 업데이트
            self._update_window_rect()
            
            # mss를 사용한 빠른 스크린샷
            screenshot = self.sct.grab(self.window_rect)
            
            # PIL Image로 변환 후 numpy array로 변환
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            img_array = np.array(img)
            
            # BGR로 변환 (OpenCV 형식)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_bgr
            
        except Exception as e:
            print(f"화면 캡처 중 오류 발생: {e}")
            return None
    
    def capture_region(self, x, y, width, height):
        """
        화면의 특정 영역을 캡처합니다.
        
        Args:
            x, y (int): 캡처 영역의 좌상단 좌표
            width, height (int): 캡처 영역의 크기
            
        Returns:
            numpy.ndarray: 캡처된 이미지
        """
        full_image = self.capture_screen()
        if full_image is not None:
            return full_image[y:y+height, x:x+width]
        return None
    
    def save_screenshot(self, filename="screenshot.png"):
        """
        현재 화면을 파일로 저장합니다.
        
        Args:
            filename (str): 저장할 파일명
        """
        image = self.capture_screen()
        if image is not None:
            cv2.imwrite(filename, image)
            print(f"스크린샷 저장됨: {filename}")
        else:
            print("스크린샷 저장 실패")


# 테스트 코드
if __name__ == "__main__":
    capture = ScreenCapture()
    
    if capture.find_window():
        print("5초 후 스크린샷을 촬영합니다...")
        time.sleep(5)
        
        # 스크린샷 촬영
        img = capture.capture_screen()
        if img is not None:
            # 화면에 표시
            cv2.imshow("DOSBox 캡처", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # 파일로 저장
            capture.save_screenshot("test_capture.png")
    else:
        print("DOSBox 윈도우를 먼저 실행해주세요.")