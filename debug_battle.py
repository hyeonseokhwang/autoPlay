"""
전투 감지 디버그 도구 - 현재 화면 상태 분석
"""

import win32gui
import win32ui
import win32con
import cv2
import numpy as np
from PIL import Image

def find_dosbox():
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
        return windows[0][0]
    return None

def capture_dosbox(hwnd):
    """DOSBox 화면 캡처"""
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        
        hwndDC = win32gui.GetWindowDC(hwnd)
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
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        return np.array(img)
        
    except Exception as e:
        print(f"캡처 오류: {e}")
        return None

def analyze_battle_detection(image):
    """전투 감지 상세 분석"""
    if image is None:
        print("❌ 이미지가 없습니다.")
        return
    
    print(f"📊 이미지 크기: {image.shape}")
    
    # BGR 변환
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        print("❌ 올바르지 않은 이미지 형식")
        return
    
    height, width = image_bgr.shape[:2]
    
    # 게임 영역 추출 (DOSBox 테두리 제외)
    game_area = image_bgr[30:height-10, 10:width-10]
    game_height, game_width = game_area.shape[:2]
    
    print(f"🎮 게임 영역 크기: {game_width}x{game_height}")
    
    # 1. UI 영역 분석 (하단 30%)
    bottom_area = game_area[int(game_height * 0.7):, :]
    print(f"🔍 하단 UI 영역 크기: {bottom_area.shape[1]}x{bottom_area.shape[0]}")
    
    hsv_bottom = cv2.cvtColor(bottom_area, cv2.COLOR_BGR2HSV)
    
    # 갈색/오렌지 UI 감지
    brown_lower = np.array([10, 50, 50])
    brown_upper = np.array([25, 255, 255])
    brown_mask = cv2.inRange(hsv_bottom, brown_lower, brown_upper)
    
    brown_pixels = cv2.countNonZero(brown_mask)
    total_bottom_pixels = bottom_area.shape[0] * bottom_area.shape[1]
    ui_ratio = brown_pixels / total_bottom_pixels if total_bottom_pixels > 0 else 0
    
    print(f"📋 UI 비율: {ui_ratio:.3f} ({brown_pixels}/{total_bottom_pixels} 픽셀)")
    print(f"📋 UI 임계값: 0.15 (15%)")
    print(f"📋 UI 판정: {'✅ UI 있음' if ui_ratio > 0.15 else '❌ UI 없음'}")
    
    # 2. 캐릭터/몬스터 영역 분석 (중앙 상단)
    char_area = game_area[int(game_height * 0.1):int(game_height * 0.6), 
                         int(game_width * 0.1):int(game_width * 0.9)]
    
    print(f"👥 캐릭터 영역 크기: {char_area.shape[1]}x{char_area.shape[0]}")
    
    if char_area.shape[0] < 50 or char_area.shape[1] < 50:
        print("❌ 캐릭터 영역이 너무 작음")
        return
        
    hsv_char = cv2.cvtColor(char_area, cv2.COLOR_BGR2HSV)
    
    # 다양한 색상으로 캐릭터 감지
    color_ranges = [
        ("어두운색", [0, 0, 0], [180, 255, 100]),
        ("빨간색1", [0, 100, 100], [10, 255, 255]),
        ("빨간색2", [160, 100, 100], [180, 255, 255]),
        ("파란색/보라색", [100, 100, 100], [160, 255, 255])
    ]
    
    character_found = False
    total_candidates = 0
    
    for color_name, lower, upper in color_ranges:
        mask = cv2.inRange(hsv_char, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = 0
        valid_characters = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 3000:  # 크기 필터
                candidates += 1
                x, y, w, h = cv2.boundingRect(contour)
                ratio = w/h
                if 0.2 < ratio < 5.0 and w > 10 and h > 10:
                    valid_characters += 1
                    character_found = True
        
        total_candidates += candidates
        print(f"🎨 {color_name}: {candidates}개 후보, {valid_characters}개 유효 캐릭터")
    
    print(f"👥 캐릭터 감지 결과: {'✅ 캐릭터 있음' if character_found else '❌ 캐릭터 없음'}")
    print(f"👥 총 후보 객체: {total_candidates}개")
    
    # 최종 전투 판정
    is_battle = ui_ratio > 0.15 and character_found
    print(f"\n⚔️ 최종 전투 판정: {'✅ 전투 화면' if is_battle else '❌ 일반 화면'}")
    
    # 개선 제안
    print(f"\n💡 개선 제안:")
    if ui_ratio > 0.15 and not character_found:
        print("   - UI는 있지만 캐릭터가 없음 → 전투 대기 화면이거나 메뉴 화면일 가능성")
        print("   - 캐릭터 감지 조건을 완화하거나 다른 특징 추가 필요")
    elif character_found and ui_ratio <= 0.15:
        print("   - 캐릭터는 있지만 UI가 부족 → 필드 화면일 가능성")
        print("   - UI 감지 조건을 완화하거나 다른 UI 요소 확인 필요")
    elif ui_ratio > 0.15 and character_found:
        print("   - 조건을 모두 만족하지만 잘못 인식됨 → 조건을 더 엄격하게 할 필요")
    else:
        print("   - 현재 설정이 적절함")

def main():
    print("=" * 60)
    print("   DOSBox 전투 감지 디버그 도구")
    print("=" * 60)
    
    hwnd = find_dosbox()
    if not hwnd:
        print("❌ DOSBox 윈도우를 찾을 수 없습니다.")
        return
    
    print(f"✓ DOSBox 윈도우 발견 (핸들: {hwnd})")
    
    image = capture_dosbox(hwnd)
    if image is None:
        print("❌ 화면 캡처 실패")
        return
    
    print("✓ 화면 캡처 성공")
    
    # 분석 시작
    analyze_battle_detection(image)
    
    # 화면 저장
    cv2.imwrite("debug_screen.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"\n💾 현재 화면이 'debug_screen.png'로 저장되었습니다.")

if __name__ == "__main__":
    main()