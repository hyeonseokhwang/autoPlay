#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
웹 학습 및 키 입력 디버깅 도구
"""

import requests
from bs4 import BeautifulSoup
import win32gui
import win32con
import win32api
import time
import cv2
import numpy as np
import pyautogui

def test_web_scraping():
    """웹 스크래핑 테스트"""
    print("🌐 웹 스크래핑 테스트 중...")
    
    try:
        # 간단한 테스트 사이트
        response = requests.get("https://www.naver.com", timeout=10)
        print(f"✅ 네이버 접속 성공: {response.status_code}")
        
        # 영웅전설 관련 검색 테스트
        search_url = "https://search.naver.com/search.naver?where=blog&query=영웅전설4"
        response = requests.get(search_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 링크 찾기
        links = soup.find_all('a', href=True)
        blog_links = [link['href'] for link in links if 'blog.naver.com' in link.get('href', '') or 'tistory.com' in link.get('href', '')]
        
        print(f"✅ 검색 결과: {len(blog_links)}개 블로그 링크 발견")
        
        if blog_links:
            # 첫 번째 블로그 내용 가져오기 시도
            try:
                blog_response = requests.get(blog_links[0], timeout=10)
                blog_soup = BeautifulSoup(blog_response.text, 'html.parser')
                text_content = blog_soup.get_text()
                
                print(f"✅ 블로그 내용 추출 성공: {len(text_content)} 글자")
                print(f"📝 샘플 내용: {text_content[:200]}...")
                
                # 게임 관련 키워드 검사
                keywords = ['영웅전설', '조작', '키보드', '방향키', '엔터']
                found_keywords = [kw for kw in keywords if kw in text_content]
                print(f"🎯 발견된 키워드: {found_keywords}")
                
            except Exception as e:
                print(f"❌ 블로그 내용 추출 실패: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 웹 스크래핑 테스트 실패: {e}")
        return False

def test_window_control():
    """윈도우 제어 테스트"""
    print("🪟 윈도우 제어 테스트 중...")
    
    try:
        # DOSBox 윈도우 찾기
        def find_dosbox():
            windows = []
            def enum_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if "DOSBox" in title or "dosbox" in title.lower():
                        windows.append((hwnd, title))
                return True
            
            win32gui.EnumWindows(enum_callback, windows)
            return windows
        
        dosbox_windows = find_dosbox()
        
        if dosbox_windows:
            hwnd, title = dosbox_windows[0]
            print(f"✅ DOSBox 발견: {title} (핸들: {hwnd})")
            
            # 윈도우 활성화 테스트
            try:
                win32gui.SetForegroundWindow(hwnd)
                print("✅ DOSBox 활성화 성공")
                
                # 키 입력 테스트
                time.sleep(1)
                
                print("🎯 키 입력 테스트 (3초 후 방향키 입력)...")
                time.sleep(3)
                
                # 방향키 테스트
                keys_to_test = [
                    (win32con.VK_UP, "위"),
                    (win32con.VK_DOWN, "아래"),
                    (win32con.VK_LEFT, "왼쪽"), 
                    (win32con.VK_RIGHT, "오른쪽"),
                    (win32con.VK_RETURN, "엔터")
                ]
                
                for vk_code, name in keys_to_test:
                    print(f"  📤 {name} 키 입력...")
                    win32api.keybd_event(vk_code, 0, 0, 0)
                    time.sleep(0.1)
                    win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
                    time.sleep(0.5)
                
                print("✅ 키 입력 테스트 완료")
                return True
                
            except Exception as e:
                print(f"❌ 키 입력 실패: {e}")
                return False
                
        else:
            print("❌ DOSBox 윈도우를 찾을 수 없습니다")
            print("📋 현재 실행 중인 윈도우:")
            
            def list_windows():
                windows = []
                def enum_callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if title.strip():
                            windows.append(title)
                    return True
                
                win32gui.EnumWindows(enum_callback, windows)
                return windows
            
            all_windows = list_windows()
            for i, window in enumerate(all_windows[:10], 1):
                print(f"  {i}. {window}")
            
            return False
            
    except Exception as e:
        print(f"❌ 윈도우 제어 테스트 실패: {e}")
        return False

def test_screen_capture():
    """화면 캡처 테스트"""
    print("📸 화면 캡처 테스트 중...")
    
    try:
        # 전체 화면 캡처
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        
        print(f"✅ 화면 캡처 성공: {screenshot_np.shape}")
        
        # DOSBox 영역 감지 시도
        gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
        
        # 기본적인 이미지 분석
        height, width = gray.shape
        print(f"📐 화면 크기: {width}x{height}")
        
        # 픽셀 밝기 분포
        brightness_mean = np.mean(gray)
        print(f"💡 평균 밝기: {brightness_mean:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 화면 캡처 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🔧 영웅전설4 AI 디버깅 도구")
    print("=" * 50)
    
    # 1. 웹 스크래핑 테스트
    web_ok = test_web_scraping()
    print()
    
    # 2. 윈도우 제어 테스트
    window_ok = test_window_control()
    print()
    
    # 3. 화면 캡처 테스트
    screen_ok = test_screen_capture()
    print()
    
    # 결과 요약
    print("📊 테스트 결과 요약:")
    print(f"  🌐 웹 스크래핑: {'✅ 성공' if web_ok else '❌ 실패'}")
    print(f"  🪟 윈도우 제어: {'✅ 성공' if window_ok else '❌ 실패'}")
    print(f"  📸 화면 캡처: {'✅ 성공' if screen_ok else '❌ 실패'}")
    
    if not web_ok:
        print("\n🔧 웹 스크래핑 문제 해결 방법:")
        print("  - 인터넷 연결 확인")
        print("  - 방화벽/보안 프로그램 확인")
        print("  - requests 패키지 재설치: pip install --upgrade requests")
    
    if not window_ok:
        print("\n🔧 윈도우 제어 문제 해결 방법:")
        print("  - DOSBox 실행 확인")
        print("  - 관리자 권한으로 실행")
        print("  - pywin32 패키지 재설치: pip install --upgrade pywin32")
    
    if not screen_ok:
        print("\n🔧 화면 캡처 문제 해결 방법:")
        print("  - pyautogui 패키지 재설치: pip install --upgrade pyautogui")
        print("  - opencv-python 재설치: pip install --upgrade opencv-python")

if __name__ == "__main__":
    main()