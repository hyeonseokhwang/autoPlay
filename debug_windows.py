"""
윈도우 디버그 도구 - 현재 실행 중인 윈도우 확인
"""
import win32gui

def list_all_windows():
    """현재 실행 중인 모든 윈도우 나열"""
    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            
            if window_text.strip():  # 제목이 있는 윈도우만
                windows.append((hwnd, window_text, class_name))
        return True
    
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    
    print("=" * 80)
    print("현재 실행 중인 모든 윈도우:")
    print("=" * 80)
    
    for i, (hwnd, title, class_name) in enumerate(windows, 1):
        print(f"{i:2d}. 제목: '{title}'")
        print(f"    클래스: '{class_name}'")
        print(f"    핸들: {hwnd}")
        
        # DOSBox 관련 키워드 체크
        keywords = ["DOSBox", "dosbox", "DOSBOX", "ED4", "영웅전설", "SVN-Daum"]
        found_keywords = []
        for keyword in keywords:
            if keyword in title or keyword in class_name:
                found_keywords.append(keyword)
        
        if found_keywords:
            print(f"    ⭐ DOSBox 관련 키워드 발견: {found_keywords}")
        
        print()
    
    print(f"총 {len(windows)}개의 윈도우 발견")
    
    # DOSBox 후보 찾기
    dosbox_candidates = []
    for hwnd, title, class_name in windows:
        keywords = ["DOSBox", "dosbox", "DOSBOX", "ED4", "영웅전설", "SVN-Daum"]
        for keyword in keywords:
            if keyword in title or keyword in class_name:
                dosbox_candidates.append((hwnd, title, class_name))
                break
    
    if dosbox_candidates:
        print("\n🎯 DOSBox 후보들:")
        for hwnd, title, class_name in dosbox_candidates:
            print(f"   - '{title}' (핸들: {hwnd})")
    else:
        print("\n❌ DOSBox 관련 윈도우를 찾을 수 없습니다.")

if __name__ == "__main__":
    list_all_windows()