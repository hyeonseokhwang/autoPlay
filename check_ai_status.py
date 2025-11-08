"""
빠른 AI 상태 체크
"""
import requests
import win32gui
import time

def check_ai_status():
    print("🔍 AI 상태 체크 중...")
    
    # 1. LLM 서버 확인
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ LLM 서버 응답 정상")
        else:
            print("❌ LLM 서버 문제")
    except:
        print("❌ LLM 서버 연결 안됨")
    
    # 2. DOSBox 창 확인
    dosbox_found = False
    def enum_windows_proc(hwnd, lparam):
        nonlocal dosbox_found
        window_text = win32gui.GetWindowText(hwnd)
        if 'dosbox' in window_text.lower():
            print(f"✅ DOSBox 창 발견: {window_text}")
            dosbox_found = True
        return True
    
    win32gui.EnumWindows(enum_windows_proc, 0)
    
    if not dosbox_found:
        print("❌ DOSBox 창을 찾을 수 없음")
    
    # 3. AI 프로세스 확인
    import psutil
    ai_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and 'rag_autonomous_ai.py' in str(proc.info['cmdline']):
                ai_processes.append(proc.info['pid'])
        except:
            pass
    
    if ai_processes:
        print(f"✅ RAG AI 프로세스 실행 중: PID {ai_processes}")
    else:
        print("❌ RAG AI 프로세스 없음")

if __name__ == "__main__":
    check_ai_status()