@echo off
echo 🧠 Ollama LLM 설치 및 설정 스크립트
echo.

REM Ollama가 설치되어 있는지 확인
ollama --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Ollama가 이미 설치되어 있습니다!
    goto :setup_models
)

echo 📥 Ollama 다운로드 및 설치 중...
echo 브라우저가 열리면 Windows용 Ollama를 다운로드하여 설치하세요.
echo.
start https://ollama.ai/download

echo 설치 완료 후 Enter를 누르세요...
pause >nul

:setup_models
echo.
echo 🤖 AI 모델 다운로드 중...

REM DeepSeek Coder 모델 (추천 - 코딩 특화)
echo 1. DeepSeek Coder 다운로드 중... (약 4GB)
ollama pull deepseek-coder

REM Llama 3.2 모델 (범용)
echo 2. Llama 3.2 다운로드 중... (약 2GB)  
ollama pull llama3.2

REM Qwen2 모델 (가벼움)
echo 3. Qwen2 다운로드 중... (약 1.5GB)
ollama pull qwen2

echo.
echo 📋 설치된 모델 확인:
ollama list

echo.
echo 🚀 Ollama 서버 시작 중...
start /B ollama serve

echo.
echo ✅ 설정 완료!
echo.
echo 🎮 이제 적응형 AI를 실행하세요:
echo    python adaptive_hero_ai.py
echo.
echo 💡 팁: 
echo   - Ctrl+C로 서버 종료
echo   - ollama list: 모델 목록 확인
echo   - ollama run 모델명: 직접 채팅
echo.
pause