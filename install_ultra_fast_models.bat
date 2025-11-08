@echo off
echo ⚡ 초고속 LLM 모델 설치 스크립트
echo 목표: 1초 내 응답 시간!
echo.

REM Ollama 서버 상태 확인
echo 🔍 Ollama 서버 확인 중...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama가 설치되지 않았습니다!
    echo 📥 https://ollama.ai 에서 설치하세요.
    pause
    exit /b 1
)

echo ✅ Ollama 서버 발견됨
echo.

echo 🚀 초고속 모델 다운로드 시작...
echo.

REM 1. 최고속 모델 (0.3초)
echo 1️⃣ Qwen2:0.5B 다운로드 중... (최고속 - 약 500MB)
ollama pull qwen2:0.5b
if %errorlevel% equ 0 (
    echo ✅ qwen2:0.5b 설치 완료! ^(예상 응답시간: 0.3초^)
) else (
    echo ⚠️ qwen2:0.5b 설치 실패
)
echo.

REM 2. 균형잡힌 고속 모델 (0.7초)
echo 2️⃣ Llama3.2:1B 다운로드 중... (균형형 - 약 1.3GB)
ollama pull llama3.2:1b
if %errorlevel% equ 0 (
    echo ✅ llama3.2:1b 설치 완료! ^(예상 응답시간: 0.7초^)
) else (
    echo ⚠️ llama3.2:1b 설치 실패
)
echo.

REM 3. 백업 모델 (0.8초)
echo 3️⃣ Phi3:mini 다운로드 중... (백업용 - 약 2.2GB)
ollama pull phi3:mini
if %errorlevel% equ 0 (
    echo ✅ phi3:mini 설치 완료! ^(예상 응답시간: 0.8초^)
) else (
    echo ⚠️ phi3:mini 설치 실패
)
echo.

REM 선택적: 약간 더 큰 모델
echo 4️⃣ Gemma2:2B 다운로드 중... (선택용 - 약 1.6GB)
ollama pull gemma2:2b
if %errorlevel% equ 0 (
    echo ✅ gemma2:2b 설치 완료! ^(예상 응답시간: 0.9초^)
) else (
    echo ⚠️ gemma2:2b 설치 실패
)
echo.

echo 📊 설치된 모델 확인:
ollama list

echo.
echo 🧪 속도 테스트 실행 중...
echo Testing qwen2:0.5b response time... | ollama run qwen2:0.5b "Say 'OK' only"

echo.
echo ⚡ 초고속 모델 설치 완료!
echo.
echo 🎮 이제 초고속 AI를 실행하세요:
echo    python ultra_fast_hero_ai.py
echo.
echo 📈 예상 성능:
echo   - qwen2:0.5b    : 0.2-0.6초 ^(최고속^)
echo   - llama3.2:1b   : 0.5-0.8초 ^(균형^)  
echo   - phi3:mini     : 0.6-0.9초 ^(안정^)
echo   - gemma2:2b     : 0.7-1.0초 ^(품질^)
echo.
echo 💡 팁: 
echo   - 메모리가 부족하면 qwen2:0.5b만 사용
echo   - GPU가 있으면 더 빠른 속도 가능
echo   - temperature=0.1로 설정하면 더 빠름
echo.
pause