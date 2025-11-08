"""
🔧 LLM 연결 문제 진단 및 해결 도구
Ollama 서버 상태 확인 및 자동 복구
"""

import requests
import subprocess
import time
import json
import os
import psutil

class LLMConnectionFixer:
    """LLM 연결 문제 자동 해결"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.required_models = [
            "qwen2:0.5b",
            "llama3.2:1b", 
            "qwen2.5-coder:7b"
        ]
        
    def diagnose_connection(self):
        """연결 문제 진단"""
        print("🔍 LLM 연결 상태 진단 중...")
        
        issues = []
        
        # 1. Ollama 프로세스 확인
        print("\n1️⃣ Ollama 프로세스 확인...")
        ollama_running = self.is_ollama_process_running()
        
        if ollama_running:
            print("✅ Ollama 프로세스 실행 중")
        else:
            print("❌ Ollama 프로세스가 실행되지 않음")
            issues.append("ollama_process")
        
        # 2. 서버 응답 확인
        print("\n2️⃣ 서버 응답 확인...")
        server_responsive = self.test_server_response()
        
        if server_responsive:
            print("✅ Ollama 서버 응답 정상")
        else:
            print("❌ Ollama 서버 응답 없음")
            issues.append("server_response")
        
        # 3. 모델 설치 확인
        print("\n3️⃣ 모델 설치 상태 확인...")
        installed_models = self.get_installed_models()
        
        if installed_models:
            print(f"✅ 설치된 모델: {len(installed_models)}개")
            for model in installed_models[:5]:  # 처음 5개만 표시
                print(f"   - {model}")
        else:
            print("❌ 설치된 모델 없음")
            issues.append("no_models")
        
        # 4. 필수 모델 확인
        print("\n4️⃣ 필수 모델 확인...")
        missing_models = []
        
        for model in self.required_models:
            if model in installed_models:
                print(f"✅ {model} 설치됨")
            else:
                print(f"❌ {model} 누락")
                missing_models.append(model)
        
        if missing_models:
            issues.append(("missing_models", missing_models))
        
        # 5. API 테스트
        print("\n5️⃣ API 기능 테스트...")
        api_working = self.test_api_functionality()
        
        if api_working:
            print("✅ API 기능 정상")
        else:
            print("❌ API 기능 오류")
            issues.append("api_error")
        
        return issues
    
    def is_ollama_process_running(self):
        """Ollama 프로세스 실행 확인"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if 'ollama' in proc.info['name'].lower():
                    return True
            return False
        except:
            return False
    
    def test_server_response(self):
        """서버 응답 테스트"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def get_installed_models(self):
        """설치된 모델 목록 가져오기"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
            return []
        except:
            return []
    
    def test_api_functionality(self):
        """API 기능 테스트"""
        try:
            installed_models = self.get_installed_models()
            if not installed_models:
                return False
            
            # 첫 번째 모델로 간단한 테스트
            test_model = installed_models[0]
            
            payload = {
                "model": test_model,
                "prompt": "Say OK",
                "stream": False,
                "options": {"num_predict": 5}
            }
            
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=payload, timeout=10)
            
            return response.status_code == 200
            
        except:
            return False
    
    def fix_issues(self, issues):
        """문제 자동 해결"""
        print("\n🔧 문제 해결 시작...")
        
        for issue in issues:
            if issue == "ollama_process":
                self.start_ollama_server()
            
            elif issue == "server_response":
                self.restart_ollama_server()
            
            elif issue == "no_models":
                self.install_basic_models()
            
            elif isinstance(issue, tuple) and issue[0] == "missing_models":
                missing_models = issue[1]
                self.install_missing_models(missing_models)
            
            elif issue == "api_error":
                self.fix_api_issues()
    
    def start_ollama_server(self):
        """Ollama 서버 시작"""
        print("🚀 Ollama 서버 시작 중...")
        
        try:
            # Windows에서 ollama serve 실행
            subprocess.Popen(["ollama", "serve"], 
                           creationflags=subprocess.CREATE_NO_WINDOW)
            
            print("⏳ 서버 시작 대기 중...")
            time.sleep(5)
            
            # 시작 확인
            if self.test_server_response():
                print("✅ Ollama 서버 시작 완료!")
                return True
            else:
                print("❌ 서버 시작 실패")
                return False
                
        except Exception as e:
            print(f"❌ 서버 시작 오류: {e}")
            return False
    
    def restart_ollama_server(self):
        """Ollama 서버 재시작"""
        print("🔄 Ollama 서버 재시작 중...")
        
        # 기존 프로세스 종료
        self.kill_ollama_processes()
        time.sleep(2)
        
        # 서버 재시작
        return self.start_ollama_server()
    
    def kill_ollama_processes(self):
        """Ollama 프로세스 종료"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if 'ollama' in proc.info['name'].lower():
                    proc.terminate()
            time.sleep(1)
        except:
            pass
    
    def install_basic_models(self):
        """기본 모델 설치"""
        print("📥 기본 모델 설치 중...")
        
        basic_models = ["llama3.2:1b", "qwen2:0.5b"]
        
        for model in basic_models:
            self.install_model(model)
    
    def install_missing_models(self, missing_models):
        """누락된 모델 설치"""
        print(f"📥 누락된 모델 설치: {missing_models}")
        
        for model in missing_models:
            self.install_model(model)
    
    def install_model(self, model_name):
        """개별 모델 설치"""
        print(f"📦 {model_name} 설치 중...")
        
        try:
            result = subprocess.run(["ollama", "pull", model_name], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {model_name} 설치 완료")
                return True
            else:
                print(f"❌ {model_name} 설치 실패: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model_name} 설치 시간 초과")
            return False
        except Exception as e:
            print(f"❌ {model_name} 설치 오류: {e}")
            return False
    
    def fix_api_issues(self):
        """API 문제 해결"""
        print("🔧 API 문제 해결 중...")
        
        # 서버 재시작으로 대부분 해결됨
        return self.restart_ollama_server()
    
    def run_comprehensive_fix(self):
        """종합 문제 해결"""
        print("🛠️ LLM 연결 종합 진단 및 해결")
        print("="*50)
        
        # 1. 진단
        issues = self.diagnose_connection()
        
        if not issues:
            print("\n🎉 모든 것이 정상입니다!")
            return True
        
        print(f"\n❌ 발견된 문제: {len(issues)}개")
        
        # 2. 해결
        self.fix_issues(issues)
        
        # 3. 재검증
        print("\n🔍 해결 후 재검증...")
        time.sleep(3)
        
        remaining_issues = self.diagnose_connection()
        
        if not remaining_issues:
            print("\n🎉 모든 문제가 해결되었습니다!")
            self.test_final_connection()
            return True
        else:
            print(f"\n⚠️ 남은 문제: {len(remaining_issues)}개")
            print("수동 해결이 필요할 수 있습니다.")
            return False
    
    def test_final_connection(self):
        """최종 연결 테스트"""
        print("\n🧪 최종 연결 테스트...")
        
        try:
            installed_models = self.get_installed_models()
            if not installed_models:
                print("❌ 사용 가능한 모델이 없습니다")
                return
            
            # 가장 빠른 모델 선택
            test_model = None
            for preferred in ["qwen2:0.5b", "llama3.2:1b"]:
                if preferred in installed_models:
                    test_model = preferred
                    break
            
            if not test_model:
                test_model = installed_models[0]
            
            print(f"🧪 {test_model}로 테스트 중...")
            
            payload = {
                "model": test_model,
                "prompt": "Hello! Just say 'AI Ready!' in Korean.",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 10
                }
            }
            
            start_time = time.time()
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=payload, timeout=15)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json().get("response", "")
                response_time = end_time - start_time
                
                print(f"✅ 테스트 성공!")
                print(f"   응답: {result[:50]}...")
                print(f"   응답시간: {response_time:.2f}초")
                print(f"   사용 모델: {test_model}")
                
                # 속도 평가
                if response_time < 1.0:
                    print("🚀 초고속 응답!")
                elif response_time < 3.0:
                    print("⚡ 빠른 응답!")
                else:
                    print("🐌 응답이 다소 느림")
                
            else:
                print(f"❌ 테스트 실패: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 테스트 오류: {e}")


def main():
    """메인 실행 함수"""
    
    print("🔧 LLM 연결 문제 해결사")
    print("Ollama 서버 및 모델 상태를 진단하고 자동으로 해결합니다")
    print()
    
    fixer = LLMConnectionFixer()
    
    print("메뉴:")
    print("1. 빠른 진단 (문제만 확인)")
    print("2. 종합 해결 (진단 + 자동 해결)")
    print("3. 서버 재시작만")
    print("4. 기본 모델 설치")
    
    choice = input("\n선택하세요 (1-4, 기본값 2): ").strip() or "2"
    
    if choice == "1":
        issues = fixer.diagnose_connection()
        if issues:
            print(f"\n⚠️ 발견된 문제들: {issues}")
        else:
            print("\n✅ 문제가 발견되지 않았습니다!")
    
    elif choice == "2":
        success = fixer.run_comprehensive_fix()
        if success:
            print("\n🎊 이제 LLM AI를 사용할 준비가 되었습니다!")
            print("   python zero_knowledge_ai.py")
        else:
            print("\n💡 수동 해결 방법:")
            print("1. CMD에서: ollama serve")
            print("2. 다른 CMD에서: ollama pull llama3.2:1b")
    
    elif choice == "3":
        fixer.restart_ollama_server()
    
    elif choice == "4":
        fixer.install_basic_models()
    
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()