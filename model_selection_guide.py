"""
🧠 최적 LLM 모델 선택 가이드
용도별/성능별 추천 모델 및 설치 스크립트
"""

class ModelRecommendations:
    """모델 추천 시스템"""
    
    @staticmethod
    def get_recommendations():
        """사용 목적별 모델 추천"""
        
        return {
            "게임_ai_초보자": {
                "모델": "llama3.2:3b",
                "이유": "빠른 속도, 적은 메모리, 안정적 성능",
                "설치": "ollama pull llama3.2:3b",
                "메모리": "4GB",
                "속도": "1-2초",
                "품질": "⭐⭐⭐⭐",
                "적합한_게임": ["턴제 RPG", "전략 게임", "퍼즐 게임"]
            },
            
            "게임_ai_고성능": {
                "모델": "qwen2.5-coder:7b", 
                "이유": "뛰어난 추론능력, 패턴인식 특화, 한국어 지원",
                "설치": "ollama pull qwen2.5-coder:7b",
                "메모리": "8GB",
                "속도": "2-4초",
                "품질": "⭐⭐⭐⭐⭐",
                "적합한_게임": ["복잡한 RPG", "전략 시뮬레이션", "퍼즐 어드벤처"]
            },
            
            "실시간_게임": {
                "모델": "llama3.2:1b",
                "이유": "초고속 반응, 최소 메모리, 실시간 처리",
                "설치": "ollama pull llama3.2:1b", 
                "메모리": "2GB",
                "속도": "0.5-1초",
                "품질": "⭐⭐⭐",
                "적합한_게임": ["액션 게임", "FPS", "리듬 게임"]
            },
            
            "이미지_분석": {
                "모델": "llava:7b",
                "이유": "멀티모달, 화면 직접 분석, 시각적 이해",
                "설치": "ollama pull llava:7b",
                "메모리": "12GB",
                "속도": "5-8초",
                "품질": "⭐⭐⭐⭐⭐",
                "적합한_게임": ["모든 게임 (화면 분석)"]
            },
            
            "ai_비서_기본": {
                "모델": "qwen2.5-coder:7b",
                "이유": "범용성, 코딩능력, 한국어, 확장성",
                "설치": "ollama pull qwen2.5-coder:7b",
                "메모리": "8GB", 
                "속도": "2-4초",
                "품질": "⭐⭐⭐⭐⭐",
                "적합한_작업": ["일정관리", "코딩도움", "질문응답", "창작"]
            },
            
            "ai_비서_고급": {
                "모델": "qwen2.5-coder:14b",
                "이유": "최고 성능, 복잡한 추론, 전문적 대화",
                "설치": "ollama pull qwen2.5-coder:14b",
                "메모리": "16GB",
                "속도": "4-8초", 
                "품질": "⭐⭐⭐⭐⭐",
                "적합한_작업": ["복잡한 분석", "전문 상담", "고급 프로그래밍"]
            },
            
            "한국어_특화": {
                "모델": "eeve-korean:10.8b",
                "이유": "한국어 완벽지원, 문화적 맥락, 자연스러운 대화",
                "설치": "ollama pull eeve-korean:10.8b",
                "메모리": "12GB",
                "속도": "3-6초",
                "품질": "⭐⭐⭐⭐⭐",
                "적합한_작업": ["한국어 대화", "번역", "창작", "교육"]
            }
        }

def print_recommendations():
    """추천 모델 출력"""
    
    recs = ModelRecommendations.get_recommendations()
    
    print("🎯 용도별 최적 모델 추천")
    print("=" * 60)
    
    for category, info in recs.items():
        print(f"\n📋 {category.replace('_', ' ').title()}")
        print(f"   🤖 모델: {info['모델']}")
        print(f"   💡 이유: {info['이유']}")
        print(f"   💾 메모리: {info['메모리']}")
        print(f"   ⏱️ 속도: {info['속도']}")
        print(f"   ⭐ 품질: {info['품질']}")
        print(f"   📥 설치: {info['설치']}")
        
        if "적합한_게임" in info:
            games = ", ".join(info["적합한_게임"])
            print(f"   🎮 게임: {games}")
        
        if "적합한_작업" in info:
            tasks = ", ".join(info["적합한_작업"])
            print(f"   💼 작업: {tasks}")

def create_install_script():
    """모델 설치 스크립트 생성"""
    
    recs = ModelRecommendations.get_recommendations()
    
    # 단계별 설치 스크립트
    scripts = {
        "기본팩": [
            "llama3.2:3b",      # 빠른 기본 모델
            "qwen2.5-coder:7b"  # 고성능 메인 모델
        ],
        
        "게임팩": [
            "llama3.2:1b",      # 실시간용
            "llama3.2:3b",      # 기본용  
            "qwen2.5-coder:7b", # 전략용
            "llava:7b"          # 이미지 분석용
        ],
        
        "비서팩": [
            "qwen2.5-coder:7b",   # 기본 비서
            "qwen2.5-coder:14b",  # 고급 비서  
            "eeve-korean:10.8b"   # 한국어 특화
        ],
        
        "풀팩": [
            "llama3.2:1b",
            "llama3.2:3b", 
            "qwen2.5-coder:7b",
            "qwen2.5-coder:14b",
            "deepseek-coder:6.7b",
            "llava:7b",
            "eeve-korean:10.8b"
        ]
    }
    
    return scripts

# 인터랙티브 모델 선택기
def interactive_model_selection():
    """사용자와 대화하며 최적 모델 선택"""
    
    print("🤖 AI 모델 맞춤 추천 시스템")
    print("=" * 40)
    
    # 사용 목적 질문
    print("\n1️⃣ 주요 사용 목적은?")
    print("   1. 게임 자동 플레이")
    print("   2. AI 비서 (일정, 질문답변)")
    print("   3. 둘 다")
    
    purpose = input("선택 (1-3): ").strip()
    
    # 성능 vs 속도 선호도
    print("\n2️⃣ 더 중요한 것은?") 
    print("   1. 빠른 속도 (1초 내)")
    print("   2. 높은 품질 (3-5초)")
    print("   3. 균형")
    
    priority = input("선택 (1-3): ").strip()
    
    # 시스템 사양
    print("\n3️⃣ 시스템 메모리는?")
    print("   1. 8GB 이하")
    print("   2. 16GB")  
    print("   3. 32GB 이상")
    
    memory = input("선택 (1-3): ").strip()
    
    # 추천 생성
    recommendations = []
    
    # 게임 용도
    if purpose in ["1", "3"]:
        if priority == "1":  # 속도 중시
            recommendations.append("llama3.2:1b")
        elif priority == "2":  # 품질 중시
            recommendations.append("qwen2.5-coder:7b")
        else:  # 균형
            recommendations.append("llama3.2:3b")
    
    # 비서 용도
    if purpose in ["2", "3"]:
        if memory == "1":  # 8GB 이하
            recommendations.append("qwen2.5-coder:7b")
        elif memory == "3":  # 32GB 이상
            recommendations.append("qwen2.5-coder:14b")
            recommendations.append("eeve-korean:10.8b")
        else:  # 16GB
            recommendations.append("qwen2.5-coder:7b")
    
    # 이미지 분석 (게임용)
    if purpose in ["1", "3"] and memory in ["2", "3"]:
        recommendations.append("llava:7b")
    
    # 중복 제거
    recommendations = list(set(recommendations))
    
    print(f"\n🎯 당신에게 최적인 모델:")
    for i, model in enumerate(recommendations, 1):
        print(f"   {i}. {model}")
    
    print(f"\n📥 설치 명령어:")
    for model in recommendations:
        print(f"   ollama pull {model}")
    
    return recommendations

def estimate_requirements(models):
    """모델별 시스템 요구사항 계산"""
    
    requirements = {
        "llama3.2:1b": {"ram": 2, "time": 0.5},
        "llama3.2:3b": {"ram": 4, "time": 1.5},
        "qwen2.5-coder:7b": {"ram": 8, "time": 3},
        "qwen2.5-coder:14b": {"ram": 16, "time": 6},
        "deepseek-coder:6.7b": {"ram": 8, "time": 3.5},
        "llava:7b": {"ram": 12, "time": 7},
        "eeve-korean:10.8b": {"ram": 12, "time": 5}
    }
    
    total_ram = sum(requirements.get(model, {"ram": 4})["ram"] for model in models)
    avg_time = sum(requirements.get(model, {"time": 2})["time"] for model in models) / len(models)
    
    return {
        "total_ram_needed": total_ram,
        "avg_response_time": avg_time,
        "recommended_ram": total_ram + 4,  # 시스템 여유분
        "disk_space": len(models) * 3.5,  # 모델당 평균 3.5GB
    }

if __name__ == "__main__":
    print("🧠 LLM 모델 선택 가이드")
    print()
    
    while True:
        print("\n메뉴:")
        print("1. 용도별 추천 모델 보기")
        print("2. 맞춤 모델 추천받기")  
        print("3. 설치 스크립트 보기")
        print("4. 종료")
        
        choice = input("\n선택하세요 (1-4): ").strip()
        
        if choice == "1":
            print_recommendations()
            
        elif choice == "2":
            selected_models = interactive_model_selection()
            reqs = estimate_requirements(selected_models)
            
            print(f"\n💻 시스템 요구사항:")
            print(f"   RAM: {reqs['recommended_ram']}GB 권장")
            print(f"   디스크: {reqs['disk_space']:.1f}GB")
            print(f"   평균 응답시간: {reqs['avg_response_time']:.1f}초")
            
        elif choice == "3":
            scripts = create_install_script()
            
            print("\n📦 설치 패키지:")
            for package, models in scripts.items():
                print(f"\n{package}:")
                for model in models:
                    print(f"   ollama pull {model}")
                    
        elif choice == "4":
            print("👋 좋은 AI 개발 되세요!")
            break
            
        else:
            print("❌ 잘못된 선택입니다.")