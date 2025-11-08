"""
LLM 모델 성능 벤치마크 테스트
게임 AI 및 비서 기능을 위한 최적 모델 선택
"""

import time
import requests
import json
import psutil
import sys
from typing import Dict, List, Tuple

class LLMBenchmark:
    """LLM 모델 성능 테스트"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.test_scenarios = self.get_test_scenarios()
        self.results = {}
    
    def get_test_scenarios(self):
        """테스트 시나리오 정의"""
        return {
            "game_strategy": {
                "prompt": """
영웅전설4에서 현재 상황:
- 필드 화면, 캐릭터 레벨 5
- HP: 85/100, MP: 20/30
- 적이 보이지 않음
- 오른쪽에 숲, 왼쪽에 마을

다음 행동을 선택하세요:
1. 숲으로 가서 적 탐색
2. 마을로 돌아가서 회복
3. 현재 위치에서 대기
4. 다른 방향 탐색

선택과 이유를 간단히 설명하세요.
""",
                "expected_keywords": ["숲", "탐색", "회복", "HP", "MP"],
                "weight": 0.4  # 게임 AI에서 40% 비중
            },
            
            "quick_decision": {
                "prompt": "전투 중! 빠른 결정 필요. 공격/방어/도망 중 선택하세요. 한 단어로 답하세요.",
                "expected_keywords": ["공격", "방어", "도망", "attack", "defend", "run"],
                "weight": 0.3  # 반응속도 30% 비중
            },
            
            "korean_conversation": {
                "prompt": """
안녕하세요! 저는 당신의 AI 비서입니다. 
오늘 일정을 관리해드릴까요? 
어떤 도움이 필요하신지 알려주세요.
""",
                "expected_keywords": ["안녕", "비서", "일정", "도움", "관리"],
                "weight": 0.2  # 비서 기능 20% 비중
            },
            
            "logical_reasoning": {
                "prompt": """
다음 패턴을 분석하세요:
전투1: 적3마리 → 승리 → 경험치 150
전투2: 적5마리 → 승리 → 경험치 280  
전투3: 적2마리 → 승리 → 경험치 ?

적 2마리일 때 예상 경험치는?
""",
                "expected_keywords": ["100", "계산", "패턴", "비례"],
                "weight": 0.1  # 논리적 추론 10% 비중
            }
        }
    
    def check_ollama_server(self):
        """Ollama 서버 상태 확인"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                print(f"✅ Ollama 서버 연결됨. 모델 {len(models)}개 발견")
                return models
            else:
                print("❌ Ollama 서버 응답 오류")
                return []
        except Exception as e:
            print(f"❌ Ollama 서버 연결 실패: {e}")
            print("\n🔧 해결 방법:")
            print("1. Ollama 설치: https://ollama.ai/")
            print("2. 서버 시작: ollama serve")
            print("3. 모델 다운로드: ollama pull llama3.2")
            return []
    
    def test_model(self, model_name: str) -> Dict:
        """개별 모델 테스트"""
        print(f"\n🧪 {model_name} 테스트 중...")
        
        results = {
            "model": model_name,
            "scenarios": {},
            "avg_response_time": 0,
            "memory_usage": 0,
            "total_score": 0,
            "errors": []
        }
        
        total_time = 0
        scenario_count = 0
        
        # 메모리 사용량 측정 시작
        initial_memory = psutil.virtual_memory().used
        
        for scenario_name, scenario in self.test_scenarios.items():
            try:
                print(f"  📝 {scenario_name} 테스트...")
                
                # 응답 시간 측정
                start_time = time.time()
                
                response = self.call_ollama(model_name, scenario["prompt"])
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # 품질 평가
                quality_score = self.evaluate_response_quality(
                    response, scenario["expected_keywords"]
                )
                
                # 가중치 적용한 점수
                weighted_score = quality_score * scenario["weight"]
                
                results["scenarios"][scenario_name] = {
                    "response_time": response_time,
                    "quality_score": quality_score,
                    "weighted_score": weighted_score,
                    "response": response[:100] + "..." if len(response) > 100 else response
                }
                
                total_time += response_time
                scenario_count += 1
                
                print(f"    ⏱️ {response_time:.2f}초, 품질: {quality_score:.2f}")
                
            except Exception as e:
                error_msg = f"{scenario_name}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"    ❌ 오류: {error_msg}")
        
        # 최종 메모리 사용량
        final_memory = psutil.virtual_memory().used
        results["memory_usage"] = (final_memory - initial_memory) / (1024**2)  # MB
        
        # 평균 응답 시간
        if scenario_count > 0:
            results["avg_response_time"] = total_time / scenario_count
        
        # 총점 계산
        results["total_score"] = sum(
            scenario["weighted_score"] 
            for scenario in results["scenarios"].values()
        ) * 100  # 100점 만점으로 스케일링
        
        return results
    
    def call_ollama(self, model_name: str, prompt: str, timeout: int = 30) -> str:
        """Ollama API 호출"""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9
            }
        }
        
        response = requests.post(
            f"{self.ollama_url}/api/generate", 
            json=payload, 
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"API 오류: {response.status_code}")
    
    def evaluate_response_quality(self, response: str, expected_keywords: List[str]) -> float:
        """응답 품질 평가"""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        
        # 키워드 매칭 점수
        keyword_matches = sum(
            1 for keyword in expected_keywords 
            if keyword.lower() in response_lower
        )
        keyword_score = keyword_matches / len(expected_keywords)
        
        # 응답 길이 점수 (너무 짧거나 길면 감점)
        length_score = min(1.0, max(0.1, len(response) / 100))
        
        # 한국어 응답 보너스
        korean_chars = sum(1 for char in response if ord(char) > 127)
        korean_bonus = min(0.2, korean_chars / len(response)) if response else 0
        
        total_score = (keyword_score * 0.6) + (length_score * 0.3) + korean_bonus
        return min(1.0, total_score)
    
    def run_benchmark(self, models_to_test: List[str] = None):
        """벤치마크 실행"""
        
        print("🚀 LLM 모델 성능 벤치마크 시작!")
        print("=" * 50)
        
        # 서버 확인
        available_models = self.check_ollama_server()
        if not available_models:
            return
        
        # 테스트할 모델 결정
        if models_to_test is None:
            models_to_test = available_models
        else:
            # 사용자 지정 모델이 설치되어 있는지 확인
            models_to_test = [m for m in models_to_test if m in available_models]
        
        if not models_to_test:
            print("❌ 테스트할 모델이 없습니다!")
            return
        
        print(f"\n📋 테스트 대상: {models_to_test}")
        
        # 각 모델 테스트
        for model in models_to_test:
            self.results[model] = self.test_model(model)
        
        # 결과 분석 및 출력
        self.analyze_results()
        self.save_results()
    
    def analyze_results(self):
        """결과 분석 및 출력"""
        
        print("\n" + "="*60)
        print("📊 벤치마크 결과 분석")
        print("="*60)
        
        if not self.results:
            print("❌ 테스트 결과가 없습니다.")
            return
        
        # 모델별 점수 정렬
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True
        )
        
        print("\n🏆 종합 순위:")
        for i, (model, result) in enumerate(sorted_models, 1):
            score = result["total_score"]
            time_avg = result["avg_response_time"]
            memory = result["memory_usage"]
            
            print(f"{i}. {model}")
            print(f"   📈 종합점수: {score:.1f}/100")
            print(f"   ⏱️ 평균응답: {time_avg:.2f}초")
            print(f"   💾 메모리: {memory:.1f}MB")
            
            if result["errors"]:
                print(f"   ⚠️ 오류: {len(result['errors'])}개")
            print()
        
        # 카테고리별 최고 모델
        print("\n🎯 카테고리별 최고 성능:")
        
        categories = {
            "빠른 반응": ("avg_response_time", False),  # 낮을수록 좋음
            "게임 전략": ("scenarios.game_strategy.quality_score", True),
            "한국어 대화": ("scenarios.korean_conversation.quality_score", True),
            "논리적 추론": ("scenarios.logical_reasoning.quality_score", True)
        }
        
        for category, (metric, higher_better) in categories.items():
            try:
                best_model = self.find_best_in_category(metric, higher_better)
                if best_model:
                    model_name, value = best_model
                    print(f"   {category}: {model_name} ({value:.2f})")
            except:
                print(f"   {category}: 데이터 없음")
        
        # 추천 모델
        print("\n💡 추천:")
        self.recommend_models(sorted_models)
    
    def find_best_in_category(self, metric: str, higher_better: bool = True):
        """카테고리별 최고 모델 찾기"""
        valid_results = []
        
        for model, result in self.results.items():
            try:
                # 중첩된 딕셔너리 접근
                value = result
                for key in metric.split('.'):
                    value = value[key]
                
                valid_results.append((model, value))
            except (KeyError, TypeError):
                continue
        
        if not valid_results:
            return None
        
        return max(valid_results, key=lambda x: x[1] if higher_better else -x[1])
    
    def recommend_models(self, sorted_models):
        """사용 목적별 모델 추천"""
        
        if len(sorted_models) == 0:
            print("   추천할 모델이 없습니다.")
            return
        
        # 1위 모델 (종합)
        best_overall = sorted_models[0]
        print(f"   🥇 종합 최고: {best_overall[0]}")
        
        # 빠른 응답이 필요한 경우
        fast_models = sorted(
            [(m, r) for m, r in self.results.items() if r["avg_response_time"] > 0],
            key=lambda x: x[1]["avg_response_time"]
        )
        
        if fast_models:
            print(f"   ⚡ 실시간 게임용: {fast_models[0][0]} ({fast_models[0][1]['avg_response_time']:.2f}초)")
        
        # 메모리 효율적인 모델
        memory_efficient = sorted(
            [(m, r) for m, r in self.results.items() if r["memory_usage"] > 0],
            key=lambda x: x[1]["memory_usage"]
        )
        
        if memory_efficient:
            print(f"   💾 메모리 효율: {memory_efficient[0][0]} ({memory_efficient[0][1]['memory_usage']:.1f}MB)")
    
    def save_results(self):
        """결과를 JSON 파일로 저장"""
        filename = f"llm_benchmark_{int(time.time())}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 결과가 {filename}에 저장되었습니다.")
        
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")


def main():
    """메인 실행 함수"""
    
    print("🧠 LLM 모델 성능 벤치마크")
    print("게임 AI 및 AI 비서를 위한 최적 모델 선택")
    print()
    
    benchmark = LLMBenchmark()
    
    # 사용자가 원하는 모델 목록 (없으면 설치된 모든 모델 테스트)
    preferred_models = [
        "llama3.2:1b",
        "llama3.2:3b", 
        "qwen2.5-coder:7b",
        "deepseek-coder:6.7b",
        "llava:7b"
    ]
    
    print("테스트 시나리오:")
    for name, scenario in benchmark.test_scenarios.items():
        print(f"  - {name} (가중치: {scenario['weight']*100:.0f}%)")
    
    print(f"\n우선 테스트할 모델: {preferred_models}")
    print("\n시작하려면 Enter를 누르세요 (또는 Ctrl+C로 종료)...")
    
    try:
        input()
        benchmark.run_benchmark(preferred_models)
        
        print("\n🎉 벤치마크 완료!")
        print("\n다음 단계:")
        print("1. 결과를 바탕으로 adaptive_hero_ai.py 수정")
        print("2. 선택된 모델로 게임 AI 테스트")
        print("3. 성능에 따라 추가 모델 다운로드")
        
    except KeyboardInterrupt:
        print("\n👋 벤치마크가 취소되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()